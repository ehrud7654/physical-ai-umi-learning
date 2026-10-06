"""Shared-machine etiquette, enforced in code instead of pasted into commands.
공유 머신 규약을 명령에 붙여 쓰는 대신 코드로 강제한다.

UMI's rule is **one allocated GPU, one job**. On 2026-09-07 that was broken three
separate ways in one afternoon, all by hand-written command lines 🟢:

- `MUJOCO_GL=egl` was omitted once and the job died on an OpenGL error
- the same `probe_freeze` command was launched **four times** (queued input from a
  busy terminal), all four writing to the same `>` log, making the log unusable
- `CUDA_VISIBLE_DEVICES=1` took a second GPU, and the server operators warned that
  the machine was fully loaded and jobs would be killed
- every process span BLAS threads equal to the core count (80), so six processes
  produced load average 165 on 80 cores

UMI 규약은 **할당 GPU 한 장, 잡 하나**다. 2026-09-07 하루 오후에 그것이 세 가지로
깨졌고 전부 손으로 쓴 명령줄 때문이었다 🟢 — `MUJOCO_GL=egl` 누락으로 OpenGL 에러,
같은 `probe_freeze` 를 **4번** 실행(바쁜 터미널에 큐잉된 입력)해서 넷이 같은 로그에
써서 로그가 무효, `CUDA_VISIBLE_DEVICES=1` 로 두 번째 장을 잡아 서버 측 경고,
그리고 프로세스마다 BLAS 스레드를 코어 수(80)만큼 띄워 6개에서 load average 165.

None of those are knowledge failures -- the rules were known and written down. They
are failures of a workflow that re-types the rules every time. So this module is
imported **first**, before numpy or torch, and does it once.
넷 다 지식의 실패가 아니다 — 규칙은 알고 있었고 적혀 있었다. 규칙을 매번 다시
타이핑하는 작업방식의 실패다. 그래서 이 모듈을 numpy·torch **보다 먼저** 임포트해
한 번에 처리한다.

    import runtime_limits          # noqa: F401  — 반드시 numpy/torch 앞
    runtime_limits.claim("probe_freeze")   # 중복 실행이면 여기서 종료
"""

from __future__ import annotations

import atexit
import os
import platform
import sys
from pathlib import Path

# Threads. These must be set before numpy/torch load their BLAS backend, which is
# why this module has to be imported first -- setting them later is silently
# ignored and the process still takes every core.
# 스레드. numpy/torch 가 BLAS 백엔드를 로드하기 전에 정해져야 하므로 이 모듈이
# 먼저 임포트돼야 한다. 나중에 설정하면 조용히 무시되고 코어를 다 먹는다.
DEFAULT_THREADS = os.environ.get("AI_THREADS", "2")
_THREAD_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)

# Headless EGL. Omitting it is a hard crash inside mujoco.MjrContext, and the
# message ("an OpenGL platform library has not been loaded") does not say which
# variable is missing.
# 헤드리스 EGL. 빼먹으면 mujoco.MjrContext 안에서 크래시하고, 메시지는 어느 변수가
# 없는지 말해주지 않는다.
DEFAULT_MUJOCO_GL = "egl"

# The GPUs allocated to this account (j15a103). **0, 5, 7, 8, 9 belong to other
# people.** The first version of this module defaulted to "0" -- a GPU we do not
# have -- which is exactly the mistake it was written to prevent, made in the file
# that prevents it. 🟢 2026-09-07, told by the user after the operators warned
# about load.
# 이 계정(j15a103)에 할당된 GPU. **0, 5, 7, 8, 9 는 남의 것이다.** 이 모듈의 첫
# 판은 기본값을 "0" — 우리에게 없는 장 — 으로 뒀다. 막으려고 쓴 파일에서 막으려던
# 실수를 냈다. 🟢 2026-09-07, 서버 측 부하 경고 후 사용자가 알려줬다.
ALLOWED_GPUS: tuple[int, ...] = (1, 2, 3, 4, 6)

# Our own card. Always available to us, so it is the default and nothing has to be
# decided to use it. The other allocated cards are taken **only when idle** --
# borrowed, not owned.
# 우리 장. 항상 쓸 수 있으므로 기본값이고, 쓰려고 아무것도 결정하지 않아도 된다.
# 나머지 할당분은 **한가할 때만** 잡는다. 빌리는 것이고 소유가 아니다.
HOME_GPU = 2

# "Idle" thresholds. Memory first: a card with memory held has someone's process on
# it even at 0% utilisation, and a 1.3M-parameter job shows 1 GB while its GPU
# utilisation reads 1% 🟢 -- utilisation alone would call that free.
# "한가함"의 기준. 메모리 우선 — 메모리가 잡혀 있으면 사용률 0% 여도 남의 프로세스가
# 올라가 있다. 실측 🟢: 1.3M 파라미터 잡이 GPU 사용률 1% 인데 메모리 1GB 를 쓴다.
# 사용률만 보면 그걸 한가하다고 부른다.
IDLE_MEM_MB = 300
IDLE_UTIL_PCT = 20

LOCK_DIR = Path(__file__).resolve().parent / "out"


def _gpu_load() -> dict[int, tuple[int, int]]:
    """Utilisation and used memory per GPU index. Empty when nvidia-smi is absent.
    GPU 번호별 (사용률, 사용 메모리). nvidia-smi 가 없으면 빈 dict."""
    import subprocess

    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=8, check=False,
        )
        if out.returncode != 0:
            return {}
        rows: dict[int, tuple[int, int]] = {}
        for line in out.stdout.strip().splitlines():
            parts = [x.strip() for x in line.split(",")]
            if len(parts) >= 3:
                rows[int(parts[0])] = (int(parts[1]), int(parts[2]))
        return rows
    except (OSError, ValueError, subprocess.SubprocessError):
        return {}


def _is_idle(stat: tuple[int, int]) -> bool:
    util, mem = stat
    return mem <= IDLE_MEM_MB and util <= IDLE_UTIL_PCT


def free_gpus(max_n: int | None = None) -> list[int]:
    """Allocated cards that look idle, home card first.
    한가해 보이는 할당분. 우리 장이 맨 앞.

    For fanning a multi-seed block out. Read once at launch -- a card that frees up
    later is not picked up, and a card that gets busy mid-run is not released. That
    is the cost of not preempting anyone.
    다시드 블록을 뿌리기 위한 것. 시작 시점에 한 번 읽는다 — 나중에 비는 장은
    잡지 않고, 도중에 바빠지는 장을 놓아주지도 않는다. 아무도 밀어내지 않는 대가다.
    """
    load = _gpu_load()
    if not load:
        return [HOME_GPU]
    order = [HOME_GPU] + [i for i in ALLOWED_GPUS if i != HOME_GPU]
    out = [HOME_GPU]
    for i in order:
        if i == HOME_GPU or i not in load:
            continue
        if _is_idle(load[i]):
            out.append(i)
    return out[:max_n] if max_n else out


def pick_gpu() -> str:
    """One card: ours if it is free, otherwise the quietest idle allocated card.
    한 장: 우리 장이 비었으면 그것, 아니면 한가한 할당분 중 가장 조용한 것."""
    load = _gpu_load()
    if not load:
        return str(HOME_GPU)
    home = load.get(HOME_GPU)
    if home is None or _is_idle(home):
        return str(HOME_GPU)
    others = [(load[i], i) for i in ALLOWED_GPUS if i != HOME_GPU and i in load
              and _is_idle(load[i])]
    if not others:
        # 전부 바쁘면 우리 장으로 간다. 남의 장을 밀어내지 않는다.
        return str(HOME_GPU)
    others.sort(key=lambda kv: (kv[0][1], kv[0][0], kv[1]))
    return str(others[0][1])


def _unbuffer() -> None:
    """Make stdout line-buffered so a redirected log shows progress.
    stdout 을 줄 단위 버퍼로 바꿔서 리다이렉트된 로그에 진행이 보이게 한다.

    Python buffers stdout in blocks when it is not a terminal, so
    `nohup ... > log &` shows nothing until the buffer fills or the process exits.
    2026-09-07 🟢: a live `probe_freeze` sat at one log line for two minutes and
    looked dead. Every server command carried `PYTHONUNBUFFERED=1` by hand until
    this module took the env vars over -- and it did not take that one.
    파이썬은 stdout 이 터미널이 아니면 블록 단위로 버퍼링하므로
    `nohup ... > log &` 는 버퍼가 차거나 프로세스가 끝날 때까지 아무것도 안 보인다.
    2026-09-07 실측 🟢: 살아 있는 `probe_freeze` 가 2분간 로그 1줄에 머물러 죽은 것처럼
    보였다. 서버 명령마다 손으로 `PYTHONUNBUFFERED=1` 을 붙이고 있었는데, 이 모듈이
    환경변수를 인수하면서 **그것만 빠뜨렸다.**
    """
    os.environ.setdefault("PYTHONUNBUFFERED", "1")   # 자식 프로세스용
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(line_buffering=True)   # 이미 열린 이 프로세스용
        except Exception:
            pass


def _set_mujoco_gl() -> None:
    """Ask for headless EGL only where EGL exists.
    EGL 이 있는 곳에서만 헤드리스 EGL 을 요청한다.

    `MUJOCO_GL` is read by mujoco **at import time** and the backend is loaded
    eagerly, so setting it unconditionally breaks any machine without EGL even
    when no renderer is created. 2026-09-07 🟢: setting it here made
    `probe_freeze --policy scripted` (render off) crash on the local VM inside
    `OpenGL.raw.EGL._types`, and the traceback points at OpenGL rather than at the
    variable that caused it.
    `MUJOCO_GL` 은 mujoco 가 **임포트 시점에** 읽고 백엔드를 즉시 로드한다. 그래서
    무조건 설정하면 EGL 없는 기계에서 렌더를 안 켜도 깨진다. 2026-09-07 실측 🟢:
    여기서 설정한 탓에 로컬 VM 에서 `probe_freeze --policy scripted`(렌더 off)가
    `OpenGL.raw.EGL._types` 안에서 죽었고, 트레이스백은 원인이 된 변수가 아니라
    OpenGL 을 가리킨다.
    """
    if "MUJOCO_GL" in os.environ:
        return
    from ctypes.util import find_library

    if find_library("EGL"):
        os.environ["MUJOCO_GL"] = DEFAULT_MUJOCO_GL


def _apply() -> None:
    _unbuffer()
    for var in _THREAD_VARS:
        os.environ.setdefault(var, DEFAULT_THREADS)
    _set_mujoco_gl()

    given = os.environ.get("CUDA_VISIBLE_DEVICES")
    if given is None:
        os.environ["CUDA_VISIBLE_DEVICES"] = pick_gpu()
        return
    # 명시 지정은 존중하되, 남의 장이면 소리내어 말한다.
    try:
        asked = [int(x) for x in given.split(",") if x.strip() != ""]
    except ValueError:
        return
    outside = [i for i in asked if i not in ALLOWED_GPUS]
    if outside:
        print(f"⚠️ CUDA_VISIBLE_DEVICES={given} 에 할당분이 아닌 장이 있다: {outside}. "
              f"할당분은 {list(ALLOWED_GPUS)} 다. 남의 작업을 밀어낸다.", file=sys.stderr)
    if len(asked) > 1:
        print(f"⚠️ GPU {len(asked)}장을 잡는다. 잡 하나는 한 장이면 된다 — "
              "여러 장이 필요하면 잡을 나눠서 각각 한 장씩 잡아라.", file=sys.stderr)


_apply()


def torch_threads(n: int | None = None) -> None:
    """Cap torch's intra- and inter-op threads. Call right after torch is imported.
    torch 의 intra-op·inter-op 스레드를 제한한다. torch 임포트 직후에 부른다.

    The BLAS caps do not always reach torch's own thread pools, so this is set
    explicitly rather than assumed.
    BLAS 캡이 torch 자체 스레드 풀까지 항상 닿지는 않으므로 가정하지 않고 명시한다."""
    k = int(n or DEFAULT_THREADS)
    try:
        import torch

        torch.set_num_threads(k)
        # inter-op 은 한 번만 설정 가능하고 이후 호출은 예외다. 조용히 넘긴다.
        try:
            torch.set_num_interop_threads(max(1, k // 2))
        except Exception:
            pass
    except Exception:
        pass


def claim(name: str, stale_ok: bool = False) -> None:
    """Refuse to start if another instance of this tool is already running.
    같은 도구가 이미 돌고 있으면 시작을 거부한다.

    A duplicate run is worse than a slow one: four copies of `probe_freeze` all
    redirected to one log produced a file that could not be read, and the wasted
    load is charged to everyone on the machine.
    중복 실행은 느린 것보다 나쁘다. `probe_freeze` 사본 4개가 한 로그로 리다이렉트돼
    읽을 수 없는 파일이 됐고, 낭비된 부하는 머신을 쓰는 모두에게 청구된다.
    """
    hostname = platform.node().lower()
    if hostname.startswith("jupyter"):
        raise SystemExit(
            "This diagnostic is local-only. Shared GPU hosts allow training "
            "and only MuJoCo tools that explicitly declare no learned-policy "
            "inference."
        )

    # 큐 러너가 같은 도구를 조건별로 병렬 기동할 수 있게 락 이름을 덮어쓴다.
    # 옵트인이다 — 환경변수가 없으면 기존 동작과 완전히 동일하다.
    # ⚠️ 공유 코드다 (트랙 A 영향). 기본 동작 불변이므로 기존 실행에는 변화가 없다.
    name = os.environ.get("AI_CLAIM_NAME", name)
    LOCK_DIR.mkdir(parents=True, exist_ok=True)
    lock = LOCK_DIR / f".lock_{name}"
    if lock.exists():
        try:
            pid = int(lock.read_text(encoding="utf-8").split()[0])
        except (ValueError, IndexError, OSError):
            pid = -1
        alive = pid > 0 and Path(f"/proc/{pid}").exists()
        if alive and not stale_ok:
            print(f"✗ `{name}` 이 이미 PID {pid} 로 돌고 있다. "
                  f"중복 실행하지 않는다.\n"
                  f"  끝내려면: kill {pid}\n"
                  f"  락만 지우려면: rm {lock}", file=sys.stderr)
            raise SystemExit(3)
        if not alive:
            print(f"· 죽은 락 정리: {lock.name} (PID {pid})")

    lock.write_text(f"{os.getpid()} {name}\n", encoding="utf-8")
    atexit.register(lambda: lock.unlink(missing_ok=True))
    # 매 실행이 뭘 얼마나 가져갔는지 로그 첫 줄에 남긴다. 나중에 "우리가 풀점유
    # 했다" 는 말을 들었을 때 대조할 것이 있어야 한다.
    print(banner(), flush=True)
    warn_if_busy()


def warn_if_busy(frac: float = 0.7) -> None:
    """Say so when the machine is already loaded by someone else.
    머신이 이미 남의 작업으로 차 있으면 말한다.

    Not a refusal -- we cannot tell whose load it is, and refusing on a shared box
    would mean never running. But starting a render sweep into load 165 was how
    the operators came to warn us 🟢, and a line on stdout is cheap.
    거부는 아니다. 부하가 누구 것인지 알 수 없고, 공유 머신에서 거부하면 아무것도
    못 돌린다. 다만 load 165 에 렌더 스윕을 얹은 것이 서버 측 경고를 부른 경로였고 🟢,
    stdout 한 줄은 싸다.
    """
    try:
        one, five, _ = os.getloadavg()
        cores = os.cpu_count() or 1
    except (OSError, AttributeError):
        return
    if one > cores * frac:
        print(f"⚠️ 머신 부하가 이미 높다 (1분 {one:.0f} / {cores}코어). "
              f"내 것이 아니면 나중에 돌려라. 스레드 캡 {DEFAULT_THREADS} 로 돈다.",
              file=sys.stderr)


def banner() -> str:
    """One line naming the limits actually in effect.
    실제로 적용된 제한을 한 줄로."""
    return (f"[런타임] GPU {os.environ.get('CUDA_VISIBLE_DEVICES')} "
            f"(우리 장 {HOME_GPU} · 할당분 {','.join(map(str, ALLOWED_GPUS))} · "
            f"지금 한가함 {free_gpus()}) · "
            f"스레드 {os.environ.get('OMP_NUM_THREADS')} · "
            f"MUJOCO_GL {os.environ.get('MUJOCO_GL') or '미설정(EGL 없음)'}")
