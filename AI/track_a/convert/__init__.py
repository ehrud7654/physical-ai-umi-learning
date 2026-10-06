"""Raw recordings to dataset format — project-task.
raw → dataset 포맷 변환 — project-task.

⚠️ **2026-09-08 정정.** 이 파일의 이전 내용은 "여기서 `contract.episode` 를 직접
   임포트해 `Episode` 를 만들라"고 안내했다. **더 이상 맞지 않다.**

변환기는 `AI/umi/` 로 옮겨졌다 (D-AI-30). 이슈 31 과 127 이 같은 코드를 써야 하고,
raw 스키마는 양 트랙이 함께 읽어야 하는데 `track_a/` 안에 있으면 트랙 B 가
"track_a 를 임포트하지 않는다" 규칙을 깨야 하기 때문이다.

**그래서 여기서 만들 것은 `Episode` 가 아니라 `RawEpisode` 다.**

    track_a/convert/arcore.py   ARCore 로그 → umi.raw.RawEpisode   ← 트랙 A 가 채운다
    umi/convert.py              RawEpisode → contract.Episode      ← 완료
    data/verify.py              계약 검증                          ← 완료

임포트는 이렇게 한다 (`AI/` 에서 실행):

    from umi.raw import RawEpisode, RawMeta, write_raw, FRAME_ROBOT_BASE

`contract/` 를 직접 임포트할 필요가 없다. 계약을 만족시키는 일은 `umi/convert.py`
가 이미 한다 — 합성 20편으로 계약 위반 0 을 확인했다 🟢.

착수 조건과 결정 기록은 `track_a/HANDOVER_track_a.md` 를 읽어라.
"""
