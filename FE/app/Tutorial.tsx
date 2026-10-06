'use client';

import { useEffect, useLayoutEffect, useRef, useState } from 'react';

const steps = [
  { label: '대시 보드', title: '전체 상태를 한눈에 봐요', description: '장치 연결과 학습 진행 상황을 모아 보여줘요.' },
  { label: '학습', title: '로봇에게 동작을 가르쳐요', description: '진행 중이거나 끝난 학습을 한눈에 보고, 가르친 동작은 동작 목록에서 볼 수 있어요.' },
  { label: '작업', title: '배운 동작을 묶어 실행해요', description: '로봇이 할 일을 순서대로 만들고 실행해요.' },
  { label: '장치 관리', title: '수집 장치와 로봇을 관리해요', description: '동작을 기록하는 수집 핸들과 학습한 동작을 실행할 로봇을 관리해요.' },
];

/* 실제 콘솔 사이드바 5개 메뉴. steps(4개)의 index를 실제 활성 메뉴로 매핑한다. */
const NAV = ['대시 보드', '동작 목록', '학습', '작업', '장치 관리'];
const ACTIVE_NAV = [0, 2, 3, 4];

/* 단계별 번호 콜아웃(①②) — 미니 화면의 .tour-spot 요소 순서와 1:1 매칭된다. */
const LEGENDS = [
  [['실행 중인 작업', '지금 로봇이 하는 일과 진행률을 확인해요'], ['요약 지표', '등록 장치·진행 중 학습·오늘 완료 작업을 한눈에']],
  [['상태 요약', '전체·진행 중·완료·확인 필요 수'], ['학습 진행', '진행률·현재 단계·성공/실패 확인']],
  [['실행 중인 작업', '로봇에서 도는 작업을 확인하고 중지해요'], ['작업 목록', '저장한 작업을 열어 수정·실행']],
  [['수집 핸들·로봇 탭', '보고 싶은 장치 종류를 전환해요'], ['장치 카드', '장치 정보와 연결 상태를 확인해요']],
];

/* 1번은 왼쪽 노트, 2번은 오른쪽 노트로 이어지므로 핀도 각각 대상의 좌·우 모서리에 붙인다(연결선이 목업을 가로지르지 않게). */
const Pin = ({ n }: { n: number }) => <span className={`tour-pin tour-pin-${n}`} aria-hidden="true">{n}</span>;

/** 각 단계의 미니 화면 — 실제 페이지 레이아웃을 축약해 그리고, 콜아웃 대상엔 .tour-spot+핀을 단다. */
function ScreenBody({ index }: { index: number }) {
  if (index === 0) return <>
    <h3>대시 보드</h3>
    <div className="tour-run tour-spot"><Pin n={1} />
      <div className="tour-run-head"><b>현재 실행 중인 작업</b><em>1개 실행 중</em></div>
      <div className="tour-row"><span className="tour-av">R</span><div className="tour-rmain"><strong>청기 잡기</strong><small>SHANKS · 1/1 단계</small></div><em className="tour-badge ghost">중지 요청</em></div>
    </div>
    <div className="tour-stat4 tour-spot"><Pin n={2} />
      <span><small>수집 핸들</small><strong>1<i>개</i></strong></span>
      <span><small>등록된 로봇</small><strong>2<i>개</i></strong></span>
      <span><small>진행 중 학습</small><strong>1<i>건</i></strong></span>
      <span><small>오늘 완료</small><strong>0<i>건</i></strong></span>
    </div>
    <div className="tour-two">
      <div className="tour-panel"><b>로봇 상태</b><div className="tour-row"><span className="tour-av sm">R</span><small>SHANKS</small><em className="tour-badge soft">작업 중</em></div></div>
      <div className="tour-panel"><b>최근 학습</b><div className="tour-row"><small>청기잡기</small><em className="tour-badge soft">진행 중</em></div></div>
    </div>
  </>;
  if (index === 1) return <>
    <h3>학습 현황</h3>
    <div className="tour-tabs4 tour-spot"><Pin n={1} /><span className="on">전체 <b>7</b></span><span>진행 중 <b>1</b></span><span>완료 <b>2</b></span><span>확인 <b>4</b></span></div>
    <div className="tour-panel list tour-spot"><Pin n={2} />
      <div className="tour-lrow"><span className="tour-av star">✦</span><div className="tour-rmain"><strong>청기잡기</strong><small>재할당 대기</small></div><div className="tour-prog"><div className="tour-progress"><i style={{ width: '73%' }} /></div><b>73%</b></div><em className="tour-badge soft">진행 중</em></div>
      <div className="tour-lrow"><span className="tour-av star">✦</span><div className="tour-rmain"><strong>팔 올리기</strong><small>학습 실패</small></div><div className="tour-prog"><div className="tour-progress"><i style={{ width: '4%' }} /></div><b>0%</b></div><em className="tour-badge fail">실패</em></div>
    </div>
  </>;
  if (index === 2) return <>
    <h3>작업 목록</h3>
    <div className="tour-run tour-spot"><Pin n={1} />
      <div className="tour-run-head"><b>현재 실행 중인 작업</b><em>1개 실행 중</em></div>
      <div className="tour-row"><span className="tour-av">R</span><div className="tour-rmain"><strong>청기 잡기</strong><small>SHANKS · 1/1 단계</small></div><em className="tour-badge ghost">중지 요청</em></div>
    </div>
    <div className="tour-panel list tour-spot"><Pin n={2} />
      <div className="tour-wrow"><span className="tour-av sq">W</span><div className="tour-rmain"><strong>청기 잡기</strong><small>학습 동작 1개 · 제어 0개</small></div><em>약 25초</em><i className="tour-chev">›</i></div>
      <div className="tour-wrow"><span className="tour-av sq">W</span><div className="tour-rmain"><strong>청기 들기</strong><small>학습 동작 2개 · 제어 0개</small></div><em>약 50초</em><i className="tour-chev">›</i></div>
    </div>
  </>;
  return <>
    <h3>장치 관리</h3>
    <div className="tour-dtabs tour-spot"><Pin n={1} /><span className="on">수집 핸들 <b>1</b></span><span>로봇 <b>2</b></span></div>
    <div className="tour-dcard tour-spot"><Pin n={2} />
      <div className="tour-dcard-top"><span className="tour-device-icon">U</span><em className="tour-badge soft">● 온라인</em></div>
      <strong>UMI-001</strong><small>801호</small>
      <div className="tour-dsep" />
      <div className="tour-dmeta"><span>최근 사용</span><b>09. 09. 오전 09:30</b></div>
      <div className="tour-dmeta"><span>보유 데이터</span><b>0개</b></div>
    </div>
  </>;
}

function ScreenPreview({ index }: { index: number }) {
  const active = ACTIVE_NAV[index];
  const visual = useRef<HTMLElement>(null);
  const [links, setLinks] = useState<{ d: string; x: number; y: number }[]>([]);
  /* 데스크톱(>960): 핀 위치를 재서 ①은 왼쪽·②는 오른쪽 설명을 핀 높이에 맞추고 곡선으로 잇는다. 좁은 폭은 아래로 쌓고 선은 생략. */
  useLayoutEffect(() => {
    const root = visual.current;
    if (!root) return;
    const measure = () => {
      const notes = Array.from(root.querySelectorAll<HTMLElement>('.tour-note'));
      if (window.innerWidth <= 960) {
        notes.forEach((note) => { note.style.cssText = ''; });
        setLinks([]);
        return;
      }
      const bounds = root.getBoundingClientRect();
      /* 큰 화면에서 다이얼로그에 zoom이 걸리면 getBoundingClientRect는 렌더 px을 주는데
         style.top/left/width는 zoom 안쪽 로컬 px이라 그대로 쓰면 배율이 두 번 먹는다 → 로컬 좌표로 환산한다. */
      const scale = root.offsetWidth ? bounds.width / root.offsetWidth : 1;
      const boundsW = root.offsetWidth || bounds.width;
      const toX = (clientX: number) => (clientX - bounds.left) / scale;
      const toY = (clientY: number) => (clientY - bounds.top) / scale;
      const browser = root.querySelector('.tour-browser');
      if (!browser) return;
      const brLeft = toX(browser.getBoundingClientRect().left);
      const spots = Array.from(root.querySelectorAll('.tour-mini-main .tour-spot'));
      const next: { d: string; x: number; y: number }[] = [];
      spots.forEach((spot, i) => {
        const note = notes[i];
        const pin = spot.querySelector('.tour-pin');
        if (!note || !pin) return;
        const p = pin.getBoundingClientRect();
        const px = toX(p.left + p.width / 2);
        const py = toY(p.top + p.height / 2);
        const left = i === 0;
        /* ①은 핀이 목업 안쪽(사이드바 옆)이라 그대로 붙이면 UI를 덮는다 → 목업 왼쪽 바깥 여백으로 물린다.
           ②는 핀 옆(장치 카드처럼 안쪽 여백이면 그 옆)에 붙인다. */
        const edge = left ? Math.min(px - 34, brLeft - 14) : px + 34;
        const room = (left ? edge : boundsW - edge) - 6;
        note.style.top = `${py}px`;
        note.style.width = `${Math.max(120, Math.min(190, room))}px`;
        if (left) { note.style.left = 'auto'; note.style.right = `${boundsW - edge}px`; }
        else { note.style.right = 'auto'; note.style.left = `${edge}px`; }
        let n = note.getBoundingClientRect();
        /* ②가 ①의 강조 링과 겹치면 링 아래로 내린다. */
        if (!left && spots[0]) {
          const o = spots[0].getBoundingClientRect();
          if (n.left < o.right && n.right > o.left && n.top < o.bottom && n.bottom > o.top) {
            note.style.top = `${toY(o.bottom) + n.height / scale / 2 + 10}px`;
            n = note.getBoundingClientRect();
          }
        }
        const sx = toX(left ? n.right : n.left);
        const sy = toY(n.top + n.height / 2);
        next.push({
          d: left ? `M ${sx + 4} ${sy} C ${sx + 36} ${sy}, ${px - 36} ${py}, ${px - 10} ${py}` : `M ${sx - 4} ${sy} C ${sx - 36} ${sy}, ${px + 36} ${py}, ${px + 10} ${py}`,
          x: px, y: py,
        });
      });
      setLinks(next);
    };
    const observer = new ResizeObserver(measure);
    observer.observe(root);
    root.querySelectorAll('.tour-browser,.tour-mini-main').forEach((element) => observer.observe(element));
    window.addEventListener('resize', measure);
    measure();
    /* 부모의 showModal()(useEffect)이 이 layout effect 뒤에 실행되므로, 다이얼로그가 실제로 표시된 다음 프레임과 폰트 적용 뒤에 한 번씩 더 잰다. */
    const frame = requestAnimationFrame(measure);
    const settle = window.setTimeout(measure, 150);
    return () => { observer.disconnect(); window.removeEventListener('resize', measure); cancelAnimationFrame(frame); window.clearTimeout(settle); };
  }, [index]);
  return <figure ref={visual} className="tour-visual" aria-label={`${steps[index].label} 화면 예시`}>
    <svg className="tour-connections" aria-hidden="true">{links.map((link, i) => <g key={i}><path d={link.d} /><circle cx={link.x} cy={link.y} r="3" /></g>)}</svg>
    <div className="tour-browser"><div className="tour-browser-bar"><i /><i /><i /><span>UMI Studio</span></div>
      <div className="tour-screen"><div className="tour-mini-nav" aria-hidden="true"><b>S.</b>{NAV.map((name, i) => <span key={name} className={i === active ? 'chosen' : ''}>{name}</span>)}</div>
        <div className="tour-mini-main"><ScreenBody index={index} /></div>
      </div></div>
    {LEGENDS[index].map(([title, desc], i) => <aside key={title} className={`tour-note tour-note-${i + 1}`}><span>{i + 1}</span><div><strong>{title}</strong><small>{desc}</small></div></aside>)}
  </figure>;
}

export default function Tutorial({ onClose }: { onClose: () => void }) {
  const [index, setIndex] = useState(0);
  const dialog = useRef<HTMLDialogElement>(null);
  const heading = useRef<HTMLHeadingElement>(null);
  const content = useRef<HTMLDivElement>(null);
  const step = steps[index];
  useEffect(() => {
    const previous = document.activeElement as HTMLElement | null;
    const overflow = document.body.style.overflow;
    const modal = dialog.current;
    modal?.showModal();
    document.body.style.overflow = 'hidden';
    return () => { modal?.close(); document.body.style.overflow = overflow; previous?.focus(); };
  }, []);
  useEffect(() => { content.current?.scrollTo(0, 0); heading.current?.focus(); }, [index]);
  return <dialog ref={dialog} className="tutorial-dialog" aria-labelledby="tutorial-title" onCancel={onClose}>
    <header className="tutorial-header"><span>UMI Studio <small>시작 가이드</small></span><button type="button" onClick={onClose} aria-label="튜토리얼 닫기">×</button></header>
    <div className="tutorial-body" ref={content}>
      <section className="tutorial-content"><p className="tutorial-counter">{String(index + 1).padStart(2, '0')} — {step.label}</p><h2 id="tutorial-title" ref={heading} tabIndex={-1}>{step.title}</h2><p className="tutorial-description">{step.description}</p>
      </section>
      <ScreenPreview index={index} />
    </div>
    <footer className="tutorial-footer"><nav className="tour-dots" aria-label="튜토리얼 단계">{steps.map((item, i) => <button key={item.label} type="button" aria-label={`${i + 1}단계 ${item.label}`} aria-current={index === i ? 'step' : undefined} onClick={() => setIndex(i)}><i /></button>)}</nav><span className="tour-page-count" aria-live="polite">{index + 1} / {steps.length}</span><div className="tour-footer-actions"><button type="button" className="tour-skip" onClick={onClose}>건너뛰기</button><button type="button" className="tour-back" disabled={index === 0} onClick={() => setIndex(index - 1)}>이전</button><button type="button" className="primary-button" onClick={() => index === steps.length - 1 ? onClose() : setIndex(index + 1)}>{index === steps.length - 1 ? '안내 마치기' : '다음'} <span aria-hidden="true">→</span></button></div></footer>
  </dialog>;
}
