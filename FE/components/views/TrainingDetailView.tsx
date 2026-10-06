'use client';

import { useEffect, useState } from 'react';
import type { Training, TrainingStatus } from '@/types';
import { useAppStore } from '@/hooks/useAppStore';
import { fetchTrainingDetail } from '@/services';

const TERMINAL: TrainingStatus[] = ['완료', '실패', '취소됨'];

/** 학습 하나의 상세(라우트). 진행 중이면 3초마다 갱신하고, 상태에 따라 취소·재시도·작업 생성으로 이어간다. */
export function TrainingDetailView({ trainingId, onBack, onUseResult, onCancelled, onToast }: { trainingId: string; onBack: () => void; onUseResult: (skillName: string) => void; onCancelled: () => void; onToast: (message: string) => void }) {
  const { skills, trainings, refreshTrainings, cancelTrainingById, retryTraining } = useAppStore();
  const listed = trainings.find((item) => item.id === trainingId) ?? null;
  const [fetched, setFetched] = useState<Training | null>(null);
  const [loadFailed, setLoadFailed] = useState(false);
  const [confirmCancel, setConfirmCancel] = useState(false);
  const [busy, setBusy] = useState(false);
  const detail = fetched ?? listed;
  const terminal = !!detail && TERMINAL.includes(detail.status);

  /* 진입 시 1회 조회하고, 종료 상태가 아니면 3초 주기로 갱신한다. */
  useEffect(() => {
    if (trainingId.startsWith('draft-')) return;
    let active = true;
    const load = async () => {
      try {
        const latest = await fetchTrainingDetail(trainingId);
        if (active) { setFetched(latest); setLoadFailed(false); }
      } catch {
        if (active) setLoadFailed(true);
      }
    };
    void load();
    if (terminal) return () => { active = false; };
    const timer = window.setInterval(() => void load(), 3000);
    return () => { active = false; window.clearInterval(timer); };
  }, [trainingId, terminal]);

  if (!detail) {
    return <><button className="back-link" type="button" onClick={onBack}>← 학습 목록</button><div className="center-panel"><section className="card state-card">{loadFailed ? <><span className="state-icon">!</span><h2>학습을 찾을 수 없어요</h2><p>삭제되었거나 주소가 잘못되었을 수 있어요.</p></> : <><span className="spinner" aria-hidden="true" /><h2>학습 정보를 불러오는 중이에요</h2><p>잠시만 기다려 주세요.</p></>}</section></div></>;
  }

  const badgeClass = detail.status === '실패' ? 'failed' : detail.status === '완료' ? 'complete' : '';
  const linkedSkill = skills.some((skill) => skill.name === detail.name);

  const cancelConfirmed = async () => {
    setBusy(true);
    try {
      await cancelTrainingById(detail.id);
      await refreshTrainings();
      onToast(`‘${detail.name}’ 학습을 취소했어요.`);
      setConfirmCancel(false);
      onCancelled();
    } catch {
      onToast('학습을 취소하지 못했어요.');
    } finally {
      setBusy(false);
    }
  };
  const retry = async () => {
    setBusy(true);
    try {
      await retryTraining(detail);
      onToast(`‘${detail.name}’ 학습을 다시 시작했어요.`);
    } catch {
      onToast('학습을 다시 시작하지 못했어요.');
    } finally {
      setBusy(false);
    }
  };

  return <>
    <button className="back-link" type="button" onClick={onBack}>← 학습 목록</button>
    <div className="page-heading list-page-heading title-divider">
      <div>
        <div className="data-modal-title"><h1>{detail.name}</h1><span className={`training-badge ${badgeClass}`}>{detail.status}</span></div>
        <p>{detail.robot} · {detail.started}</p>
      </div>
      <div className="training-detail-actions">
        {detail.status === '진행 중' && <button className="danger-button" type="button" disabled={busy} onClick={() => setConfirmCancel(true)}>학습 취소</button>}
        {detail.status === '실패' && <button className="primary-button" type="button" disabled={busy} onClick={() => void retry()}>{busy ? '요청 중...' : '다시 시도'}</button>}
        {detail.status === '완료' && <button className="primary-button" type="button" disabled={!linkedSkill} onClick={() => onUseResult(detail.name)}>이 동작으로 작업 생성</button>}
      </div>
    </div>

    <section className="card training-detail-card" aria-label="학습 정보">
      <dl className="data-detail-grid">
        <div><dt>현재 단계</dt><dd>{detail.stage}</dd></div>
        <div><dt>대상 로봇</dt><dd>{detail.robot}</dd></div>
        <div><dt>시작 시각</dt><dd>{detail.started}</dd></div>
        <div><dt>입력 방식</dt><dd>{detail.inputMode === 'EXISTING_EPISODES' ? '수집 데이터' : '신규 수집'}</dd></div>
        <div><dt>수집 데이터</dt><dd>{detail.receivedEpisodeCount} / {detail.expectedEpisodeCount}개</dd></div>
        <div><dt>진행률</dt><dd>{detail.progress}%</dd></div>
      </dl>
      <div className="training-modal-progress"><div className="training-progress"><div><i style={{ width: `${detail.progress}%` }} /></div><strong>{detail.progress}%</strong></div></div>
      {detail.status === '완료' && !linkedSkill && <p className="modal-note">이 학습 결과의 동작 연결 정보가 아직 없어요.</p>}
      {loadFailed && <p className="active-execution-warning">최신 진행 상태를 갱신하지 못했어요. 목록의 값으로 표시 중이에요.</p>}
    </section>

    {confirmCancel && <div className="modal-backdrop" onMouseDown={() => !busy && setConfirmCancel(false)}><section className="confirm-modal" role="dialog" aria-modal="true" aria-label="학습 취소 확인" onMouseDown={(e) => e.stopPropagation()}><div className="modal-heading"><div><h2>학습을 취소할까요?</h2><p>취소 요청 후 서버가 중단을 확인하면 취소 완료로 표시돼요.</p></div><button className="close-button" type="button" onClick={() => setConfirmCancel(false)} aria-label="닫기">×</button></div><div className="card-actions"><button className="secondary-button" type="button" disabled={busy} onClick={() => setConfirmCancel(false)}>계속 진행</button><button className="danger-button" type="button" disabled={busy} onClick={() => void cancelConfirmed()}>{busy ? '요청 중...' : '취소 요청'}</button></div></section></div>}
  </>;
}
