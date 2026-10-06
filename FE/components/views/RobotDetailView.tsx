'use client';

import { type FormEvent, useState, useEffect } from 'react';
import type { RobotDetail, RobotExecution, View } from '@/types';
import { useAppStore } from '@/hooks/useAppStore';
import { fetchRobotDetail, fetchRobotExecutions } from '@/services';
export function RobotDetailView({ robotId, onBack, onNavigate, onToast }: { robotId: string; onBack: () => void; onNavigate: (view: View) => void; onToast: (message: string) => void }) {
  const { robots, savedWorks, updateRobotInfo } = useAppStore();
  const listedRobot = robots.find(item => item.id === robotId) ?? robots[0];
  const detailRobotId = listedRobot?.robotId;
  const [detail, setDetail] = useState<RobotDetail | null>(null);
  const [executionHistory, setExecutionHistory] = useState<RobotExecution[] | null>(null);
  const [detailError, setDetailError] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  /* eslint-disable react-hooks/set-state-in-effect -- 로봇 변경 시 상세·실행 기록을 다시 조회한다. */
  useEffect(() => {
    let active = true;
    setDetail(null);
    setExecutionHistory(null);
    setDetailError(false);
    if (!detailRobotId) return () => { active = false; };
    Promise.allSettled([fetchRobotDetail(detailRobotId), fetchRobotExecutions(detailRobotId)]).then(([robotResult, executionResult]) => {
      if (!active) return;
      if (robotResult.status === 'fulfilled') setDetail(robotResult.value);
      else setDetailError(true);
      if (executionResult.status === 'fulfilled') setExecutionHistory(executionResult.value.data);
    });
    return () => { active = false; };
  }, [detailRobotId]);
  /* eslint-enable react-hooks/set-state-in-effect */
  const robot = listedRobot && detail ? { ...listedRobot, name: detail.name, status: detail.status } : listedRobot;
  if (!robot) return <div className="center-panel"><section className="card state-card"><span className="state-icon">R</span><h2>등록된 로봇이 없어요</h2><p>장치 관리에서 로봇을 먼저 등록해 주세요.</p><button className="primary-button" onClick={onBack}>장치 관리로 이동</button></section></div>;

  const today = new Date().toDateString();
  const todayExecutions = executionHistory?.filter(item => new Date(item.requestedAt).toDateString() === today);
  const succeeded = todayExecutions?.filter(item => item.status === 'SUCCEEDED').length;
  const failed = todayExecutions?.filter(item => item.status === 'FAILED').length;
  const runtimeSeconds = todayExecutions?.reduce((sum, item) => item.startedAt && item.finishedAt
    ? sum + Math.max(0, (new Date(item.finishedAt).getTime() - new Date(item.startedAt).getTime()) / 1000)
    : sum, 0);
  const runtimeLabel = runtimeSeconds === undefined ? 'API로 호출할 수 없습니다' : `${Math.floor(runtimeSeconds / 3600)}시간 ${Math.floor(runtimeSeconds % 3600 / 60)}분`;
  const executionLabel = (status: RobotExecution['status']) => status === 'SUCCEEDED' ? '성공' : status === 'FAILED' ? '실패' : status === 'CANCELED' ? '취소' : '진행 중';

  return <><button className="back-link" onClick={onBack}>← 로봇 목록</button><div className="robot-detail-hero card"><div className="device-visual large">R</div><div><h1>{robot.name}</h1><p>{robot.location} · 마지막 통신 {detail?.lastSeenAt ? new Date(detail.lastSeenAt).toLocaleString('ko-KR') : 'API로 호출할 수 없습니다'}</p></div><span className={`status-pill large status-${robot.status.replace(' ', '-')}`}><i />{detailError ? 'API로 호출할 수 없습니다' : robot.status}</span><div className="detail-actions"><button className="secondary-button" onClick={() => setSettingsOpen(true)}>설정</button><button className="primary-button" disabled={robot.status !== '온라인'} onClick={() => onNavigate('work')}>작업 생성</button></div></div><div className="detail-metrics robot-detail-metrics"><article className="card"><small>오늘 작업</small><strong>{todayExecutions ? `${todayExecutions.length}건` : 'API로 호출할 수 없습니다'}</strong><p>{todayExecutions ? `성공 ${succeeded} · 실패 ${failed}` : 'API로 호출할 수 없습니다'}</p></article><article className="card"><small>가동 시간</small><strong>{runtimeLabel}</strong><p>오늘 완료된 작업 기준</p></article></div><div className="robot-history-grid"><section className="card dashboard-section"><div className="dashboard-title"><div><h2>최근 작업 기록</h2></div></div><div className="history-list">{executionHistory === null ? <p><strong>API로 호출할 수 없습니다</strong></p> : executionHistory.length ? executionHistory.slice(0, 5).map(item => { const work = savedWorks.find(saved => saved.taskVersionId === item.taskVersionId); return <p key={item.id}><strong>{work?.name ?? 'API로 호출할 수 없습니다'}</strong><span className={`training-badge ${item.status === 'SUCCEEDED' ? 'complete' : item.status === 'FAILED' ? 'failed' : ''}`}>{executionLabel(item.status)}</span><small>{new Date(item.requestedAt).toLocaleString('ko-KR')}</small></p>; }) : <p><strong>최근 작업 기록이 없습니다.</strong></p>}</div></section></div>
    {settingsOpen && <RobotSettingsModal robot={robot} onClose={() => setSettingsOpen(false)} onSave={async (patch) => { await updateRobotInfo(robot.jetsonDeviceId ?? robot.id, patch); setSettingsOpen(false); onToast('로봇 설정을 저장했어요.'); }} />}
  </>;
}

function RobotSettingsModal({ robot, onClose, onSave }: { robot: { id: string; name: string; location: string }; onClose: () => void; onSave: (patch: { name: string; location: string }) => Promise<void> }) {
  const [name, setName] = useState(robot.name);
  const [location, setLocation] = useState(robot.location);
  const [error, setError] = useState('');
  const [saving, setSaving] = useState(false);

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    if (!name.trim() || !location.trim()) {
      setError('이름과 설치 위치를 모두 입력해 주세요.');
      return;
    }
    setError('');
    setSaving(true);
    await onSave({ name, location });
  };

  return <div className="modal-backdrop" onMouseDown={onClose}>
    <form className="register-modal" role="dialog" aria-modal="true" aria-label="로봇 설정" onMouseDown={(event) => event.stopPropagation()} onSubmit={submit}>
      <div className="modal-heading"><div><h2>로봇 설정</h2><p>로봇의 기본 정보를 변경하세요.</p></div><button type="button" className="close-button" onClick={onClose} aria-label="닫기">×</button></div>
      <div className="form-grid">
        <label>로봇 이름<input value={name} disabled={saving} onChange={(event) => setName(event.target.value)} placeholder="예: 5번 피킹 로봇" /></label>
        <label>설치 위치<input value={location} disabled={saving} onChange={(event) => setLocation(event.target.value)} placeholder="예: 물류 창고 2층" /></label>
      </div>
      {error && <p className="form-error" role="alert">{error}</p>}
      <div className="card-actions"><button type="button" className="secondary-button" disabled={saving} onClick={onClose}>취소</button><button className="primary-button" disabled={saving}>{saving ? '저장 중...' : '설정 저장'}</button></div>
    </form>
  </div>;
}
