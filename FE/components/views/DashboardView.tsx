'use client';

import { useState, useEffect, useCallback } from 'react';
import type { DashboardData, DashboardRobotStatus, DeviceType, RobotStatus, TrainingStatus, View } from '@/types';
import { fetchDashboard } from '@/services';
import { trainingStatusLabel } from '@/services/trainingService';
import { ActiveRobotExecutionsCard } from '@/components/ActiveRobotExecutionsCard';
export function DashboardView({ onNavigate, onOpenDevices, onSelectRobot, onOpenWork }: { onNavigate: (view: View) => void; onOpenDevices: (tab: DeviceType) => void; onSelectRobot: (id: string) => void; onOpenWork: (workId: string, robotId: string) => void }) {
  const [dashboard, setDashboard] = useState<DashboardData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(false);
  const loadDashboard = useCallback(async () => {
    setLoading(true);
    setError(false);
    try { setDashboard(await fetchDashboard()); }
    catch { setError(true); }
    finally { setLoading(false); }
  }, []);
  /* eslint-disable react-hooks/set-state-in-effect -- 화면 진입 시 서버 대시보드를 조회한다. */
  useEffect(() => { void loadDashboard(); }, [loadDashboard]);
  /* eslint-enable react-hooks/set-state-in-effect */

  if (loading) return <div className="center-panel"><section className="card state-card"><span className="spinner" aria-hidden="true" /><h2>대시보드를 불러오는 중이에요</h2></section></div>;
  if (error || !dashboard) return <div className="center-panel"><section className="card state-card"><span className="state-icon">!</span><h2>대시보드를 불러오지 못했어요</h2><button className="primary-button" onClick={() => void loadDashboard()}>다시 시도</button></section></div>;

  const { summary, robots, recentTrainings } = dashboard;
  return <>
    <div className="page-heading title-divider"><div><h1>대시 보드</h1></div></div>
    <ActiveRobotExecutionsCard onOpenWork={onOpenWork} />
    <div className="overview-cards">
      <button onClick={() => onOpenDevices('umi')}><div><small>등록된 수집 핸들</small><strong>{summary.umi.totalCount}<em>개</em></strong></div></button>
      <button onClick={() => onOpenDevices('robot')}><div><small>등록된 로봇</small><strong>{summary.robots.totalCount}<em>개</em></strong><p><i className="online-dot" /> {summary.robots.availableCount}개 사용 가능</p></div></button>
      <button onClick={() => onNavigate('training')}><div><small>진행 중인 학습</small><strong>{summary.trainings.runningCount}<em>건</em></strong><p>평균 진행률 {Math.round(summary.trainings.averageProgressPercent)}%</p></div></button>
      <button onClick={() => onNavigate('library')}><div><small>오늘 완료한 작업</small><strong>{summary.executions.completedTodayCount}<em>건</em></strong><p>성공률 {Math.round(summary.executions.successRatePercent)}%</p></div></button>
    </div>
    <div className="dashboard-grid">
      <section className="card dashboard-section"><div className="dashboard-title"><div><h2>로봇 상태</h2></div><button className="text-button" onClick={() => onOpenDevices('robot')}>전체 보기 →</button></div><div className="compact-robot-list">{robots.map(robot => { const label = dashboardRobotStatusLabel[robot.status]; return <button key={robot.id} onClick={() => onSelectRobot(robot.id)}><span className="robot-tile">R</span><div><strong>{robot.name}</strong><small>{formatDashboardTime(robot.lastSeenAt)}</small></div><span className={`status-pill status-${label.replace(' ', '-')}`}><i />{label}</span><em>›</em></button>; })}{!robots.length && <p className="table-empty">등록된 로봇이 없어요.</p>}</div></section>
      <section className="card dashboard-section"><div className="dashboard-title"><div><h2>최근 학습</h2></div><button className="text-button" onClick={() => onNavigate('training')}>전체 보기 →</button></div><div className="compact-training-list">{recentTrainings.map(item => { const status = dashboardTrainingStatus(item.status); return <button key={item.id} onClick={() => onNavigate('training')}><div><strong>{item.skillName}</strong><small>{item.umiDeviceName} · {trainingStatusLabel(item.status)}</small></div><span className={`training-badge ${status === '실패' ? 'failed' : status === '완료' ? 'complete' : ''}`}>{status}</span><div className="compact-training-progress"><div className="tiny-progress"><i style={{width: `${item.progressPercent}%`}} /></div><em>{Math.round(item.progressPercent)}%</em></div></button>; })}{!recentTrainings.length && <p className="table-empty">최근 학습이 없어요.</p>}</div></section>
    </div>
  </>;
}

const dashboardRobotStatusLabel: Record<DashboardRobotStatus, RobotStatus> = {
  AVAILABLE: '온라인', BUSY: '작업 중', OFFLINE: '오프라인', UNKNOWN: '상태 확인 불가',
};

function dashboardTrainingStatus(status: string): TrainingStatus {
  if (status === 'SUCCEEDED') return '완료';
  if (status === 'FAILED') return '실패';
  if (status === 'CANCELED') return '취소됨';
  if (status === 'CANCEL_REQUESTED') return '취소 요청';
  return '진행 중';
}

function formatDashboardTime(value: string | null): string {
  if (!value) return '통신 기록 없음';
  return new Intl.DateTimeFormat('ko-KR', { month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit' }).format(new Date(value));
}
