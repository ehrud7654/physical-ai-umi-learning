'use client';

import { useState, useEffect, useCallback } from 'react';
import type { RobotExecution } from '@/types';
import { useAppStore } from '@/hooks/useAppStore';
import { fetchRobotExecutions } from '@/services';
const activeExecutionStatuses: RobotExecution['status'][] = ['REQUESTED', 'STARTING', 'RUNNING', 'CANCEL_REQUESTED'];

export function ActiveRobotExecutionsCard({ onOpenWork }: { onOpenWork?: (workId: string, robotId: string) => void }) {
  const { robots, savedWorks } = useAppStore();
  const [robotFilter, setRobotFilter] = useState('all');
  const [executions, setExecutions] = useState<RobotExecution[]>([]);
  const [loading, setLoading] = useState(true);
  const [loadFailed, setLoadFailed] = useState(false);

  const loadExecutions = useCallback(async () => {
    const registeredRobots = robots.filter((robot) => robot.robotId);
    if (!registeredRobots.length) {
      setExecutions([]);
      setLoading(false);
      return;
    }
    const results = await Promise.allSettled(
      registeredRobots.map((robot) => fetchRobotExecutions(robot.robotId!)),
    );
    const successful = results.flatMap((result) => result.status === 'fulfilled' ? result.value.data : []);
    setExecutions(successful.filter((execution) => activeExecutionStatuses.includes(execution.status)));
    setLoadFailed(results.some((result) => result.status === 'rejected'));
    setLoading(false);
  }, [robots]);

  /* eslint-disable react-hooks/set-state-in-effect -- 진입 시 실행 목록을 조회하고 5초 주기로 갱신한다. */
  useEffect(() => {
    void loadExecutions();
    const timer = window.setInterval(() => void loadExecutions(), 5000);
    return () => window.clearInterval(timer);
  }, [loadExecutions]);
  /* eslint-enable react-hooks/set-state-in-effect */

  const visible = robotFilter === 'all'
    ? executions
    : executions.filter((execution) => execution.robotId === robotFilter);
  const executionStatus = (status: RobotExecution['status']) => {
    if (status === 'REQUESTED') return '실행 요청';
    if (status === 'STARTING') return '시작 중';
    if (status === 'CANCEL_REQUESTED') return '중지 요청';
    return '실행 중';
  };

  return <section className="card active-execution-card" aria-label="현재 실행 중인 작업">
    <div className="active-execution-heading"><div><h2>현재 실행 중인 작업{visible.length > 0 && <span className="training-badge active-execution-count">{visible.length}개 실행 중</span>}</h2><p>로봇에서 수행 중인 작업을 확인할 수 있어요.</p></div><select value={robotFilter} onChange={(event) => setRobotFilter(event.target.value)} aria-label="실행 작업 로봇 필터"><option value="all">전체 로봇</option>{robots.filter((robot) => robot.robotId).map((robot) => <option key={robot.id} value={robot.robotId}>{robot.name}</option>)}</select></div>
    <div className="active-execution-list">
      {visible.map((execution) => {
        const robot = robots.find((item) => item.robotId === execution.robotId);
        const work = savedWorks.find((item) => item.taskVersionId === execution.taskVersionId);
        const content = <><span className="robot-tile">R</span><div><strong>{work?.name ?? '작업 이름을 확인할 수 없습니다'}</strong><small>{robot?.name ?? '알 수 없는 로봇'} · {execution.currentStep == null ? '실행 준비' : `${execution.currentStep}/${execution.totalSteps} 단계`}</small></div><span className="training-badge">{executionStatus(execution.status)}</span><div className="active-execution-progress"><div className="tiny-progress"><i style={{ width: `${execution.progressPercent}%` }} /></div><em>{Math.round(execution.progressPercent)}%</em></div></>;
        return work && onOpenWork
          ? <button key={execution.id} className="active-execution-item" type="button" onClick={() => onOpenWork(work.id, execution.robotId)} aria-label={`${work.name} 실행 제어 화면 열기`}>{content}</button>
          : <article key={execution.id}>{content}</article>;
      })}
      {!loading && !visible.length && <p className="table-empty">{robotFilter === 'all' ? '현재 실행 중인 작업이 없어요.' : '선택한 로봇에서 실행 중인 작업이 없어요.'}</p>}
      {loading && <p className="table-empty">실행 중인 작업을 확인하고 있어요.</p>}
      {loadFailed && <p className="active-execution-warning">일부 로봇의 실행 상태를 불러오지 못했어요.</p>}
    </div>
  </section>;
}
