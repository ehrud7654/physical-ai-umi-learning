'use client';

import { useState } from 'react';
import type { TrainingStatus, View } from '@/types';
import { Pagination } from '@/components/Pagination';
import { usePagination } from '@/hooks/usePagination';
import { useAppStore } from '@/hooks/useAppStore';
/** 전체 학습 현황(읽기 중심). 학습 시작은 동작 상세에서, 개별 학습의 취소·재시도는 학습 상세 라우트에서 한다. */
export function TrainingStatusView({ onNavigate, onToast, onOpen }: { onNavigate: (view: View) => void; onToast: (message: string) => void; onOpen: (trainingId: string) => void }) {
  const { trainings, trainingRefreshing, trainingRefreshError, trainingUpdatedAt, refreshTrainings, deleteTrainingsById } = useAppStore();
  const [statusFilter, setStatusFilter] = useState<'all' | TrainingStatus>('all');
  const [busy, setBusy] = useState(false);
  const [selecting, setSelecting] = useState(false);
  const [selectedIds, setSelectedIds] = useState<string[]>([]);
  const [deleteOpen, setDeleteOpen] = useState(false);
  const countOf = (status: TrainingStatus) => trainings.filter((item) => item.status === status).length;
  const filtered = statusFilter === 'all' ? trainings : trainings.filter((item) => item.status === statusFilter);
  const { page, pageCount, pageItems, setPage } = usePagination(filtered);
  const refreshStatus = async () => {
    const refreshed = await refreshTrainings();
    onToast(refreshed ? '학습 상태를 최신 정보로 갱신했어요.' : '학습 상태를 불러오지 못했어요. 다시 시도해 주세요.');
  };
  const deletable = filtered.filter((item) => item.status !== '진행 중' && item.status !== '취소 요청');
  const deleteSelected = async () => {
    setBusy(true);
    try {
      await deleteTrainingsById(selectedIds);
      onToast(`${selectedIds.length}개 학습을 삭제했어요.`);
      setSelectedIds([]);
      setSelecting(false);
      setDeleteOpen(false);
    } catch { onToast('삭제 API가 아직 연결되지 않았거나 삭제에 실패했어요.'); }
    finally { setBusy(false); }
  };
  const updatedLabel = trainingUpdatedAt
    ? new Intl.DateTimeFormat('ko-KR', { hour: '2-digit', minute: '2-digit', second: '2-digit' }).format(trainingUpdatedAt)
    : '확인 중';

  return <>
    <div className="page-heading title-divider"><div><h1>학습 현황</h1><p>진행 중이거나 완료한 학습을 한눈에 확인해요. 새 학습은 동작에서 시작해요.</p></div></div>
    <div className="training-summary" aria-label="학습 상태별 보기"><button className={statusFilter === 'all' ? 'active' : ''} onClick={() => setStatusFilter('all')}><small>전체</small><strong>{trainings.length}</strong></button><button className={statusFilter === '진행 중' ? 'active' : ''} onClick={() => setStatusFilter('진행 중')}><small>진행 중</small><strong>{countOf('진행 중')}</strong></button><button className={statusFilter === '완료' ? 'active' : ''} onClick={() => setStatusFilter('완료')}><small>완료</small><strong>{countOf('완료')}</strong></button><button className={statusFilter === '실패' ? 'active' : ''} onClick={() => setStatusFilter('실패')}><small>확인 필요</small><strong>{countOf('실패')}</strong></button></div>
    <section className="card training-table">
      <div className="training-table-head"><div className="training-controls">{selecting && <input className="list-select-checkbox" type="checkbox" aria-label="삭제 가능한 학습 전체 선택" disabled={!deletable.length} checked={deletable.length > 0 && selectedIds.length === deletable.length} onChange={() => setSelectedIds(selectedIds.length === deletable.length ? [] : deletable.map(item => item.id))} />}{selecting && <span className="selected-count">{selectedIds.length}개 선택됨</span>}<button className={selecting ? 'danger-button list-delete-button' : 'secondary-button'} onClick={() => { if (!selecting) setSelecting(true); else if (selectedIds.length) setDeleteOpen(true); else setSelecting(false); }}>삭제</button><select value={statusFilter} onChange={(e) => setStatusFilter(e.target.value as 'all' | TrainingStatus)} aria-label="학습 상태 필터"><option value="all">전체 상태</option><option value="진행 중">진행 중</option><option value="취소 요청">취소 요청</option><option value="취소됨">취소됨</option><option value="완료">완료</option><option value="실패">실패</option></select></div><div className="training-controls training-controls-status"><p className={`refresh-status ${trainingRefreshError ? 'failed' : ''}`} role="status">{trainingRefreshError ? '마지막 갱신에 실패했어요.' : `마지막 갱신 ${updatedLabel}`}</p><button className="secondary-button refresh-button" disabled={trainingRefreshing} onClick={() => void refreshStatus()}>{trainingRefreshing ? '갱신 중...' : '↻ 새로고침'}</button></div></div>
      {pageItems.map(item => <article key={item.id}>
        {selecting && <input className="list-select-checkbox" type="checkbox" aria-label={`${item.name} 선택`} disabled={item.status === '진행 중' || item.status === '취소 요청'} checked={selectedIds.includes(item.id)} onChange={() => setSelectedIds(current => current.includes(item.id) ? current.filter(id => id !== item.id) : [...current, item.id])} />}
        <div className="training-main"><span className="learning-symbol">✦</span><div><button className="training-title-button" type="button" onClick={() => onOpen(item.id)}>{item.name}</button><small>{item.robot} · {item.started}</small></div></div>
        <div><small>현재 단계</small><strong>{item.stage}</strong></div>
        <div className="training-progress"><div><i style={{width: `${item.progress}%`}} /></div><strong>{item.progress}%</strong></div>
        <span className={`training-badge ${item.status === '실패' ? 'failed' : item.status === '완료' ? 'complete' : ''}`}>{item.status}</span>
      </article>)}
      {!filtered.length && <div className="table-empty empty-with-action"><p>{trainings.length ? '선택한 상태의 학습이 없어요.' : '아직 진행한 학습이 없어요. 동작에서 학습을 시작해 보세요.'}</p><button className="secondary-button" onClick={() => trainings.length ? setStatusFilter('all') : onNavigate('skills')}>{trainings.length ? '전체 상태 보기' : '동작 목록으로 이동'}</button></div>}
      <Pagination page={page} pageCount={pageCount} onChange={setPage} />
    </section>
    {deleteOpen && <div className="modal-backdrop" onMouseDown={() => !busy && setDeleteOpen(false)}><section className="confirm-modal" role="dialog" aria-modal="true" aria-label="학습 선택 삭제 확인" onMouseDown={(e) => e.stopPropagation()}><div className="modal-heading"><div><h2>선택한 학습을 삭제할까요?</h2><p>{selectedIds.length}개 학습이 삭제됩니다.</p></div><button className="close-button" onClick={() => setDeleteOpen(false)} aria-label="닫기">×</button></div><div className="card-actions"><button className="secondary-button" disabled={busy} onClick={() => setDeleteOpen(false)}>돌아가기</button><button className="danger-button" disabled={busy} onClick={() => void deleteSelected()}>{busy ? '삭제 중...' : '삭제'}</button></div></section></div>}
  </>;
}
