'use client';

import { Pagination } from '@/components/Pagination';
import { usePagination } from '@/hooks/usePagination';
import { useState } from 'react';
import type { SavedWork } from '@/types';
import { useAppStore } from '@/hooks/useAppStore';
import { isSkillBlock, flattenWorkflowBlocks, formatWorkDuration, formatUpdatedAt } from '@/lib/workflow';
import { ActiveRobotExecutionsCard } from '@/components/ActiveRobotExecutionsCard';

type WorkSortKey = 'recent' | 'oldest' | 'name';
const WORK_SORT_OPTIONS: { value: WorkSortKey; label: string }[] = [
  { value: 'recent', label: '최근 수정순' },
  { value: 'oldest', label: '오래된 수정순' },
  { value: 'name', label: '이름순' },
];
/** 작업 목록 정렬. updatedAt이 비어있는 항목은 날짜 정렬에서 항상 뒤로 보낸다. */
function sortWorks(items: SavedWork[], sort: WorkSortKey): SavedWork[] {
  const byName = (a: SavedWork, b: SavedWork) => a.name.localeCompare(b.name, 'ko-KR');
  const ts = (w: SavedWork) => (w.updatedAt ? Date.parse(w.updatedAt) : null);
  return [...items].sort((a, b) => {
    if (sort === 'name') return byName(a, b);
    const ta = ts(a), tb = ts(b);
    if (ta === null && tb === null) return byName(a, b);
    if (ta === null) return 1;
    if (tb === null) return -1;
    return sort === 'recent' ? tb - ta : ta - tb;
  });
}

/** 작업 목록. 행을 누르면 작업 상세 라우트로 이동한다(배포·실행은 상세에서). */
export function WorkflowLibraryView({ onEdit, onOpen, onToast }: { onEdit: (workId: string) => void; onOpen: (workId: string, robotId?: string) => void; onToast: (message: string) => void }) {
  const { savedWorks, deleteWorksById } = useAppStore();
  const [query, setQuery] = useState('');
  const [workSort, setWorkSort] = useState<WorkSortKey>('recent');
  const [selectingWorks, setSelectingWorks] = useState(false);
  const [selectedWorkIds, setSelectedWorkIds] = useState<string[]>([]);
  const [deleteWorksOpen, setDeleteWorksOpen] = useState(false);
  const [deletingWorks, setDeletingWorks] = useState(false);

  const filteredWorks = savedWorks.filter((work) => {
    const blocks = flattenWorkflowBlocks(work.blocks);
    const normalizedQuery = query.trim().toLocaleLowerCase('ko-KR');
    return !normalizedQuery || [work.name, ...blocks.map((block) => block.title)].some((value) => value.toLocaleLowerCase('ko-KR').includes(normalizedQuery));
  });
  const sortedWorks = sortWorks(filteredWorks, workSort);
  const { page, pageCount, pageItems, setPage } = usePagination(sortedWorks);

  const deleteSelectedWorks = async () => {
    setDeletingWorks(true);
    try {
      await deleteWorksById(selectedWorkIds);
      onToast(`${selectedWorkIds.length}개 작업을 삭제했어요.`);
      setSelectedWorkIds([]); setSelectingWorks(false); setDeleteWorksOpen(false);
    } catch { onToast('삭제 API가 아직 연결되지 않았거나 삭제에 실패했어요.'); }
    finally { setDeletingWorks(false); }
  };

  if (!savedWorks.length) {
    return <><div className="page-heading library-page-heading title-divider"><div><h1>작업 목록</h1></div><button className="primary-button page-create-button" type="button" onClick={() => onEdit('')}>작업 추가</button></div><ActiveRobotExecutionsCard onOpenWork={onOpen} /><div className="center-panel"><section className="card state-card"><span className="state-icon">i</span><h2>저장된 작업이 없어요</h2><p>첫 작업을 만들어 로봇이 할 일을 구성해 보세요.</p></section></div></>;
  }

  return <>
    <div className="page-heading library-page-heading title-divider"><div><h1>작업 목록</h1></div><button className="primary-button page-create-button" type="button" onClick={() => onEdit('')}>작업 추가</button></div>
    <ActiveRobotExecutionsCard onOpenWork={onOpen} />

    <section className="workflow-library-toolbar" aria-label="작업 검색">
      {selectingWorks && <div className="selection-summary"><input className="list-select-checkbox" type="checkbox" aria-label="작업 전체 선택" disabled={!filteredWorks.length} checked={filteredWorks.length > 0 && selectedWorkIds.length === filteredWorks.length} onChange={() => setSelectedWorkIds(selectedWorkIds.length === filteredWorks.length ? [] : filteredWorks.map(work => work.id))} /><span>{selectedWorkIds.length}개 선택됨</span></div>}
      <button className={`${selectingWorks ? 'danger-button' : 'secondary-button'} list-delete-button`} onClick={() => { if (!selectingWorks) setSelectingWorks(true); else if (selectedWorkIds.length) setDeleteWorksOpen(true); else setSelectingWorks(false); }}>삭제</button>
      <label className="workflow-search"><span aria-hidden="true">⌕</span><input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="작업명 또는 포함된 블록 검색" aria-label="저장된 작업 검색" /></label>
      <select className="work-sort-select" value={workSort} onChange={(event) => setWorkSort(event.target.value as WorkSortKey)} aria-label="작업 정렬">{WORK_SORT_OPTIONS.map((option) => <option key={option.value} value={option.value}>{option.label}</option>)}</select>
    </section>

    <div className="workflow-library-layout">
      <section className="card saved-work-list" aria-label="저장된 작업 목록">
        <div className="saved-work-columns" aria-hidden="true"><span>작업</span><span>예상 시간</span><span>최근 수정일</span></div>
        {filteredWorks.length ? pageItems.map((work) => {
          const workBlocks = flattenWorkflowBlocks(work.blocks);
          const workSkills = workBlocks.filter(isSkillBlock).length;
          const workControls = workBlocks.length - workSkills;
          return <div key={work.id} className="saved-work-select-row">{selectingWorks && <input className="list-select-checkbox" type="checkbox" aria-label={`${work.name} 선택`} checked={selectedWorkIds.includes(work.id)} onChange={() => setSelectedWorkIds(current => current.includes(work.id) ? current.filter(id => id !== work.id) : [...current, work.id])} />}<button className="saved-work-row" onClick={() => selectingWorks ? setSelectedWorkIds(current => current.includes(work.id) ? current.filter(id => id !== work.id) : [...current, work.id]) : onOpen(work.id)}>
            <span className="saved-work-main"><i>W</i><span><strong>{work.name}</strong><small>{workSkills ? `학습 동작 ${workSkills}개` : '학습 동작 없음'} · 제어 블록 {workControls}개</small><em>{workBlocks.slice(0, 3).map((block) => block.title).join(' · ')}{workBlocks.length > 3 ? ` 외 ${workBlocks.length - 3}개` : ''}</em></span></span>
            <span className="saved-work-duration">{formatWorkDuration(work.estimatedSeconds)}</span>
            <span className="saved-work-updated">{formatUpdatedAt(work.updatedAt)}</span>
            <span className="saved-work-chevron">›</span>
          </button></div>;
        }) : <div className="library-empty"><strong>검색 결과가 없어요</strong><p>다른 작업명이나 동작을 검색해 보세요.</p></div>}
        <Pagination page={page} pageCount={pageCount} onChange={setPage} />
      </section>

    </div>
    
    {deleteWorksOpen && <div className="modal-backdrop" onMouseDown={() => !deletingWorks && setDeleteWorksOpen(false)}><section className="confirm-modal" role="dialog" aria-modal="true" aria-label="작업 선택 삭제 확인" onMouseDown={(e) => e.stopPropagation()}><div className="modal-heading"><div><h2>선택한 작업을 삭제할까요?</h2><p>{selectedWorkIds.length}개 작업이 삭제됩니다.</p></div><button className="close-button" onClick={() => setDeleteWorksOpen(false)} aria-label="닫기">×</button></div><div className="card-actions"><button className="secondary-button" disabled={deletingWorks} onClick={() => setDeleteWorksOpen(false)}>돌아가기</button><button className="danger-button" disabled={deletingWorks} onClick={() => void deleteSelectedWorks()}>{deletingWorks ? '삭제 중...' : '삭제'}</button></div></section></div>}
  </>;
}
