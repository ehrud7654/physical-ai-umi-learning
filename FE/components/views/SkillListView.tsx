'use client';

import { deploymentStatusMeta } from '@/lib/datasets';
import { type FormEvent, useState, useEffect, useCallback, useMemo } from 'react';
import type { SkillOverview } from '@/types';
import { createCustomSkill, fetchSkillOverviews } from '@/services';
import { formatApiDateTimeParts } from '@/lib/workflow';
import { Pagination } from '@/components/Pagination';
import { usePagination } from '@/hooks/usePagination';

type SortKey = 'recent' | 'oldest' | 'name' | 'status';
const SORT_OPTIONS: { value: SortKey; label: string }[] = [
  { value: 'recent', label: '최신 업데이트순' },
  { value: 'oldest', label: '오래된 업데이트순' },
  { value: 'name', label: '이름순' },
  { value: 'status', label: '상태순' },
];
/** 배포 상태 정렬 순서 — 실행 가능이 맨 앞, 검증 실패가 맨 뒤. */
const STATUS_ORDER: Record<SkillOverview['deploymentStatus'], number> = { DEPLOYED: 0, TRAINING: 1, COLLECTING: 2, PLANNED: 3, GATE_FAILED: 4 };
/** 동작 목록 정렬. 버전이 없는 항목(생성 시각 null)은 날짜 정렬에서 항상 뒤로 보낸다. */
function sortSkills(items: SkillOverview[], sort: SortKey): SkillOverview[] {
  const byName = (a: SkillOverview, b: SkillOverview) => a.displayName.localeCompare(b.displayName, 'ko-KR');
  const ts = (s: SkillOverview) => (s.latestVersionCreatedAt ? Date.parse(s.latestVersionCreatedAt) : null);
  return [...items].sort((a, b) => {
    if (sort === 'name') return byName(a, b);
    if (sort === 'status') return STATUS_ORDER[a.deploymentStatus] - STATUS_ORDER[b.deploymentStatus] || byName(a, b);
    const ta = ts(a), tb = ts(b);
    if (ta === null && tb === null) return byName(a, b);
    if (ta === null) return 1;
    if (tb === null) return -1;
    return sort === 'recent' ? tb - ta : ta - tb;
  });
}

export function SkillListView({ onToast, onOpen }: { onToast: (message: string) => void; onOpen: (skillId: string) => void }) {
  const [skills, setSkills] = useState<SkillOverview[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [createOpen, setCreateOpen] = useState(false);
  const [displayName, setDisplayName] = useState('');
  const [description, setDescription] = useState('');
  const [createError, setCreateError] = useState('');
  const [creating, setCreating] = useState(false);
  const [query, setQuery] = useState('');
  const [sort, setSort] = useState<SortKey>('recent');
  const keyword = query.trim().toLowerCase();
  const filtered = useMemo(() => keyword
    ? skills.filter((skill) => skill.displayName.toLowerCase().includes(keyword) || (skill.description ?? '').toLowerCase().includes(keyword))
    : skills, [skills, keyword]);
  const sorted = useMemo(() => sortSkills(filtered, sort), [filtered, sort]);
  const { page, pageCount, pageItems, setPage } = usePagination(sorted);

  const loadSkills = useCallback(async () => {
    setLoading(true);
    setError('');
    try {
      setSkills(await fetchSkillOverviews());
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : '동작 목록을 불러오지 못했어요.');
      onToast('동작 목록을 불러오지 못했어요.');
    } finally {
      setLoading(false);
    }
  }, [onToast]);

  /* eslint-disable react-hooks/set-state-in-effect -- 화면 진입 시 동작 목록을 조회한다. */
  useEffect(() => {
    void loadSkills();
  }, [loadSkills]);
  /* eslint-enable react-hooks/set-state-in-effect */

  const submitSkill = async (event: FormEvent) => {
    event.preventDefault();
    const name = displayName.trim();
    if (!name) {
      setCreateError('동작 이름을 입력해 주세요.');
      return;
    }
    setCreating(true);
    setCreateError('');
    try {
      await createCustomSkill({ displayName: name, description });
      await loadSkills();
      setCreateOpen(false);
      setDisplayName('');
      setDescription('');
      onToast('동작이 추가되었어요.');
    } catch (cause) {
      setCreateError(cause instanceof Error ? cause.message : '동작을 추가하지 못했어요.');
    } finally {
      setCreating(false);
    }
  };

  return <>
    <div className="page-heading data-page-heading">
      <div>
        <h1>동작 목록</h1>
        {loading && <p role="status">등록된 동작을 불러오는 중이에요.</p>}
        {error && <p className="form-error" role="alert">동작 목록을 불러오지 못했어요. <button className="text-button" type="button" onClick={() => void loadSkills()}>다시 시도</button></p>}
      </div>
      <button className="primary-button page-create-button" type="button" onClick={() => { setCreateError(''); setCreateOpen(true); }}>동작 추가</button>
    </div>
    <section className="workflow-library-toolbar" aria-label="동작 검색·정렬">
      <label className="workflow-search"><span aria-hidden="true">⌕</span><input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="동작 이름 또는 설명 검색" aria-label="동작 검색" /></label>
      <select className="skill-sort-select" value={sort} onChange={(event) => setSort(event.target.value as SortKey)} aria-label="동작 정렬">{SORT_OPTIONS.map((option) => <option key={option.value} value={option.value}>{option.label}</option>)}</select>
    </section>
    <section className="card episode-panel data-table-panel">
      <div className="episode-table" role="table" aria-label="동작 목록">
        <div className="episode-row skill-list-row header" role="row">
          <span>동작 이름</span><span>설명</span><span>최신 버전</span><span>마지막 업데이트</span><span>상태</span><span>최근 학습</span>
        </div>
        {pageItems.map((skill) => { const deployment = deploymentStatusMeta[skill.deploymentStatus]; const versionCreatedAt = skill.latestVersionCreatedAt ? formatApiDateTimeParts(skill.latestVersionCreatedAt) : null; const latestTrainingRequestedAt = skill.latestTrainingRequestedAt ? formatApiDateTimeParts(skill.latestTrainingRequestedAt) : null; return <div className="episode-row skill-list-row" role="row" key={skill.id}>
          <span className="skill-list-name" data-label="동작 이름"><button className="text-button skill-name-button" type="button" onClick={() => onOpen(skill.id)}>{skill.displayName}</button></span>
          <span className="skill-list-desc" data-label="설명" title={skill.description || undefined}>{skill.description || '-'}</span>
          <span className="skill-list-version" data-label="최신 버전">{skill.latestVersion == null ? '-' : `v${skill.latestVersion}`}</span>
          <span className="skill-list-datetime" data-label="마지막 업데이트">{versionCreatedAt ? <><strong>{versionCreatedAt.date}</strong><small>{versionCreatedAt.time}</small></> : '-'}</span>
          <span data-label="상태"><span className={`dataset-status ${deployment.className}`}><i />{deployment.label}</span></span>
          <span className="skill-list-datetime" data-label="최근 학습">{latestTrainingRequestedAt ? <><strong>{latestTrainingRequestedAt.date}</strong><small>{latestTrainingRequestedAt.time}</small></> : '-'}</span>
        </div>; })}
        {!loading && !error && !filtered.length && <div className="table-empty"><p>{skills.length ? '검색 결과가 없어요.' : '등록된 동작이 없어요.'}</p></div>}
      </div>
      <Pagination page={page} pageCount={pageCount} onChange={setPage} />
    </section>
    {createOpen && <div className="modal-backdrop" onMouseDown={() => !creating && setCreateOpen(false)}>
      <form className="register-modal skill-create-modal" role="dialog" aria-modal="true" aria-label="동작 추가" onSubmit={submitSkill} onMouseDown={(event) => event.stopPropagation()}>
        <div className="modal-heading"><div><p>새로운 커스텀 동작</p><h2>동작 추가</h2></div><button className="close-button" type="button" aria-label="닫기" disabled={creating} onClick={() => setCreateOpen(false)}>×</button></div>
        <label>동작 이름 <strong aria-hidden="true">*</strong><input maxLength={150} value={displayName} onChange={(event) => { setDisplayName(event.target.value); setCreateError(''); }} placeholder="예: 백기잡기" autoFocus required /></label>
        <label>설명 <textarea maxLength={2000} value={description} onChange={(event) => setDescription(event.target.value)} placeholder="동작에 대한 설명을 입력해 주세요. (선택)" rows={5} /></label>
        <p className="modal-note">동작 키, 분류와 상태는 서버에서 자동으로 생성됩니다.</p>
        {createError && <p className="form-error" role="alert">{createError}</p>}
        <div className="modal-actions"><button className="secondary-button" type="button" disabled={creating} onClick={() => setCreateOpen(false)}>취소</button><button className="primary-button" disabled={creating}>{creating ? '추가 중...' : '추가'}</button></div>
      </form>
    </div>}
  </>;
}
