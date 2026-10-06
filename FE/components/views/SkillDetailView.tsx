'use client';

import { type FormEvent, useCallback, useEffect, useState } from 'react';
import type { SkillOverview, TrainingSkill } from '@/types';
import { useAppStore } from '@/hooks/useAppStore';
import { fetchSkillOverviews, fetchTrainingSkills, updateSkillDescription } from '@/services';
import { datasetStatusMeta, deploymentStatusMeta, recoveryActionLabel } from '@/lib/datasets';
import { DatasetWhen } from '@/components/DatasetWhen';
import { Pagination } from '@/components/Pagination';
import { usePagination } from '@/hooks/usePagination';

/**
 * 동작 하나의 허브: 이 동작으로 진행한 학습 현황과 수집 데이터를 보여주고, 여기서 학습을 시작한다.
 * 라우트 파라미터는 skill id. 에피소드·학습은 skillKey로 묶이므로 `/skills` 카탈로그에서 id → skillKey를 푼다
 * (`/skills/overview`는 배포판에 따라 skillKey가 없을 수 있다).
 */
export function SkillDetailView({ skillId, onBack, onStartTraining, onOpenTraining, onToast }: { skillId: string; onBack: () => void; onStartTraining: (skillKey: string) => void; onOpenTraining: (trainingId: string) => void; onToast: (message: string) => void }) {
  const { datasets, trainings } = useAppStore();
  const [skill, setSkill] = useState<TrainingSkill | null>(null);
  const [overview, setOverview] = useState<SkillOverview | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(false);
  const [editOpen, setEditOpen] = useState(false);
  const [editDescription, setEditDescription] = useState('');
  const [editError, setEditError] = useState('');
  const [updating, setUpdating] = useState(false);

  const load = useCallback(async () => {
    setLoading(true);
    setError(false);
    try {
      const [catalog, overviews] = await Promise.all([fetchTrainingSkills(), fetchSkillOverviews()]);
      setSkill(catalog.find((item) => item.id === skillId) ?? null);
      setOverview(overviews.find((item) => item.id === skillId) ?? null);
    } catch {
      setError(true);
      onToast('동작 정보를 불러오지 못했어요.');
    } finally {
      setLoading(false);
    }
  }, [skillId, onToast]);

  /* eslint-disable react-hooks/set-state-in-effect -- 진입·동작 변경 시 동작 정보를 조회한다. */
  useEffect(() => {
    void load();
  }, [load]);
  /* eslint-enable react-hooks/set-state-in-effect */

  /* 콘솔 진입 시 store에 이미 로드된 데이터를 skillKey로 걸러 쓴다 — 상세 전용 목록 API 호출 없음. */
  const skillKey = skill?.skillKey ?? '';
  const skillTrainings = trainings.filter((item) => item.skillKey === skillKey);
  const skillDatasets = datasets.filter((item) => item.skillKey === skillKey);
  const attentionCount = skillDatasets.filter((item) => item.status === 'INVALID' || item.status === 'DELETE_FAILED').length;
  /* 학습 현황은 보조 목록이라 5개, 수집 데이터는 한 번에 여러 개씩 찍어 비교하며 보므로 10개 단위. */
  const dataPage = usePagination(skillDatasets, 10);
  const trainingPage = usePagination(skillTrainings, 5);

  const submitDescription = async (event: FormEvent) => {
    event.preventDefault();
    if (!skill) return;
    setUpdating(true);
    setEditError('');
    try {
      await updateSkillDescription(skill.id, editDescription);
      await load();
      setEditOpen(false);
      onToast('동작 설명이 수정되었어요.');
    } catch (cause) {
      setEditError(cause instanceof Error ? cause.message : '동작 설명을 수정하지 못했어요.');
    } finally {
      setUpdating(false);
    }
  };

  if (loading) {
    return <div className="center-panel"><section className="card state-card"><span className="spinner" aria-hidden="true" /><h2>동작 정보를 불러오는 중이에요</h2><p>잠시만 기다려 주세요.</p></section></div>;
  }
  if (error || !skill) {
    return <><button className="back-link" type="button" onClick={onBack}>← 동작 목록</button><div className="center-panel"><section className="card state-card"><span className="state-icon">!</span><h2>{error ? '동작 정보를 불러오지 못했어요' : '동작을 찾을 수 없어요'}</h2><p>{error ? '잠시 후 다시 시도해 주세요.' : '삭제되었거나 주소가 잘못되었을 수 있어요.'}</p>{error && <button className="secondary-button" type="button" onClick={() => void load()}>다시 시도</button>}</section></div></>;
  }

  const deployment = overview ? deploymentStatusMeta[overview.deploymentStatus] : null;

  return <>
    <button className="back-link" type="button" onClick={onBack}>← 동작 목록</button>
    <div className="page-heading list-page-heading title-divider">
      <div>
        <h1>{skill.displayName}</h1>
        <p className="skill-detail-description">{skill.description || '등록된 설명이 없어요.'} <button className="text-button" type="button" onClick={() => { setEditDescription(skill.description ?? ''); setEditError(''); setEditOpen(true); }}>설명 수정</button></p>
        {(deployment || overview?.latestVersion != null) && <p className="skill-detail-meta">{deployment && <span className={`dataset-status ${deployment.className}`}><i />{deployment.label}</span>}{overview?.latestVersion != null && <span>최신 버전 v{overview.latestVersion}</span>}</p>}
      </div>
      <button className="primary-button page-create-button" type="button" onClick={() => onStartTraining(skill.skillKey)}>학습 시작</button>
    </div>

    <section className="card training-table skill-detail-section" aria-label="학습 현황">
      <div className="training-table-head"><div><h2>학습 현황</h2><p>이 동작으로 진행한 학습이에요.</p></div></div>
      {trainingPage.pageItems.map((item) => <article key={item.id}>
        <div className="training-main"><span className="learning-symbol">✦</span><div><button className="training-title-button" type="button" onClick={() => onOpenTraining(item.id)}>{item.name}</button><small>{item.robot} · {item.started}</small></div></div>
        <div><small>현재 단계</small><strong>{item.stage}</strong></div>
        <div className="training-progress"><div><i style={{ width: `${item.progress}%` }} /></div><strong>{item.progress}%</strong></div>
        <span className={`training-badge ${item.status === '실패' ? 'failed' : item.status === '완료' ? 'complete' : ''}`}>{item.status}</span>
      </article>)}
      {!skillTrainings.length && <p className="table-empty">아직 이 동작으로 진행한 학습이 없어요. 수집 데이터를 확인하고 학습을 시작해 보세요.</p>}
      <Pagination page={trainingPage.page} pageCount={trainingPage.pageCount} onChange={trainingPage.setPage} />
    </section>

    <details className="card skill-data-panel" open={attentionCount > 0}>
      <summary>
        <span>수집 데이터 <strong>{skillDatasets.length}개</strong></span>
        {attentionCount > 0 && <span className="dataset-status invalid"><i />확인 필요 {attentionCount}개</span>}
        <span className="skill-data-toggle" aria-hidden="true">▾</span>
      </summary>
      <div className="skill-data-list">
        {skillDatasets.length > 0 && <div className="skill-data-row header" aria-hidden="true"><span>업로드 시각</span><span>수집 핸들</span><span className="dataset-length">길이</span><span>상태</span><span>보관</span></div>}
        {dataPage.pageItems.map((item) => { const meta = datasetStatusMeta[item.status]; const needsAttention = item.status === 'INVALID' || item.status === 'DELETE_FAILED'; return <div className="skill-data-row" key={item.id}>
          <DatasetWhen item={item} />
          <span className="dataset-handle">{item.umiId}</span>
          <span className="dataset-length">{item.duration}</span>
          <span><span className={`dataset-status ${meta.className}`}><i />{meta.label}</span></span>
          <span>{item.retentionDays}일 남음</span>
          {needsAttention && (item.blockingReasons[0] || item.recoveryAction !== 'NONE') && <small className="skill-data-reason">{item.blockingReasons[0]?.message ?? '확인이 필요해요.'}{item.recoveryAction !== 'NONE' ? ` · ${recoveryActionLabel[item.recoveryAction]}` : ''}</small>}
        </div>; })}
        {!skillDatasets.length && <p className="table-empty">아직 수집된 데이터가 없어요. 수집 앱에서 ‘{skill.displayName}’을 선택해 데이터를 수집해 주세요.</p>}
        <Pagination page={dataPage.page} pageCount={dataPage.pageCount} onChange={dataPage.setPage} />
      </div>
    </details>

    {editOpen && <div className="modal-backdrop" onMouseDown={() => !updating && setEditOpen(false)}>
      <form className="register-modal skill-create-modal" role="dialog" aria-modal="true" aria-label="동작 설명 수정" onSubmit={submitDescription} onMouseDown={(event) => event.stopPropagation()}>
        <div className="modal-heading"><div><p>동작 정보</p><h2>동작 설명 수정</h2></div><button className="close-button" type="button" aria-label="닫기" disabled={updating} onClick={() => setEditOpen(false)}>×</button></div>
        <label>동작 이름<input value={skill.displayName} disabled aria-readonly="true" /></label>
        <label>설명 <textarea maxLength={2000} value={editDescription} onChange={(event) => { setEditDescription(event.target.value); setEditError(''); }} placeholder="동작에 대한 설명을 입력해 주세요." rows={5} autoFocus /></label>
        {editError && <p className="form-error" role="alert">{editError}</p>}
        <div className="modal-actions"><button className="secondary-button" type="button" disabled={updating} onClick={() => setEditOpen(false)}>취소</button><button className="primary-button" disabled={updating}>{updating ? '수정 중...' : '수정'}</button></div>
      </form>
    </div>}
  </>;
}
