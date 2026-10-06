'use client';

import { recoveryActionLabel } from '@/lib/datasets';
import { DatasetWhen } from '@/components/DatasetWhen';
import { useState, useEffect } from 'react';
import type { Dataset, TrainingSkill, View } from '@/types';
import { useAppStore } from '@/hooks/useAppStore';
import { createTrainingJob, fetchTrainingSkills } from '@/services';
function isSelectableDataset(item: Dataset): boolean {
  return item.trainable;
}

/** presetSkillKey가 있으면(동작 상세에서 진입) 동작 선택 단계를 건너뛰고 데이터 선택부터 시작한다. */
export function LearnView({ onNavigate, onToast, presetSkillKey = null, onDone }: { onNavigate: (view: View) => void; onToast: (message: string) => void; presetSkillKey?: string | null; onDone?: (trainingId?: string) => void }) {
  const { umis, datasets, learningDraft, updateLearningDraft, resetLearningDraft, refreshDatasets, refreshTrainings } = useAppStore();
  const { step, skillKey, selected } = learningDraft;
  const [catalog, setCatalog] = useState<TrainingSkill[]>([]);
  const [catalogLoading, setCatalogLoading] = useState(true);
  const [catalogError, setCatalogError] = useState(false);
  const [submitting, setSubmitting] = useState(false);

  /* eslint-disable react-hooks/set-state-in-effect -- 화면 진입 시 학습 동작·데이터 카탈로그를 조회한다. */
  useEffect(() => {
    setCatalogLoading(true);
    setCatalogError(false);
    void Promise.all([fetchTrainingSkills(), refreshDatasets()])
      .then(([skills]) => setCatalog(skills))
      .catch(() => {
        setCatalogError(true);
        onToast('학습 동작과 수집 데이터를 불러오지 못했어요.');
      })
      .finally(() => setCatalogLoading(false));
  }, [onToast, refreshDatasets]);
  /* eslint-enable react-hooks/set-state-in-effect */

  /* 카탈로그에 실제로 있는 preset만 인정한다. 잘못된 키면 기존 3단계 흐름으로 돌아간다. */
  const preset = presetSkillKey && catalog.some((item) => item.skillKey === presetSkillKey) ? presetSkillKey : null;
  useEffect(() => {
    if (step < 1 || step > 3) {
      updateLearningDraft({ step: 1, method: 'existing', selected: [], trainingJobId: undefined, error: undefined });
      return;
    }
    if (!catalog.length) return;
    if (preset) {
      if (skillKey !== preset) updateLearningDraft({ step: 2, skillKey: preset, method: 'existing', selected: [], name: '' });
      else if (step === 1) updateLearningDraft({ step: 2, method: 'existing' });
      return;
    }
    if (!catalog.some((item) => item.skillKey === skillKey)) {
      updateLearningDraft({ step: 1, skillKey: catalog[0].skillKey, method: 'existing', selected: [] });
    }
  }, [catalog, skillKey, step, updateLearningDraft, preset]);

  const stepLabels = preset ? ['수집 데이터 선택', '최종 확인'] : ['동작 선택', '수집 데이터 선택', '최종 확인'];
  const stepOffset = preset ? 1 : 0;
  const shownStep = Math.max(1, step - stepOffset);
  const skill = catalog.find((item) => item.skillKey === skillKey);
  const availableData = datasets.filter((item) => item.skillKey === skillKey);
  const readyData = availableData.filter(isSelectableDataset);
  const selectedData = readyData.filter((item) => selected.includes(item.id));
  const allSelected = readyData.length > 0 && readyData.every((item) => selected.includes(item.id));
  useEffect(() => {
    if (catalogLoading || catalogError) return;
    const validIds = new Set(datasets.filter((item) => item.skillKey === skillKey && isSelectableDataset(item)).map((item) => item.id));
    const validSelected = selected.filter((id) => validIds.has(id));
    if (validSelected.length !== selected.length) updateLearningDraft({ selected: validSelected });
  }, [catalogLoading, catalogError, datasets, skillKey, selected, updateLearningDraft]);

  const selectSkill = (nextSkillKey: string) => {
    updateLearningDraft({ skillKey: nextSkillKey, method: 'existing', selected: [], name: '' });
  };
  const toggleEpisode = (episodeId: string) => {
    if (!readyData.some((item) => item.id === episodeId)) return;
    const nextSelected = selected.includes(episodeId)
      ? selected.filter((id) => id !== episodeId)
      : [...selected, episodeId];
    const nextData = readyData.filter((item) => nextSelected.includes(item.id));
    if (nextData.length && nextData.some((item) => item.umiDeviceId !== nextData[0].umiDeviceId)) {
      onToast('같은 수집 핸들에서 수집한 데이터만 함께 선택할 수 있어요.');
      return;
    }
    updateLearningDraft({ selected: nextSelected });
  };

  const submitTraining = async () => {
    if (!skill || !selectedData.length) return;
    const umi = umis.find((item) => item.deviceId === selectedData[0].umiDeviceId);
    if (!umi) {
      onToast('수집 데이터의 수집 핸들을 확인할 수 없어요.');
      return;
    }
    setSubmitting(true);
    try {
      const created = await createTrainingJob({
        skillKey: skill.skillKey,
        umiDeviceId: umi.deviceId,
        episodeIds: selectedData.map((item) => item.id),
      });
      resetLearningDraft();
      await refreshTrainings();
      onToast('학습 작업을 생성했어요.');
      if (onDone) onDone(created?.id);
      else onNavigate('training');
    } catch {
      onToast('학습 작업을 생성하지 못했어요. 선택한 데이터와 수집 핸들 정보를 확인해 주세요.');
    } finally {
      setSubmitting(false);
    }
  };

  if (catalogLoading) {
    return <div className="center-panel"><section className="card state-card"><span className="spinner" aria-hidden="true" /><h2>학습 데이터를 불러오는 중이에요</h2><p>잠시만 기다려 주세요.</p></section></div>;
  }
  if (catalogError) {
    return <div className="center-panel"><section className="card state-card"><span className="state-icon">!</span><h2>학습 데이터를 불러오지 못했어요</h2><p>페이지를 새로고침해 다시 시도해 주세요.</p></section></div>;
  }

  return <>
    <div className="page-heading learning-heading title-divider"><div><h1>학습 생성</h1>{preset && skill && <p>‘{skill.displayName}’ 동작으로 학습을 만들어요.</p>}</div></div>
    <p className="mobile-step-status" role="status">{shownStep} / {stepLabels.length} · {stepLabels[shownStep - 1]}</p>
    <ol className="steps redesigned-steps" aria-label="학습 생성 단계">
      {stepLabels.map((label, index) => <li key={label} className={shownStep === index + 1 ? 'current' : shownStep > index + 1 ? 'done' : ''}><span>{shownStep > index + 1 ? '✓' : index + 1}</span><div><strong>{label}</strong></div></li>)}
    </ol>

    {step === 1 && !preset && <section className="card learning-wizard-card"><div className="wizard-title"><span>1</span><div><h2>학습할 동작을 선택해 주세요</h2></div></div><div className="skill-choice-grid">{catalog.map((item, index) => <button key={item.id} className={skillKey === item.skillKey ? 'selected' : ''} onClick={() => selectSkill(item.skillKey)}><span>{item.displayName.trim().charAt(0) || index + 1}</span><div><strong>{item.displayName}</strong><p>{item.description ?? '등록된 설명이 없습니다.'}</p></div><i>{skillKey === item.skillKey ? '✓' : ''}</i></button>)}</div>{!catalog.length && <p className="table-empty">등록된 동작이 없습니다.</p>}<div className="wizard-actions"><span /><button className="primary-button" disabled={!skill} onClick={() => updateLearningDraft({ step: 2, method: 'existing' })}>다음: 수집 데이터 선택 →</button></div></section>}

    {step === 2 && <section className="card learning-wizard-card"><div className="wizard-title"><span>{2 - stepOffset}</span><div><h2>학습에 사용할 수집 데이터를 선택해 주세요</h2></div></div><div className="episode-select-list"><div className="episode-selection-toolbar"><span>학습 가능 {readyData.length}개</span><button className="text-button" type="button" disabled={!readyData.length} onClick={() => updateLearningDraft({ selected: allSelected ? [] : readyData.map((item) => item.id) })}>{allSelected ? '전체 선택 해제' : '전체 선택'}</button></div>{availableData.map((item) => <label key={item.id} className={`${isSelectableDataset(item) && selected.includes(item.id) ? 'selected' : ''}${!isSelectableDataset(item) ? ' disabled' : ''}`}><input type="checkbox" disabled={!isSelectableDataset(item)} checked={isSelectableDataset(item) && selected.includes(item.id)} onChange={() => toggleEpisode(item.id)} /><DatasetWhen item={item} /><span className="dataset-handle">{item.umiId}</span><span className="dataset-length">{item.duration}</span><em>{isSelectableDataset(item) ? '확인 완료 · 학습 가능' : recoveryActionLabel[item.recoveryAction]}</em></label>)}{!availableData.length && <p className="table-empty">이 동작에 수집된 데이터가 없습니다.</p>}<div className="episode-selection-summary" role="status"><strong>{selectedData.length}개 선택됨</strong><span>{selectedData.length ? '선택한 데이터로 학습을 생성할 수 있어요.' : '학습할 데이터를 선택해 주세요.'}</span></div></div><div className="wizard-actions">{preset ? <span /> : <button className="secondary-button" onClick={() => updateLearningDraft({ step: 1, selected: [] })}>← 이전</button>}<button className="primary-button" disabled={!selectedData.length} onClick={() => updateLearningDraft({ step: 3 })}>다음: 최종 확인 →</button></div></section>}

    {step === 3 && skill && <section className="card learning-wizard-card review-card"><div className="wizard-title"><span>{3 - stepOffset}</span><div><h2>학습 내용을 확인해 주세요</h2></div></div><dl><div><dt>학습 동작</dt><dd>{skill.displayName}</dd></div><div><dt>선택 데이터</dt><dd>{selectedData.length}개</dd></div><div><dt>수집 핸들</dt><dd>{selectedData[0]?.umiId ?? '확인할 수 없음'}</dd></div></dl><div className="review-notice"><strong>생성하면 학습 목록에서 진행 상태를 확인할 수 있습니다.</strong></div><div className="wizard-actions"><button className="secondary-button" onClick={() => updateLearningDraft({ step: 2 })}>← 이전</button><button className="primary-button" disabled={submitting || !selectedData.length} onClick={() => void submitTraining()}>{submitting ? '생성 중...' : '학습 작업 생성'}</button></div></section>}
  </>;
}
