'use client';

import { useLayoutEffect, useMemo, useState } from 'react';
import type { BlockKind, LearnedSkill, SavedWork, View, WorkflowBlock } from '@/types';
import { blockCatalog, motorRanges } from '@/lib/blocks';
import { useTaskDeployment } from '@/hooks/useTaskDeployment';
import { useAppStore } from '@/hooks/useAppStore';
import { makeBlock, makeSkillBlock, flattenWorkflowBlocks, estimateWorkflowSeconds, formatWorkDuration, getWorkflowIssues, workflowContentEquals, syncNextBlockId } from '@/lib/workflow';
export function WorkView({ editingWorkId, suggestedSkillName, onNavigate, onToast }: { editingWorkId: string | null; suggestedSkillName: string | null; onNavigate: (view: View) => void; onToast: (message: string, tone?: 'success' | 'error') => void }) {
  const { robots, skills, savedWorks, saveWorkflow } = useAppStore();
  const editingWork = savedWorks.find((work) => work.id === editingWorkId);
  const [robotId, setRobotId] = useState(editingWork?.robotId || robots[0]?.id || '');
  const [blocks, setBlocks] = useState<WorkflowBlock[]>(() => structuredClone(
    editingWork?.blocks ?? (suggestedSkillName && skills.find(skill => skill.name === suggestedSkillName)
      ? [makeSkillBlock(skills.find(skill => skill.name === suggestedSkillName)!)]
      : skills.slice(0, 1).map(makeSkillBlock)),
  ));
  const [runningIndex, setRunningIndex] = useState(-1);
  const [workName, setWorkName] = useState(editingWork?.name ?? '');
  const [selectedWorkId, setSelectedWorkId] = useState(editingWork?.id ?? '');
  const [nameError, setNameError] = useState('');
  const [saving, setSaving] = useState(false);
  const [workStep, setWorkStep] = useState<1 | 2 | 3>(1);
  /* 단계 전환은 화면 교체라 새 단계를 그리기 직전(페인트 전)에 즉시 맨 위로 올린다.
     클릭 시점에 smooth로 올리면 긴 1단계에서 애니메이션이 시작된 뒤 짧은 2단계로 바뀌며 스크롤이 잘려 위로 튕겨 보였다. */
  useLayoutEffect(() => { window.scrollTo({ top: 0, behavior: 'instant' }); }, [workStep]);

  const selectedRobot = robots.find((robot) => robot.id === robotId);
  const savedSnapshot = savedWorks.find(work => work.id === selectedWorkId);
  const workForValidation = useMemo<SavedWork>(() => ({
    id: selectedWorkId || 'draft',
    taskKey: savedSnapshot?.taskKey,
    taskVersionId: savedSnapshot?.taskVersionId,
    version: savedSnapshot?.version,
    name: workName,
    robotId,
    blocks,
    estimatedSeconds: estimateWorkflowSeconds(blocks),
    updatedAt: savedSnapshot?.updatedAt ?? '',
  }), [blocks, robotId, savedSnapshot, selectedWorkId, workName]);
  const unavailableSkills = flattenWorkflowBlocks(blocks).filter((block) => block.skillId && !skills.some((skill) => skill.id === block.skillId));
  const workflowIssues = [...getWorkflowIssues(workForValidation), ...unavailableSkills.map((block) => `‘${block.title}’ 학습 동작을 사용할 수 없어요.`)];
  const robotReady = selectedRobot?.status === '온라인';
  const isSaved = !!savedSnapshot
    && savedSnapshot.name === workName.trim()
    && workflowContentEquals(savedSnapshot.blocks, blocks);
  const deployment = useTaskDeployment(workForValidation, robotId);
  const executionStatus = deployment.execution?.status;
  const executionFailed = executionStatus === 'FAILED';
  const executionSucceeded = executionStatus === 'SUCCEEDED';
  const executionCanceled = executionStatus === 'CANCELED';
  const executionTerminal = executionFailed || executionSucceeded || executionCanceled;
  const running = runningIndex >= 0 && !executionTerminal;
  const canStartRun = isSaved && robotReady && !workflowIssues.length && deployment.ready;

  const save = async () => {
    if (!workName.trim()) { setNameError('작업 이름을 입력해 주세요.'); return false; }
    const normalizedName = workName.trim().toLocaleLowerCase('ko-KR');
    const sameNameWork = savedWorks.find(
      (work) => work.name.trim().toLocaleLowerCase('ko-KR') === normalizedName,
    );
    if (sameNameWork && sameNameWork.id !== selectedWorkId) {
      setNameError('같은 이름의 작업이 이미 있어요. 이름을 바꿔 주세요.');
      window.scrollTo({ top: 0, behavior: 'smooth' });
      onToast('작업이 이미 존재합니다. 이름을 변경하세요', 'error');
      return false;
    }
    /* 이름이 지금 작업 자신의 이름이면 충돌이 아니라 수정이다 — 내용이 같으면 그대로, 다르면 같은 작업의 새 버전으로 저장한다. */
    const updatingOwn = !!sameNameWork && sameNameWork.id === selectedWorkId;
    if (updatingOwn && workflowContentEquals(sameNameWork.blocks, blocks)) return true;
    if (workflowIssues.length) { setNameError(workflowIssues[0]); return false; }
    setNameError('');
    setSaving(true);
    try {
      const saved = await saveWorkflow({ id: updatingOwn ? selectedWorkId : `draft-${crypto.randomUUID()}`, name: workName.trim(), robotId, blocks, estimatedSeconds: estimateWorkflowSeconds(blocks), updatedAt: new Date().toISOString() });
      setSelectedWorkId(saved.id);
      onToast(`‘${workName.trim()}’ 작업을 저장했어요.`);
      return true;
    } catch (cause) {
      setNameError(cause instanceof Error ? cause.message : '저장하지 못했어요. 다시 시도해 주세요.');
      return false;
    }
    finally { setSaving(false); }
  };

  const loadWork = (workId: string) => {
    const work = savedWorks.find((item) => item.id === workId);
    if (!work) return;
    const loaded = structuredClone(work.blocks);
    /* 불러온 블록 ID와 새 블록 ID가 겹치지 않게 카운터를 끌어올린다. */
    const walk = (items: WorkflowBlock[]) => items.forEach((block) => { syncNextBlockId(block.id + 1); if (block.children) walk(block.children); });
    walk(loaded);
    setBlocks(loaded);
    setWorkName(work.name);
    if (robots.some((robot) => robot.id === work.robotId)) setRobotId(work.robotId);
    setSelectedWorkId(workId);
    setNameError('');
    onToast(`‘${work.name}’ 작업을 불러왔어요.`);
  };

  const startRun = async () => {
    if (!workName.trim()) { setNameError('작업 이름을 입력해 주세요.'); return; }
    if (!canStartRun) return;
    setNameError('');
    try {
      await deployment.execute();
      setRunningIndex(0);
      onToast('로봇에 작업 실행을 요청했어요.');
    } catch (cause) {
      setNameError(cause instanceof Error ? cause.message : '작업을 시작하지 못했어요.');
    }
  };

  const addBlock = (kind: BlockKind, repeatId?: number) => {
    const block = makeBlock(kind);
    if (repeatId) {
      if (kind === 'repeat') return;
      setBlocks((items) => items.map((item) => item.id === repeatId ? { ...item, children: [...(item.children ?? []), block] } : item));
    } else setBlocks((items) => [...items, block]);
  };

  const addSkillBlock = (skill: LearnedSkill, repeatId?: number) => {
    const block = makeSkillBlock(skill);
    if (repeatId) {
      setBlocks((items) => items.map((item) => item.id === repeatId ? { ...item, children: [...(item.children ?? []), block] } : item));
    } else setBlocks((items) => [...items, block]);
  };

  const updateBlock = (id: number, patch: Partial<WorkflowBlock>) => {
    const updateItems = (items: WorkflowBlock[]): WorkflowBlock[] => items.map((item) => {
      if (item.id === id) return { ...item, ...patch };
      if (item.children?.length) return { ...item, children: updateItems(item.children) };
      return item;
    });
    setBlocks(updateItems);
  };
  const removeBlock = (id: number) => setBlocks((items) => items.filter((item) => item.id !== id));
  const moveBlock = (index: number, direction: -1 | 1) => {
    const target = index + direction;
    if (target < 0 || target >= blocks.length) return;
    setBlocks((items) => { const next = [...items]; [next[index], next[target]] = [next[target], next[index]]; return next; });
  };
  const dropKind = (event: React.DragEvent, repeatId?: number) => {
    event.preventDefault(); event.stopPropagation();
    const skillId = event.dataTransfer.getData('skill-id');
    const skill = skills.find((item) => item.id === skillId);
    if (skill) { addSkillBlock(skill, repeatId); return; }
    const kind = event.dataTransfer.getData('block-kind') as BlockKind;
    if (kind) addBlock(kind, repeatId);
  };
  const removeChildBlock = (repeatId: number, childId: number) => setBlocks((items) => items.map((item) => item.id === repeatId ? { ...item, children: (item.children ?? []).filter((child) => child.id !== childId) } : item));
  const moveChildBlock = (repeatId: number, childIndex: number, direction: -1 | 1) => setBlocks((items) => items.map((item) => {
    if (item.id !== repeatId) return item;
    const children = [...(item.children ?? [])];
    const target = childIndex + direction;
    if (target < 0 || target >= children.length) return item;
    [children[childIndex], children[target]] = [children[target], children[childIndex]];
    return { ...item, children };
  }));

  if (!selectedRobot) {
    return <><div className="page-heading work-heading title-divider"><div><h1>작업 생성</h1></div></div><div className="center-panel"><section className="card state-card"><span className="state-icon">R</span><h2>사용할 로봇이 없어요</h2><p>작업을 만들기 전에 로봇을 등록해 주세요.</p><button className="primary-button" onClick={() => onNavigate('devices')}>장치 관리로 이동</button></section></div></>;
  }

  return (
    <>
      <div className="page-heading work-heading title-divider"><div><h1>작업 생성</h1></div></div>
      <ol className="work-stage-steps" aria-label="작업 생성 단계">
        {['작업 구성', '실행 준비', '실행'].map((label, index) => <li key={label} className={workStep === index + 1 ? 'current' : workStep > index + 1 ? 'done' : ''}><span>{workStep > index + 1 ? '✓' : index + 1}</span><strong>{label}</strong></li>)}
      </ol>

      {workStep === 1 && <>
        <div className="work-namebar"><label>작업 이름<input value={workName} disabled={running} onChange={(e) => { setWorkName(e.target.value); if (nameError && e.target.value.trim()) setNameError(''); }} />{nameError && <small className="field-error" role="alert">{nameError}</small>}</label><label>저장된 작업<select disabled={running} value={selectedWorkId} onChange={(e) => loadWork(e.target.value)}><option value="" disabled>작업 불러오기</option>{savedWorks.map((work) => <option key={work.id} value={work.id}>{work.name}</option>)}</select></label><button className="secondary-button" disabled={running || saving} onClick={() => void save()}>{saving ? '저장 중...' : '저장'}</button></div>
        <div className="block-workspace work-compose-layout">
        <aside className="block-library">
          <div className="library-heading"><h2>작업 블록</h2></div>
          <p className="block-category">학습된 동작 <span>{skills.length}</span></p>
          {skills.length ? skills.map((skill) => <button key={skill.id} className={`library-block ${skill.kind}`} draggable={!running} onDragStart={(e) => e.dataTransfer.setData('skill-id', skill.id)} onClick={() => !running && addSkillBlock(skill)}><span>{skill.mark}</span><div><strong>{skill.name}</strong></div><em>＋</em></button>) : <div className="block-library-empty"><p>사용할 수 있는 학습 동작이 없어요.</p><button className="text-button" onClick={() => onNavigate('learn')}>학습 생성하기 →</button></div>}
          <details className="advanced-blocks">
            <summary>고급 블록</summary>
            <p className="block-category">직접 제어</p>
            {blockCatalog.filter((item) => item.kind === 'motor').map((item) => <button key={item.kind} className={`library-block ${item.kind}`} draggable={!running} onDragStart={(e) => e.dataTransfer.setData('block-kind', item.kind)} onClick={() => !running && addBlock(item.kind)}><span>{item.mark}</span><div><strong>{item.title}</strong></div><em>＋</em></button>)}
            <p className="block-category">흐름 제어</p>
            {blockCatalog.filter((item) => item.kind === 'repeat').map((item) => <button key={item.kind} className={`library-block ${item.kind}`} draggable={!running} onDragStart={(e) => e.dataTransfer.setData('block-kind', item.kind)} onClick={() => !running && addBlock(item.kind)}><span>{item.mark}</span><div><strong>{item.title}</strong></div><em>＋</em></button>)}
          </details>
        </aside>

        <section className="canvas" onDragOver={(e) => e.preventDefault()} onDrop={(e) => dropKind(e)}>
          <div className="canvas-head"><div><h2>작업 흐름</h2></div><span>{blocks.length}개 블록</span></div>
          <div className="flow"><div className="start-block"><span>▶</span><div><strong>작업 시작</strong></div></div><div className="connector" />
            {blocks.map((block, index) => <div className="flow-unit" key={block.id}><WorkflowBlockCard block={block} active={runningIndex === index} disabled={running} onUpdate={updateBlock} onRemove={removeBlock} onMove={(direction) => moveBlock(index, direction)} onDropChild={(event) => dropKind(event, block.id)} onRemoveChild={(childId) => removeChildBlock(block.id, childId)} onMoveChild={(childIndex, direction) => moveChildBlock(block.id, childIndex, direction)} /><div className="connector" /></div>)}
            <div className="drop-zone"><span>＋</span>이곳에 작업 블록을 놓아주세요</div>
          </div>
        </section>
        </div>
        <div className="work-stage-actions"><button className="text-button" onClick={() => onNavigate('library')}>← 작업 목록</button><button className="primary-button" disabled={saving || !workName.trim() || !!workflowIssues.length} onClick={async () => { if (isSaved || await save()) setWorkStep(2); }}>{saving ? '저장 중...' : '저장 후 실행 준비 →'}</button></div>
      </>}

      {workStep === 2 && <section className="card work-stage-card">
        <div className="work-stage-heading"><div><p className="eyebrow">실행 준비</p><h2>{workName || '이름 없는 작업'}</h2><p>실행할 로봇과 작업 상태를 확인한 뒤 적용하세요.</p></div><span>{blocks.length}개 블록 · {formatWorkDuration(estimateWorkflowSeconds(blocks))}</span></div>
        <div className="work-tools stage-robot-select"><label>실행할 로봇<select value={robotId} disabled={running} onChange={(e) => setRobotId(e.target.value)}>{robots.slice(0, 3).map((robot) => <option key={robot.id} value={robot.id}>{robot.name} · {robot.status}</option>)}</select></label><span className={`robot-status status-${selectedRobot.status.replace(' ', '-')}`}><i />{selectedRobot.status}</span></div>
        <div className="preflight-grid stage-preflight"><div className={robotReady ? 'check-pass' : 'check-fail'}><span>{robotReady ? '✓' : '!'}</span><div><strong>로봇 상태</strong><small>{selectedRobot.name} · {selectedRobot.status}</small></div></div><div className={!workflowIssues.length ? 'check-pass' : 'check-fail'}><span>{!workflowIssues.length ? '✓' : '!'}</span><div><strong>작업 유효성</strong><small>{workflowIssues[0] ?? `${flattenWorkflowBlocks(blocks).length}개 블록을 실행할 수 있어요.`}</small></div></div><div className={isSaved ? 'check-pass' : 'check-fail'}><span>{isSaved ? '✓' : '!'}</span><div><strong>저장 상태</strong><small>{isSaved ? '최신 작업이 저장되어 있어요.' : '작업을 다시 저장해 주세요.'}</small></div></div></div>
        <div className={`deployment-status${deployment.error ? ' failed' : ''}`}><strong>{deployment.ready ? `작업 v${deployment.version} · 실행 가능` : deployment.pending ? '최신화 중' : deployment.error ? '최신화 실패' : '최신화 필요'}</strong><p>{deployment.ready ? `${selectedRobot.name}에 현재 작업 버전이 준비됐어요.` : deployment.pending ? '로봇이 작업을 다운로드하고 있어요.' : '현재 작업 구성을 확인하고 선택한 로봇에 적용하세요.'}</p>{deployment.error && <p className="deployment-error" role="alert"><span aria-hidden="true">!</span>{deployment.error}</p>}{!deployment.ready && <button className="secondary-button" disabled={!isSaved || !robotReady || !!workflowIssues.length || deployment.pending} onClick={() => void deployment.deploy()}>{deployment.pending ? '적용 중...' : deployment.error ? '다시 적용' : '로봇에 적용'}</button>}</div>
        <div className="safety-note"><span>!</span><p><strong>실행 전 확인해 주세요</strong>로봇 주변에 사람이나 장애물이 없는지 확인하세요.</p></div>
        <div className="work-stage-actions inside"><button className="secondary-button" onClick={() => setWorkStep(1)}>← 작업 구성</button><button className="primary-button" disabled={!deployment.ready} onClick={() => setWorkStep(3)}>실행 단계로 →</button></div>
      </section>}

      {workStep === 3 && <section className="card work-stage-card execution-stage">
        <div className="work-stage-heading"><div><p className="eyebrow">작업 실행</p><h2>{workName}</h2><p>{selectedRobot.name}에서 작업을 실행합니다.</p></div><span>{formatWorkDuration(estimateWorkflowSeconds(blocks))}</span></div>
        <div className="run-robot"><span className="robot-illustration">R</span><div><strong>{selectedRobot.name}</strong><small>{selectedRobot.location}</small></div><span className={`status-dot ${robotReady ? 'online' : 'busy'}`} /></div>
        {running ? <div className="execution-focus"><strong>{deployment.execution?.currentStep ?? 0} / {blocks.length}</strong><p>로봇이 작업 실행 요청을 처리하고 있어요.</p><div className="mini-progress"><i style={{ width: `${deployment.execution?.progressPercent ?? 0}%` }} /></div><button className="danger-button" onClick={() => void deployment.cancel().then(() => { setRunningIndex(-1); onToast('작업 중지를 요청했어요.'); }).catch((cause) => onToast(cause instanceof Error ? cause.message : '작업을 중지하지 못했어요.', 'error'))}>■ 작업 중지</button></div>
          : executionFailed ? <div className="execution-focus failed" role="alert"><strong>작업 실행 실패</strong><p>오류가 발견되어 작업을 중지합니다</p><button className="primary-button run-button" disabled={!canStartRun} onClick={() => void startRun()}>▶ 다시 실행</button></div>
          : executionSucceeded ? <div className="execution-focus"><strong>작업 실행 완료</strong><p>로봇이 작업을 정상적으로 완료했어요.</p><button className="primary-button run-button" disabled={!canStartRun} onClick={() => void startRun()}>▶ 다시 실행</button></div>
          : executionCanceled ? <div className="execution-focus"><strong>작업 실행 취소됨</strong><p>로봇의 작업 실행이 취소되었어요.</p><button className="primary-button run-button" disabled={!canStartRun} onClick={() => void startRun()}>▶ 다시 실행</button></div>
          : <div className="execution-focus"><strong>실행 준비 완료</strong><p>작업 v{deployment.version}이 로봇에 적용됐어요. 주변 안전을 확인하고 시작하세요.</p><button className="primary-button run-button" disabled={!canStartRun} onClick={() => void startRun()}>▶ 작업 시작</button></div>}
        <div className="work-stage-actions inside"><button className="secondary-button" disabled={running} onClick={() => setWorkStep(2)}>← 실행 준비</button><button className="text-button" disabled={running} onClick={() => onNavigate('library')}>작업 목록으로</button></div>
      </section>}
    </>
  );
}

function MotorSettings({ block, disabled, nested = false, onUpdate }: { block: WorkflowBlock; disabled: boolean; nested?: boolean; onUpdate: (id: number, patch: Partial<WorkflowBlock>) => void }) {
  const range = motorRanges[block.motor ?? 'J1'];
  return <div className={`motor-config${nested ? ' nested-motor-config' : ''}`}>
    <label>모터<select value={block.motor} disabled={disabled} onChange={(e) => onUpdate(block.id, { motor: e.target.value, angle: 0 })}>{Object.keys(motorRanges).map((motor) => <option key={motor}>{motor}</option>)}</select></label>
    <label className="angle-field">회전 각도 <span>{block.angle}°</span><input aria-label={`${block.title} 회전 각도`} type="range" min={range[0]} max={range[1]} value={block.angle} disabled={disabled} onChange={(e) => onUpdate(block.id, { angle: Number(e.target.value) })} /><small>{range[0]}° <b>{range[1]}°</b></small></label>
  </div>;
}

function NestedWorkflowBlockCard({ block, index, count, disabled, onUpdate, onRemove, onMove }: { block: WorkflowBlock; index: number; count: number; disabled: boolean; onUpdate: (id: number, patch: Partial<WorkflowBlock>) => void; onRemove: (id: number) => void; onMove: (index: number, direction: -1 | 1) => void }) {
  const catalog = blockCatalog.find((item) => item.kind === block.kind)!;
  return <article className={`nested-workflow-block ${block.kind}`}>
    <div className="nested-block-head">
      <span className="block-mark">{catalog.mark}</span>
      <div><strong>{block.title}</strong></div>
      <div className="block-actions">
        <button disabled={disabled || index === 0} onClick={() => onMove(index, -1)} aria-label={`${block.title} 위로 이동`}>↑</button>
        <button disabled={disabled || index === count - 1} onClick={() => onMove(index, 1)} aria-label={`${block.title} 아래로 이동`}>↓</button>
        <button disabled={disabled} onClick={() => onRemove(block.id)} aria-label={`${block.title} 삭제`}>×</button>
      </div>
    </div>
    {block.kind === 'motor' && <MotorSettings block={block} disabled={disabled} nested onUpdate={onUpdate} />}
  </article>;
}

function WorkflowBlockCard({ block, active, disabled, onUpdate, onRemove, onMove, onDropChild, onRemoveChild, onMoveChild }: { block: WorkflowBlock; active: boolean; disabled: boolean; onUpdate: (id: number, patch: Partial<WorkflowBlock>) => void; onRemove: (id: number) => void; onMove: (direction: -1 | 1) => void; onDropChild: (event: React.DragEvent) => void; onRemoveChild: (id: number) => void; onMoveChild: (index: number, direction: -1 | 1) => void }) {
  const catalog = blockCatalog.find((item) => item.kind === block.kind)!;
  return <div className={`workflow-block ${block.kind} ${active ? 'executing' : ''}`}><div className="block-top"><span className="block-mark">{catalog.mark}</span><div><strong>{block.title}</strong></div>{active && <em className="executing-badge">실행 중</em>}<div className="block-actions"><button disabled={disabled} onClick={() => onMove(-1)} aria-label="위로 이동">↑</button><button disabled={disabled} onClick={() => onMove(1)} aria-label="아래로 이동">↓</button><button disabled={disabled} onClick={() => onRemove(block.id)} aria-label="삭제">×</button></div></div>
    {block.kind === 'motor' && <MotorSettings block={block} disabled={disabled} onUpdate={onUpdate} />}
    {block.kind === 'repeat' && <div className="repeat-config">
      <div className="repeat-toolbar"><label>반복 횟수<input aria-label="반복 횟수" type="number" min="1" max="20" value={block.repeat} disabled={disabled} onChange={(e) => onUpdate(block.id, { repeat: Math.max(1, Math.min(20, Number(e.target.value))) })} /></label><div><strong>반복 동작</strong></div><span>{block.children?.length ?? 0}개</span></div>
      <div className="repeat-drop" onDragOver={(e) => e.preventDefault()} onDrop={onDropChild}>
        {block.children?.length ? block.children.map((child, index) => <NestedWorkflowBlockCard key={child.id} block={child} index={index} count={block.children?.length ?? 0} disabled={disabled} onUpdate={onUpdate} onRemove={onRemoveChild} onMove={onMoveChild} />) : <p>반복할 블록을 이 안에 놓아주세요</p>}
        <div className="nested-drop-hint"><span>＋</span>이곳에 모터 또는 학습 동작 블록 놓기</div>
      </div>
    </div>}
  </div>;
}
