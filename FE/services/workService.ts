import type { LearnedSkill, SavedWork, WorkflowBlock } from '@/types';
import { ApiError, apiRequest } from './client';

interface ApiSkill { id: string; displayName: string; description: string | null; }
interface ApiSkillVersion { id: string; skillId: string; version: number; }
interface ApiTaskStep {
  order: number;
  type: 'SKILL' | 'MOTOR' | 'REPEAT';
  skillVersionId: string | null;
  parameters: Record<string, unknown>;
  timeoutSeconds: number | null;
  repeatCount: number | null;
  children: ApiTaskStep[];
}
interface ApiTaskVersion { id: string; taskId: string; version: number; steps: ApiTaskStep[]; createdAt: string; }
interface ApiTask {
  id: string;
  taskKey: string;
  displayName: string;
  description: string | null;
  latestVersion: number | null;
  latestVersionDetails: ApiTaskVersion | null;
  updatedAt: string;
}

export async function fetchLearnedSkills(): Promise<LearnedSkill[]> {
  const skills = await apiRequest<ApiSkill[]>('/skills');
  const resolved = await Promise.all(skills.map(async (skill): Promise<LearnedSkill | null> => {
    try {
      const version = await apiRequest<ApiSkillVersion>(`/skills/${skill.id}/versions/latest`);
      return {
        id: version.id,
        skillId: skill.id,
        name: skill.displayName,
        kind: 'umi' as const,
        description: skill.description ?? '',
        mark: 'S',
      };
    } catch (cause) {
      if (cause instanceof ApiError && cause.status === 404) return null;
      throw cause;
    }
  }));
  return resolved.filter((skill): skill is LearnedSkill => skill !== null);
}

export async function fetchSavedWorks(): Promise<SavedWork[]> {
  const [tasks, skills] = await Promise.all([
    apiRequest<ApiTask[]>('/tasks?includeLatestVersion=true'),
    fetchLearnedSkills(),
  ]);
  const skillByVersionId = new Map(skills.map((skill) => [skill.id, skill]));
  return tasks.map((task) => toSavedWork(task, skillByVersionId));
}

/** 백엔드 DELETE 계약이 추가되면 선택 삭제에 사용한다. */
export async function deleteWorks(taskIds: string[]): Promise<SavedWork[]> {
  await Promise.all(taskIds.map((id) => apiRequest<void>(`/tasks/${id}`, { method: 'DELETE' })));
  return fetchSavedWorks();
}

export interface SaveWorkResult { works: SavedWork[]; savedId: string; }

export async function saveWork(work: SavedWork): Promise<SaveWorkResult> {
  if (!work.blocks.length) {
    throw new Error('작업에 학습된 동작을 하나 이상 추가해 주세요.');
  }
  const steps = toApiSteps(work.blocks);

  /* 저장된 작업(draft- 가 아닌 서버 id)은 같은 작업에 새 버전을 올린다 — PATCH에 steps를 주면 서버가 새 버전을 발행한다. */
  if (!work.id.startsWith('draft-')) {
    await apiRequest<ApiTask>(`/tasks/${work.id}`, { method: 'PATCH', body: JSON.stringify({ steps }) });
    return { works: await fetchSavedWorks(), savedId: work.id };
  }

  const created = await apiRequest<ApiTask>('/tasks', {
    method: 'POST',
    body: JSON.stringify({
      taskKey: `web_${crypto.randomUUID()}`,
      displayName: work.name,
      description: null,
      steps,
    }),
  });
  const taskId = created.id;
  return { works: await fetchSavedWorks(), savedId: taskId };
}

function toApiSteps(blocks: WorkflowBlock[]): ApiTaskStep[] {
  return blocks.map((block, index) => ({
    order: index + 1,
    type: block.kind === 'repeat' ? 'REPEAT' : block.kind === 'motor' ? 'MOTOR' : 'SKILL',
    skillVersionId: block.skillId ?? null,
    parameters: block.kind === 'motor'
      ? { motor: block.motor, angle: block.angle }
      : {},
    timeoutSeconds: block.kind === 'repeat' ? null : block.kind === 'motor' ? 10 : 30,
    repeatCount: block.kind === 'repeat' ? (block.repeat ?? 1) : null,
    children: block.kind === 'repeat' ? toApiSteps(block.children ?? []) : [],
  }));
}

function toSavedWork(task: ApiTask, skills: Map<string, LearnedSkill>): SavedWork {
  const version = task.latestVersionDetails;
  let blockId = 1;
  const toBlocks = (steps: ApiTaskStep[]): WorkflowBlock[] => steps.map((step) => {
    const skill = step.skillVersionId ? skills.get(step.skillVersionId) : undefined;
    const id = blockId++;
    if (step.type === 'MOTOR') return {
      id,
      kind: 'motor' as const,
      title: '모터 제어',
      motor: typeof step.parameters.motor === 'string' ? step.parameters.motor : 'J1',
      angle: typeof step.parameters.angle === 'number' ? step.parameters.angle : 0,
    };
    if (step.type === 'REPEAT') return {
      id,
      kind: 'repeat' as const,
      title: '반복 수행',
      repeat: step.repeatCount ?? 1,
      children: toBlocks(step.children ?? []),
    };
    return {
      id,
      kind: 'umi' as const,
      title: skill?.name ?? '사용할 수 없는 학습 동작',
      skillId: step.skillVersionId ?? undefined,
    };
  });
  const blocks = toBlocks(version?.steps ?? []);
  return {
    id: task.id,
    taskKey: task.taskKey,
    taskVersionId: version?.id ?? null,
    version: version?.version ?? task.latestVersion,
    name: task.displayName,
    robotId: '',
    blocks,
    estimatedSeconds: blocks.length * 25,
    updatedAt: task.updatedAt,
  };
}
