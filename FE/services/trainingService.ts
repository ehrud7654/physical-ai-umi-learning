import type { SkillOverview, Training, TrainingSkill, TrainingStatus } from '@/types';
import { apiRequest } from './client';

type ApiTrainingStatus =
  | 'CREATED' | 'COLLECTING' | 'INPUT_VALIDATING' | 'INPUT_READY' | 'QUEUED'
  | 'PREPARING' | 'RUNNING' | 'MODEL_UPLOADING' | 'MODEL_VALIDATING'
  | 'CANCEL_REQUESTED' | 'CANCELED' | 'SUCCEEDED' | 'FAILED';

interface ApiTrainingJob {
  id: string;
  skillKey: string;
  skillDisplayName: string;
  umiDeviceId: string;
  inputMode: 'COLLECT_NEW' | 'EXISTING_EPISODES';
  status: ApiTrainingStatus;
  progressStage: string | null;
  progressPercent: number;
  expectedEpisodeCount: number;
  receivedEpisodeCount: number;
  receivedBytes: number;
  episodeCount: number;
  resultSkillVersionId: string | null;
  error: { code: string; message: string; recoveryAction: string } | null;
  createdAt: string;
  startedAt: string | null;
  finishedAt: string | null;
  updatedAt: string;
}

interface ApiDevice {
  id: string;
  handleNumber: string;
  name: string;
}

interface ApiEpisodeInput {
  episodeId: string;
  inputOrder: number;
}

interface ApiSkill {
  id: string;
  skillKey: string;
  tier: 'TUTORIAL' | 'DEMO' | 'CUSTOM';
  displayName: string;
  description: string | null;
  status: string;
}

export async function fetchTrainingSkills(): Promise<TrainingSkill[]> {
  return apiRequest<ApiSkill[]>('/skills');
}

export async function fetchSkillOverviews(): Promise<SkillOverview[]> {
  return apiRequest<SkillOverview[]>('/skills/overview');
}

export async function createCustomSkill(input: {
  displayName: string;
  description?: string;
}): Promise<TrainingSkill> {
  return apiRequest<ApiSkill>('/skills', {
    method: 'POST',
    headers: { 'Idempotency-Key': crypto.randomUUID() },
    body: JSON.stringify({
      displayName: input.displayName.trim(),
      description: input.description?.trim() || null,
    }),
  });
}

export async function updateSkillDescription(
  skillId: string,
  description: string,
): Promise<TrainingSkill> {
  return apiRequest<ApiSkill>(`/skills/${skillId}`, {
    method: 'PATCH',
    body: JSON.stringify({ description: description.trim() || null }),
  });
}

type CreateTrainingJobInput = {
  umiDeviceId: string;
  episodeIds: string[];
} & (
  | { skillKey: string; newSkill?: never }
  | { skillKey?: never; newSkill: { displayName: string; description?: string } }
);

export async function createTrainingJob(input: CreateTrainingJobInput): Promise<Training> {
  const job = await apiRequest<ApiTrainingJob>('/training-jobs', {
    method: 'POST',
    headers: { 'Idempotency-Key': crypto.randomUUID() },
    body: JSON.stringify(input),
  });
  return toTraining(job);
}

export async function fetchTrainings(status?: TrainingStatus | 'all'): Promise<Training[]> {
  const [jobs, devices] = await Promise.all([
    apiRequest<ApiTrainingJob[]>('/training-jobs'),
    apiRequest<ApiDevice[]>('/devices?deviceType=UMI_CAMERA'),
  ]);
  const deviceNames = new Map(devices.map((device) => [device.id, device.name || device.handleNumber]));
  const trainings = jobs.map((job) => toTraining(job, deviceNames.get(job.umiDeviceId)));
  return !status || status === 'all' ? trainings : trainings.filter((item) => item.status === status);
}

export async function fetchTrainingDetail(trainingJobId: string): Promise<Training> {
  const job = await apiRequest<ApiTrainingJob>(`/training-jobs/${trainingJobId}`);
  const device = await apiRequest<ApiDevice>(`/devices/${job.umiDeviceId}`);
  return toTraining(job, device.name || device.handleNumber);
}

export async function cancelTraining(trainingJobId: string): Promise<Training[]> {
  await apiRequest<ApiTrainingJob>(`/training-jobs/${trainingJobId}/cancel`, { method: 'POST' });
  return fetchTrainings();
}

/** 백엔드 DELETE 계약이 추가되면 선택 삭제에 사용한다. */
export async function deleteTrainings(trainingJobIds: string[]): Promise<Training[]> {
  await Promise.all(trainingJobIds.map((id) => apiRequest<void>(`/training-jobs/${id}`, { method: 'DELETE' })));
  return fetchTrainings();
}

/** 별도 retry API 대신 기존 Episode를 입력으로 새 TrainingJob을 생성한다. */
export async function retryTraining(training: Training): Promise<Training[]> {
  const episodes = await apiRequest<ApiEpisodeInput[]>(`/training-jobs/${training.id}/episodes`);
  if (episodes.length === 0) {
    throw new Error('다시 학습할 Episode가 없습니다.');
  }
  const trainingInputs = episodes.sort((a, b) => a.inputOrder - b.inputOrder);
  await apiRequest<ApiTrainingJob>('/training-jobs', {
    method: 'POST',
    headers: { 'Idempotency-Key': crypto.randomUUID() },
    body: JSON.stringify({
      skillKey: training.skillKey,
      umiDeviceId: training.umiDeviceId,
      episodeIds: trainingInputs.map((item) => item.episodeId),
    }),
  });
  return fetchTrainings();
}

function toTraining(job: ApiTrainingJob, deviceName?: string): Training {
  return {
    id: job.id,
    skillKey: job.skillKey,
    umiDeviceId: job.umiDeviceId,
    inputMode: job.inputMode,
    expectedEpisodeCount: job.expectedEpisodeCount,
    receivedEpisodeCount: job.receivedEpisodeCount,
    apiStatus: job.status,
    name: job.skillDisplayName,
    robot: deviceName ?? job.umiDeviceId,
    progress: Math.round(Number(job.progressPercent)),
    stage: trainingStatusLabel(job.status),
    status: statusFor(job.status),
    started: formatDateTime(job.startedAt ?? job.createdAt),
  };
}

function statusFor(status: ApiTrainingStatus): TrainingStatus {
  if (status === 'SUCCEEDED') return '완료';
  if (status === 'FAILED') return '실패';
  if (status === 'CANCEL_REQUESTED') return '취소 요청';
  if (status === 'CANCELED') return '취소됨';
  return '진행 중';
}

export function trainingStatusLabel(status: string): string {
  const labels: Record<string, string> = {
    CREATED: '학습 생성', COLLECTING: '데이터 수집', INPUT_VALIDATING: '입력 검증', INPUT_READY: '입력 준비 완료',
    QUEUED: '학습 대기', PREPARING: '학습 준비', RUNNING: '모델 학습', MODEL_UPLOADING: '모델 업로드',
    MODEL_VALIDATING: '모델 검증', CANCEL_REQUESTED: '취소 확인 중', CANCELED: '취소 완료', SUCCEEDED: '학습 완료', FAILED: '학습 실패',
  };
  return labels[status] ?? '상태 확인 중';
}

function formatDateTime(value: string): string {
  return new Intl.DateTimeFormat('ko-KR', {
    year: 'numeric', month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit',
  }).format(new Date(value));
}
