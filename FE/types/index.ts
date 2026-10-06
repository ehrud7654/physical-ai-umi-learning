export type View = 'dashboard' | 'skills' | 'library' | 'devices' | 'robot-detail' | 'training' | 'learn' | 'work' | 'admin';

export type BlockKind = 'umi' | 'vlm' | 'motor' | 'repeat';

export type RobotStatus = '온라인' | '오프라인' | '작업 중' | '오류' | '상태 확인 불가';

export type TrainingStatus = '진행 중' | '취소 요청' | '취소됨' | '완료' | '실패';

export type DeviceType = 'robot' | 'umi';

export interface AuthenticatedUser {
  id: string;
  email: string;
  name: string;
  createdAt: string;
}

export interface LoginResponse {
  accessToken: string;
  tokenType: 'Bearer';
  expiresIn: number;
  user: AuthenticatedUser;
  sessionExpiresAt: string;
  absoluteExpiresAt: string;
}

export type DashboardRobotStatus = 'AVAILABLE' | 'BUSY' | 'OFFLINE' | 'UNKNOWN';

export interface DashboardData {
  summary: {
    umi: { totalCount: number; onlineCount: number };
    robots: { totalCount: number; availableCount: number };
    trainings: { runningCount: number; averageProgressPercent: number };
    executions: {
      completedTodayCount: number;
      succeededTodayCount: number;
      failedTodayCount: number;
      successRatePercent: number;
    };
  };
  robots: Array<{
    id: string;
    name: string;
    status: DashboardRobotStatus;
    lastSeenAt: string | null;
    currentDeploymentId: string | null;
  }>;
  recentTrainings: Array<{
    id: string;
    skillId: string;
    skillName: string;
    umiDeviceId: string;
    umiDeviceName: string;
    status: string;
    stage: string | null;
    progressPercent: number;
    startedAt: string | null;
    updatedAt: string;
  }>;
  generatedAt: string;
}

export interface Robot {
  id: string;
  /** robots.id. 과거 등록 데이터처럼 Robot 행이 없는 경우에는 존재하지 않는다. */
  robotId?: string;
  jetsonDeviceId?: string;
  name: string;
  location: string;
  status: RobotStatus;
  /** 현재 배포된 Skill 모델 이름 */
  model: string;
}

export interface RobotDetail {
  id: string;
  jetsonDeviceId: string;
  name: string;
  status: RobotStatus;
  currentDeploymentId: string | null;
  currentTaskVersionId: string | null;
  lastSeenAt: string | null;
  createdAt: string;
}

export interface Umi {
  id: string;
  /** 서버 devices.id(UUID). 화면 표시용 핸들 번호인 id와 구분한다. */
  deviceId: string;
  place: string;
  status: '온라인' | '오프라인' | '상태 확인 불가';
  collected: string;
  data: number;
}

export type DatasetStatus = 'REGISTERED' | 'UPLOADING' | 'VALIDATING' | 'READY' | 'INVALID' | 'DELETING' | 'DELETED' | 'DELETE_FAILED';

/** EPISODE-01/02 응답을 데이터 탭 표시용으로 정리한 모델. id는 서버 Episode UUID다. */
export interface Dataset {
  id: string;
  fileName: string;
  /** 화면 표시용 순번(등록 오래된 순 1부터). 새로고침마다 다시 계산되므로 저장 키로 쓰지 않는다. */
  seq: number;
  /** 업로드 완료 시각(ISO). 목록 행의 시각·날짜 칸은 이걸로 렌더 시점에 계산한다(오늘/어제가 자정을 넘겨도 맞게). */
  uploadedAt: string | null;
  /** 전체 시각 `YYYY. MM. DD  HH:mm:ss` — 행의 툴팁용. */
  date: string;
  duration: string;
  skillKey: string;
  /** 서버 devices.id(UUID). 매핑·시연 Episode가 같은 UMI인지 판별할 때 사용한다. */
  umiDeviceId: string;
  umiId: string;
  status: DatasetStatus;
  retentionDays: number;
  retentionExpiresAt: string;
  trainable: boolean;
  blockingReasons: BlockingReason[];
  recoveryAction: RecoveryAction;
}

export type RecoveryAction = 'WAIT' | 'RETRY_UPLOAD' | 'RECORD_AGAIN' | 'SET_LABEL' | 'EXTEND_RETENTION' | 'CREATE_DEPLOYMENT' | 'RETRY' | 'CONTACT_SUPPORT' | 'NONE';

export interface BlockingReason { code: string; message: string; recoveryAction: RecoveryAction; }

export interface EpisodePreviewSession {
  episodeId: string;
  status: 'READY';
  previewType: 'FRAME_SEQUENCE' | 'VIDEO';
  manifestUrl: string;
  thumbnailUrl: string;
  expiresAt: string;
}

export interface TrainingSkill {
  id: string;
  skillKey: string;
  displayName: string;
  description: string | null;
  tier: 'TUTORIAL' | 'DEMO' | 'CUSTOM';
  status: string;
}

export interface SkillOverview {
  id: string;
  /** 에피소드·학습과 이어주는 공통 키. 최신 BE는 내려주지만 배포판엔 없을 수 있어 선택 필드로 둔다. */
  skillKey?: string;
  displayName: string;
  description: string | null;
  latestVersion: number | null;
  latestVersionCreatedAt: string | null;
  deploymentStatus: 'PLANNED' | 'COLLECTING' | 'TRAINING' | 'GATE_FAILED' | 'DEPLOYED';
  latestTrainingRequestedAt: string | null;
}

export type TaskReadiness = 'PREPARING' | 'READY' | 'BLOCKED' | 'UNKNOWN';
export interface TaskReadinessResult {
  valid: boolean; readiness: TaskReadiness; robotId: string; taskVersionId: string;
  deploymentId: string | null; observedAt: string; blockingReasons: BlockingReason[]; warnings: string[];
}

export type RobotExecutionStatus = 'REQUESTED' | 'STARTING' | 'RUNNING' | 'CANCEL_REQUESTED' | 'CANCELED' | 'SUCCEEDED' | 'FAILED' | 'STATUS_UNKNOWN';
export interface RobotExecution {
  id: string; robotId: string; taskVersionId: string; deploymentId: string; status: RobotExecutionStatus;
  currentStep: number | null; totalSteps: number; progressPercent: number;
  error: { code: string; message: string; recoveryAction: RecoveryAction } | null;
  requestedAt: string; startedAt: string | null; finishedAt: string | null; updatedAt: string;
}

export interface Training {
  id: string;
  skillKey: string;
  umiDeviceId: string;
  inputMode: 'COLLECT_NEW' | 'EXISTING_EPISODES';
  expectedEpisodeCount: number;
  receivedEpisodeCount: number;
  apiStatus: string;
  name: string;
  robot: string;
  progress: number;
  stage: string;
  status: TrainingStatus;
  started: string;
}

export interface WorkflowBlock {
  id: number;
  kind: BlockKind;
  title: string;
  /** 학습된 Skill 블록일 때 참조하는 Skill ID */
  skillId?: string;
  motor?: string;
  angle?: number;
  repeat?: number;
  children?: WorkflowBlock[];
}

/** 학습이 완료되어 WORKFLOW에서 사용할 수 있는 단일 동작 */
export interface LearnedSkill {
  id: string;
  skillId?: string;
  name: string;
  kind: Extract<BlockKind, 'umi' | 'vlm'>;
  description: string;
  mark: string;
}

/** 여러 Skill과 제어 블록을 순서대로 조합한 저장 작업 */
export interface SavedWork {
  id: string;
  taskKey?: string;
  taskVersionId?: string | null;
  version?: number | null;
  name: string;
  /** 편집기에서 마지막으로 선택한 로봇. 실행 시 다른 로봇으로 변경할 수 있다. */
  robotId: string;
  blocks: WorkflowBlock[];
  estimatedSeconds: number;
  updatedAt: string;
}

export interface BlockCatalogItem {
  kind: BlockKind;
  title: string;
  description: string;
  mark: string;
}

export interface AdminUser {
  id: string;
  name: string;
  team: string;
}

export interface LearningDraft {
  step: number;
  skillKey: string;
  method: 'new' | 'existing' | null;
  umiId: string;
  expectedEpisodes: number;
  /** 선택한 Episode ID(UUID) */
  selected: string[];
  progress: number;
  startedAt: number | null;
  name: string;
  error?: string;
  trainingJobId?: string;
}
