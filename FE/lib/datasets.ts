import type { Dataset, SkillOverview } from '@/types';

/**
 * 데이터 행의 시각 칸. 한 번에 몰아 찍으면 분까지 같아서 위에 `HH:mm:ss`(구분 핵심)를, 아래에 날짜를 둔다.
 * 날짜는 `오늘` / `어제` / `9월 22일`(다른 해면 연도 포함) — 렌더 시점에 계산해 자정을 넘겨도 맞다.
 */
export function datasetWhen(iso: string | null, now = new Date()): { time: string; day: string } | null {
  if (!iso) return null;
  const d = new Date(iso);
  const pad = (n: number) => String(n).padStart(2, '0');
  const dayStart = (value: Date) => new Date(value.getFullYear(), value.getMonth(), value.getDate()).getTime();
  const days = Math.round((dayStart(now) - dayStart(d)) / 86_400_000);
  const year = d.getFullYear() === now.getFullYear() ? '' : `${d.getFullYear()}년 `;
  return {
    time: `${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`,
    day: days === 0 ? '오늘' : days === 1 ? '어제' : `${year}${d.getMonth() + 1}월 ${d.getDate()}일`,
  };
}

/** 데이터(수집 에피소드) 상태 표시 메타 — 데이터 목록·상세, 학습 생성 위저드가 공유한다. */
export const datasetStatusMeta: Record<Dataset['status'], { label: string; description: string; className: string }> = {
  REGISTERED: { label: '등록 완료', description: '파일 업로드를 기다리고 있어요.', className: 'registered' },
  UPLOADING: { label: '업로드 중', description: '영상과 센서 데이터를 전송하고 있어요.', className: 'uploading' },
  VALIDATING: { label: '확인 중', description: '파일 형식과 시간 정보를 확인하고 있어요.', className: 'validating' },
  READY: { label: '학습 가능', description: '', className: 'ready' },
  INVALID: { label: '확인 실패', description: '문제를 확인하고 데이터를 다시 수집해 주세요.', className: 'invalid' },
  DELETING: { label: '삭제 중', description: '보관 기한이 지나 파일을 삭제하고 있어요.', className: 'validating' },
  DELETED: { label: '삭제됨', description: '보관 기한이 지나 파일이 삭제됐어요.', className: 'invalid' },
  DELETE_FAILED: { label: '삭제 실패', description: '파일 삭제에 실패했어요. 관리자에게 문의해 주세요.', className: 'invalid' },
};

export const deploymentStatusMeta: Record<SkillOverview['deploymentStatus'], { label: string; className: string }> = {
  DEPLOYED: { label: '실행 가능', className: 'ready' },
  PLANNED: { label: '학습 필요', className: 'deployment-warning' },
  COLLECTING: { label: '데이터 수집 중', className: 'deployment-warning' },
  TRAINING: { label: '학습 중', className: 'deployment-warning' },
  GATE_FAILED: { label: '검증 실패', className: 'invalid' },
};

export const recoveryActionLabel: Record<Dataset['recoveryAction'], string> = {
  WAIT: '처리가 끝날 때까지 기다려 주세요.', RETRY_UPLOAD: '데이터를 다시 전송해 주세요.', RECORD_AGAIN: '동작을 다시 보여주세요.',
  SET_LABEL: '데이터의 동작 정보를 확인해 주세요.', EXTEND_RETENTION: '보관 기간을 연장해 주세요.', CREATE_DEPLOYMENT: '로봇에 작업을 먼저 적용해 주세요.',
  RETRY: '잠시 후 다시 시도해 주세요.', CONTACT_SUPPORT: '관리자에게 문의해 주세요.', NONE: '',
};
