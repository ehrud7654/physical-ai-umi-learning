import type { BlockingReason, Dataset, DatasetStatus, RecoveryAction } from '@/types';
import { apiRequest } from './client';

/** GET /episodes 목록 전용 응답. 상세 메타데이터는 포함하지 않는다. */
interface ApiEpisode {
  id: string;
  fileName: string | null;
  uploadedAt: string | null;
  durationMs: number;
  umiDeviceId: string;
  umiDeviceHandleNumber: string | null;
  skillKey: string | null;
  status: 'UPLOADING' | 'VALIDATING' | 'READY' | 'INVALID' | 'DELETING' | 'DELETED' | 'DELETE_FAILED';
  retentionExpiresAt: string;
  trainable: boolean;
  blockingReasons: BlockingReason[];
  recoveryAction: RecoveryAction;
}

/** EPISODE-01: 내 Episode 전체. 서버는 최신순으로 주므로 표시 순번은 오래된 순으로 매긴다. */
export async function fetchDatasets(): Promise<Dataset[]> {
  const episodes = await apiRequest<ApiEpisode[]>('/episodes');
  const total = episodes.length;
  return episodes.map((episode, index) => toDataset(episode, total - index));
}

function toDataset(episode: ApiEpisode, seq: number): Dataset {
  const status: DatasetStatus = episode.status;
  return {
    id: episode.id,
    seq,
    fileName: episode.fileName ?? '파일 정보 없음',
    uploadedAt: episode.uploadedAt,
    date: episode.uploadedAt ? formatDate(episode.uploadedAt) : '업로드 시각 없음',
    duration: formatDuration(episode.durationMs),
    skillKey: episode.skillKey ?? '',
    umiDeviceId: episode.umiDeviceId,
    umiId: episode.umiDeviceHandleNumber ?? episode.umiDeviceId,
    status,
    retentionDays: Math.max(0, Math.ceil((Date.parse(episode.retentionExpiresAt) - Date.now()) / 86_400_000)),
    retentionExpiresAt: episode.retentionExpiresAt,
    trainable: episode.trainable,
    blockingReasons: episode.blockingReasons,
    recoveryAction: episode.recoveryAction,
  };
}

function formatDate(iso: string): string {
  const d = new Date(iso);
  const pad = (n: number) => String(n).padStart(2, '0');
  return `${d.getFullYear()}. ${pad(d.getMonth() + 1)}. ${pad(d.getDate())}  ${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`;
}

/** 영상 길이. `00:01`은 1분 1초로도 읽혀 `1초` / `1분 5초` / `2분`으로 쓴다. */
function formatDuration(ms: number): string {
  const total = Math.max(0, Math.round(ms / 1000));
  const minutes = Math.floor(total / 60);
  const seconds = total % 60;
  if (!minutes) return `${seconds}초`;
  return seconds ? `${minutes}분 ${seconds}초` : `${minutes}분`;
}
