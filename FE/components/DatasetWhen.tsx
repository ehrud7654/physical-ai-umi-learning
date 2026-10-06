import { datasetWhen } from '@/lib/datasets';
import type { Dataset } from '@/types';

/** 데이터 행의 시각 칸 — 위 시:분:초, 아래 오늘/어제/날짜. 동작 상세 수집 데이터 표와 학습 생성 데이터 선택이 같이 쓴다. */
export function DatasetWhen({ item }: { item: Pick<Dataset, 'uploadedAt' | 'date'> }) {
  const when = datasetWhen(item.uploadedAt);
  return <span className="dataset-when" title={item.date}>{when ? <><strong>{when.time}</strong><small>{when.day}</small></> : <small>{item.date}</small>}</span>;
}
