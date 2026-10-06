import type { BlockCatalogItem } from '@/types';

/** 작업 편집기의 블록 팔레트. 서버 카탈로그가 아니라 FE 고정 정의다. */
export const blockCatalog: BlockCatalogItem[] = [
  { kind: 'umi', title: '빨간 블록 집기', description: '수집 핸들 체크포인트', mark: 'U' },
  { kind: 'vlm', title: '물체를 상자로 이동', description: 'VLM 사전 학습 작업', mark: 'V' },
  { kind: 'motor', title: '모터 움직이기', description: '모터 번호와 각도 설정', mark: 'M' },
  { kind: 'repeat', title: '반복하기', description: '안쪽 동작을 여러 번 실행', mark: '↻' },
];

/** 모터 블록의 관절별 허용 각도 범위 [min, max] */
export const motorRanges: Record<string, [number, number]> = {
  J1: [-180, 180], J2: [-90, 90], J3: [-120, 120], J4: [-180, 180], J5: [-110, 110], J6: [-180, 180],
};
