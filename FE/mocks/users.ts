import type { AdminUser } from '@/types';

export const initialUsers: AdminUser[] = [
  { id: 'user01', name: '김작업', team: '시연팀' },
  { id: 'user02', name: '이연구', team: '연구개발팀' },
  { id: 'user03', name: '박테스트', team: '검증팀' },
];

export const initialPermissions: Record<string, string[]> = {
  user01: ['RB-07', 'RB-12'],
  user02: ['RB-07'],
  user03: ['RB-18'],
};
