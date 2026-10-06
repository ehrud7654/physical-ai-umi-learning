import type { View } from '@/types';

/** 화면(View) → URL. 기존 뷰 컴포넌트의 onNavigate(view) 호출을 그대로 살리기 위한 어댑터. */
const paths: Record<View, string> = {
  dashboard: '/dashboard',
  skills: '/skills',
  training: '/learning',
  learn: '/learning/new',
  library: '/work',
  work: '/work/new',
  devices: '/devices',
  'robot-detail': '/devices',
  admin: '/admin',
};

export function viewToPath(view: View): string {
  return paths[view];
}

/** URL → 사이드바 활성 표시용 View. */
export function pathToView(pathname: string): View {
  if (pathname.startsWith('/learning/new')) return 'learn';
  if (pathname.startsWith('/learning')) return 'training';
  if (pathname.startsWith('/work/')) return 'work';
  if (pathname.startsWith('/work')) return 'library';
  if (pathname.startsWith('/devices/')) return 'robot-detail';
  if (pathname.startsWith('/devices')) return 'devices';
  if (pathname.startsWith('/skills')) return 'skills';
  if (pathname.startsWith('/admin')) return 'admin';
  return 'dashboard';
}
