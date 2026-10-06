'use client';

import { useEffect } from 'react';
import { usePathname, useRouter } from 'next/navigation';
import { GlobalHeader } from '@/components/GlobalHeader';
import { useAppStore } from '@/hooks/useAppStore';
import { useAuth } from '@/hooks/useAuth';
import { useToast } from '@/hooks/useToast';
import { pathToView, viewToPath } from '@/lib/routes';

/* 콘솔 공통 셸: 인증 게이트 + 사이드바 + 데이터 로딩/오류 게이트. 하위 route는 화면 본문만 렌더한다. */
export default function ConsoleLayout({ children }: { children: React.ReactNode }) {
  const router = useRouter();
  const pathname = usePathname();
  const { authenticated, initializing, currentUserId, logout } = useAuth();
  const { loading, loadError, refresh } = useAppStore();
  const { showToast } = useToast();

  useEffect(() => {
    if (!initializing && !authenticated) router.replace('/');
  }, [authenticated, initializing, router]);

  useEffect(() => {
    window.scrollTo({ top: 0, behavior: 'instant' });
  }, [pathname]);

  /* 콘솔 모달(.modal-backdrop)이 열려 있는 동안 문서 스크롤을 잠근다 — 뷰 12곳을 각각 고치지 않고 한 곳에서. */
  useEffect(() => {
    const update = () => {
      document.body.style.overflow = document.querySelector('.modal-backdrop') ? 'hidden' : '';
    };
    const observer = new MutationObserver(update);
    observer.observe(document.body, { childList: true, subtree: true });
    update();
    return () => {
      observer.disconnect();
      document.body.style.overflow = '';
    };
  }, []);

  useEffect(() => {
    if (loadError) showToast('데이터를 불러오지 못했어요.', 'error');
  }, [loadError, showToast]);

  if (initializing || !authenticated) return null;

  return (
    <main className="app-shell">
      <GlobalHeader variant="sidebar" authenticated userId={currentUserId} activeView={pathToView(pathname)} onHome={() => router.push('/')} onNavigate={(view) => router.push(viewToPath(view))} onAuth={logout} />

      <section className="content">
        {loading && <div className="center-panel"><section className="card state-card"><span className="spinner" aria-hidden="true" /><h2>데이터를 불러오는 중이에요</h2><p>잠시만 기다려 주세요.</p></section></div>}
        {!loading && loadError && <div className="center-panel"><section className="card state-card"><span className="state-icon">!</span><h2>데이터를 불러오지 못했어요</h2><p>네트워크 상태를 확인한 뒤 다시 시도해 주세요.</p><button className="primary-button" onClick={() => void refresh()}>다시 시도</button></section></div>}
        {!loading && !loadError && children}
      </section>
    </main>
  );
}
