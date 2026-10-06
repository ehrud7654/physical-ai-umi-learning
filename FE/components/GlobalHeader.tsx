'use client';

import { useState, useEffect, useRef } from 'react';
import type { View } from '@/types';
function NavIcon({ name }: { name: 'dashboard' | 'skills' | 'learning' | 'work' | 'devices' }) {
  const paths = {
    dashboard: <><rect x="3" y="3" width="7" height="7" rx="2" /><rect x="14" y="3" width="7" height="7" rx="2" /><rect x="3" y="14" width="7" height="7" rx="2" /><rect x="14" y="14" width="7" height="7" rx="2" /></>,
    skills: <><path d="M5 4h14v16H5z" /><path d="M8 8h8M8 12h8M8 16h5" /></>,
    learning: <><path d="M3 5.5 12 2l9 3.5-9 3.5-9-3.5Z" /><path d="M6 8v6.5c0 1.7 2.7 3.5 6 3.5s6-1.8 6-3.5V8" /><path d="M21 6v7" /></>,
    work: <><rect x="5" y="4" width="14" height="17" rx="2" /><path d="M9 4V2h6v2M8.5 10h7M8.5 15h5" /></>,
    devices: <><rect x="3" y="4" width="18" height="13" rx="2" /><path d="M8 21h8M12 17v4" /><circle cx="12" cy="10.5" r="2.5" /></>,
  };
  return <span className="global-nav-icon" aria-hidden="true"><svg viewBox="0 0 24 24">{paths[name]}</svg></span>;
}

export function GlobalHeader({ authenticated, activeView, onHome, onNavigate, onAuth, variant = 'header', userId = 'admin01' }: { authenticated: boolean; activeView?: View; onHome: () => void; onNavigate: (view: View) => void; onAuth: () => void; variant?: 'header' | 'sidebar'; userId?: string }) {
  const [menuOpen, setMenuOpen] = useState(false);
  const menuRef = useRef<HTMLElement>(null);
  useEffect(() => {
    if (menuOpen) menuRef.current?.scrollTo({ top: 0, left: 0 });
  }, [menuOpen]);
  const goTo = (view: View) => {
    setMenuOpen(false);
    onNavigate(view);
  };
  return <header className={`global-header${variant === 'sidebar' ? ' app-sidebar' : ''}${menuOpen ? ' menu-open' : ''}`}>
    <button className="global-brand" onClick={onHome} aria-label="UMI Studio 메인으로 이동"><img src="/umi-studio-mark.svg" alt="UMI Studio" />{variant === 'header' && <span className="brand-fullname">Physical AI Robot Learning</span>}</button>
    <nav ref={menuRef} className={`global-nav${menuOpen ? ' open' : ''}`} aria-label="플랫폼 메뉴">
      <div className="global-nav-item"><button className={activeView === 'dashboard' ? 'active' : ''} onClick={() => goTo('dashboard')}><NavIcon name="dashboard" />대시 보드</button></div>
      <div className="global-nav-item"><button className={activeView === 'skills' ? 'active' : ''} onClick={() => goTo('skills')}><NavIcon name="skills" />동작 목록</button></div>
      <div className="global-nav-item"><button className={activeView === 'training' || activeView === 'learn' ? 'active' : ''} onClick={() => goTo('training')}><NavIcon name="learning" />학습</button></div>
      <div className="global-nav-item"><button className={activeView === 'library' || activeView === 'work' ? 'active' : ''} onClick={() => goTo('library')}><NavIcon name="work" />작업</button></div>
      <div className="global-nav-item"><button className={activeView === 'devices' || activeView === 'robot-detail' ? 'active' : ''} onClick={() => goTo('devices')}><NavIcon name="devices" />장치 관리</button></div>
      <button className="mobile-auth" onClick={() => { setMenuOpen(false); onAuth(); }}>{authenticated ? '로그아웃' : '로그인'}</button>
    </nav>
    <div className="global-actions">{variant === 'sidebar' && <span className="sidebar-greeting">안녕하세요, <strong>{userId}</strong>님</span>}<button className="global-auth" onClick={onAuth}>{authenticated ? '로그아웃' : '로그인'}</button><button className={`global-menu-toggle${menuOpen ? ' open' : ''}`} type="button" aria-label={menuOpen ? '메뉴 닫기' : '메뉴 열기'} aria-expanded={menuOpen} onClick={() => setMenuOpen((open) => !open)}><i /><i /><i /></button></div>
    {variant === 'sidebar' && menuOpen && <button className="sidebar-backdrop" type="button" aria-label="메뉴 닫기" onClick={() => setMenuOpen(false)} />}
  </header>;
}
