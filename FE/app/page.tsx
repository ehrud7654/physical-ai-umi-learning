'use client';

import { type FormEvent, useEffect, useRef, useState } from 'react';
import { useRouter } from 'next/navigation';
import type { View } from '@/types';
import { useAppStore } from '@/hooks/useAppStore';
import { useAuth } from '@/hooks/useAuth';
import { login as requestLogin, signup as requestSignup, checkEmailAvailability } from '@/services';
import { ApiError } from '@/services/client';
import { emailValidationError, normalizeEmail } from '@/lib/authValidation';
import { viewToPath } from '@/lib/routes';
import Tutorial from './Tutorial';

/* 랜딩(로그인·회원가입·튜토리얼). 콘솔 화면은 app/(console)/* route로 분리됐다. */
export default function Landing() {
  const router = useRouter();
  const { authenticated, login, logout, sessionExpiredMessage } = useAuth();
  const { refresh } = useAppStore();
  const [loginOpen, setLoginOpen] = useState(false);
  const [signupOpen, setSignupOpen] = useState(false);
  const [tutorialOpen, setTutorialOpen] = useState(false);
  const [pendingView, setPendingView] = useState<View>('dashboard');
  const [id, setId] = useState('');
  const [password, setPassword] = useState('');
  const [loginPending, setLoginPending] = useState(false);
  const [showPassword, setShowPassword] = useState(false);
  const [loginError, setLoginError] = useState('');
  const [signupName, setSignupName] = useState('');
  const [signupId, setSignupId] = useState('');
  const [signupPassword, setSignupPassword] = useState('');
  const [signupPasswordConfirm, setSignupPasswordConfirm] = useState('');
  const [signupError, setSignupError] = useState('');
  const [signupComplete, setSignupComplete] = useState(false);
  const [signupPending, setSignupPending] = useState(false);
  const [emailCheck, setEmailCheck] = useState<{ email: string; status: 'checking' | 'available' | 'taken' | 'error' } | null>(null);
  const normalizedEmail = normalizeEmail(signupId);
  const emailError = emailValidationError(signupId);
  const currentCheck = emailCheck?.email === normalizedEmail ? emailCheck : null;
  useEffect(() => {
    if (!signupOpen || emailError) return;
    const controller = new AbortController();
    let active = true;
    const timer = setTimeout(async () => {
      setEmailCheck({ email: normalizedEmail, status: 'checking' });
      try {
        const result = await checkEmailAvailability(normalizedEmail, controller.signal);
        if (active) setEmailCheck({ email: normalizedEmail, status: result.available ? 'available' : 'taken' });
      } catch {
        if (active) setEmailCheck({ email: normalizedEmail, status: 'error' });
      }
    }, 400);
    return () => { active = false; clearTimeout(timer); controller.abort(); };
  }, [signupOpen, normalizedEmail, emailError]);
  const loginForm = useRef<HTMLFormElement>(null);
  const signupForm = useRef<HTMLFormElement>(null);
  useEffect(() => {
    if (!loginOpen && !signupOpen) return;
    const activeForm = signupOpen ? signupForm.current : loginForm.current;
    const previous = document.activeElement as HTMLElement | null;
    const overflow = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    activeForm?.querySelector('input')?.focus();
    const handle = (event: KeyboardEvent) => {
      if (event.key === 'Escape' && !signupPending) { setLoginOpen(false); setSignupOpen(false); }
      if (event.key !== 'Tab') return;
      const fields = activeForm?.querySelectorAll<HTMLElement>('button:not(:disabled), input');
      if (!fields?.length) return;
      if (event.shiftKey && document.activeElement === fields[0]) { event.preventDefault(); fields[fields.length-1].focus(); }
      if (!event.shiftKey && document.activeElement === fields[fields.length-1]) { event.preventDefault(); fields[0].focus(); }
    };
    document.addEventListener('keydown', handle);
    return () => { document.removeEventListener('keydown', handle); document.body.style.overflow = overflow; previous?.focus(); };
  }, [loginOpen, signupOpen, signupPending]);
  /* eslint-disable react-hooks/set-state-in-effect */
  useEffect(() => {
    if (sessionExpiredMessage) {
      setSignupOpen(false);
      setLoginOpen(true);
    }
  }, [sessionExpiredMessage]);
  /* eslint-enable react-hooks/set-state-in-effect */
  const submitLogin = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!id.trim() || !password) {
      setLoginError('이메일과 비밀번호를 입력해 주세요.');
      return;
    }
    setLoginPending(true);
    setLoginError('');
    try {
      const result = await requestLogin(id.trim(), password);
      setLoginOpen(false);
      login(result.user);
      void refresh();
      router.push(viewToPath(pendingView));
    } catch (error) {
      setLoginError(error instanceof Error ? error.message : '로그인하지 못했어요.');
    } finally {
      setLoginPending(false);
    }
  };
  const submitSignup = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (signupPending) return;
    if (!signupName.trim()) { setSignupError('이름을 입력해 주세요.'); return; }
    if (emailError) { setSignupError(emailError); return; }
    if (currentCheck?.status !== 'available') { setSignupError('이메일 중복 확인을 완료해 주세요.'); return; }
    if (signupPassword.length < 8 || new TextEncoder().encode(signupPassword).length > 72) {
      setSignupError('비밀번호는 8자 이상, UTF-8 기준 72바이트 이하로 입력해 주세요.'); return;
    }
    if (signupPassword !== signupPasswordConfirm) { setSignupError('비밀번호가 일치하지 않아요.'); return; }
    setSignupPending(true);
    setSignupError('');
    try {
      await requestSignup(signupName, normalizedEmail, signupPassword);
      setId(normalizedEmail);
      setPassword('');
      setSignupPassword('');
      setSignupPasswordConfirm('');
      setEmailCheck(null);
      setSignupComplete(true);
      setSignupOpen(false);
      setPendingView('dashboard');
      setLoginError('');
      setLoginOpen(true);
    } catch (error) {
      if (error instanceof ApiError && error.status === 409) setEmailCheck({ email: normalizedEmail, status: 'taken' });
      setSignupError(error instanceof Error ? error.message : '회원가입하지 못했어요. 다시 시도해 주세요.');
    } finally { setSignupPending(false); }
  };
  return <main className="landing-page">
    {tutorialOpen && <Tutorial onClose={() => setTutorialOpen(false)} />}
    <header className="landing-simple-header">
      <button className="landing-simple-logo" type="button" onClick={() => setLoginOpen(false)} aria-label="UMI Studio 홈"><img src="/umi-studio-mark.svg" alt="UMI Studio" /></button>
      <div className="landing-auth-actions">{authenticated ? <><button type="button" onClick={() => router.push('/dashboard')}>대시 보드</button><button className="filled" type="button" onClick={logout}>로그아웃</button></> : <><button type="button" onClick={() => { setLoginOpen(false); setSignupOpen(true); setSignupError(''); }}>회원가입</button><button className="filled" type="button" onClick={() => { setPendingView('dashboard'); setSignupOpen(false); setLoginOpen(true); }}>로그인</button></>}</div>
    </header>
    <section className="landing-hero">
      <div className="landing-copy">
        <p>ROBOT LEARNING PLATFORM</p>
        <h1>Smart Solutions<br />All For You</h1>
        <h2>로봇 학습부터 실행까지, UMI Studio 와 함께</h2>
        <div className="landing-actions"><button type="button" onClick={() => setTutorialOpen(true)}>튜토리얼</button></div>
      </div>
    </section>
    {loginOpen && <div className="landing-login-backdrop" onMouseDown={() => setLoginOpen(false)}><form ref={loginForm} role="dialog" aria-modal="true" aria-label="로그인" className="landing-login-card" onSubmit={submitLogin} onMouseDown={(event) => event.stopPropagation()}>
      <div className="landing-login-head"><img src="/umi-studio-mark.svg" alt="UMI Studio" /><button type="button" onClick={() => setLoginOpen(false)} aria-label="로그인 닫기">×</button></div>
      {sessionExpiredMessage && <p className="session-expired-warning" role="alert">{sessionExpiredMessage}</p>}
      {signupComplete && <p className="signup-success" role="status">회원가입이 완료됐어요. 입력한 아이디로 로그인해 주세요.</p>}
      <label>이메일<input type="email" value={id} onChange={(event) => { setId(event.target.value); setLoginError(''); setSignupComplete(false); }} autoComplete="username" /></label><label htmlFor="login-password">비밀번호</label><div className="password-field"><input id="login-password" type={showPassword ? 'text' : 'password'} value={password} onChange={(event) => { setPassword(event.target.value); setLoginError(''); }} autoComplete="current-password" /><button type="button" aria-label={showPassword ? '비밀번호 숨기기' : '비밀번호 보기'} aria-pressed={showPassword} onClick={() => setShowPassword(!showPassword)}><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M3 12s3.8-5 9-5 9 5 9 5-3.8 5-9 5-9-5-9-5z" /><circle cx="12" cy="12" r="2.5" />{showPassword && <path d="M4 4l16 16" />}</svg></button></div>{loginError && <p className="field-error" role="alert">{loginError}</p>}
      <button className="primary-button" disabled={loginPending}>{loginPending ? '로그인 중...' : '로그인'}</button>
    </form></div>}
    {signupOpen && <div className="landing-login-backdrop" onMouseDown={() => { if (!signupPending) setSignupOpen(false); }}><form ref={signupForm} role="dialog" aria-modal="true" aria-label="회원가입" className="landing-login-card signup-card" onSubmit={submitSignup} onMouseDown={(event) => event.stopPropagation()}>
      <div className="landing-login-head"><img src="/umi-studio-mark.svg" alt="UMI Studio" /><button type="button" disabled={signupPending} onClick={() => setSignupOpen(false)} aria-label="회원가입 닫기">×</button></div>
      <label>이름<input required maxLength={100} disabled={signupPending} value={signupName} onChange={(event) => { setSignupName(event.target.value); setSignupError(''); }} autoComplete="name" placeholder="이름을 입력하세요" /></label>
      <label>아이디(이메일)<input type="email" required maxLength={320} disabled={signupPending} value={signupId} onChange={(event) => { if (normalizeEmail(event.target.value) !== normalizedEmail) setEmailCheck(null); setSignupId(event.target.value); setSignupError(''); }} aria-describedby="signup-email-status" autoComplete="username" placeholder="name@example.com" /></label>
      <p id="signup-email-status" className={currentCheck?.status === 'available' ? 'signup-success' : 'signup-email-status'} role="status" aria-live="polite">{signupId && emailError ? emailError : !signupId ? '이메일을 입력하면 중복 여부를 확인합니다.' : currentCheck?.status === 'available' ? '사용 가능한 이메일입니다.' : currentCheck?.status === 'taken' ? '이미 사용 중인 이메일입니다.' : currentCheck?.status === 'error' ? '중복 확인에 실패했습니다. 이메일을 다시 입력해 주세요.' : '이메일 중복 확인 중...'}</p>
      <label>비밀번호<input type="password" required minLength={8} maxLength={72} disabled={signupPending} value={signupPassword} onChange={(event) => { setSignupPassword(event.target.value); setSignupError(''); }} autoComplete="new-password" placeholder="8자 이상 입력하세요" /></label>
      <label>비밀번호 확인<input type="password" required disabled={signupPending} value={signupPasswordConfirm} onChange={(event) => { setSignupPasswordConfirm(event.target.value); setSignupError(''); }} autoComplete="new-password" placeholder="비밀번호를 다시 입력하세요" /></label>
      {signupError && <p className="field-error" role="alert">{signupError}</p>}
      <button className="primary-button" disabled={signupPending || !!emailError || currentCheck?.status !== 'available'}>{signupPending ? '가입 중...' : '가입하기'}</button>
      <button className="signup-login-link" type="button" disabled={signupPending} onClick={() => { setSignupOpen(false); setLoginOpen(true); }}>이미 계정이 있어요 · 로그인</button>
    </form></div>}
  </main>;
}
