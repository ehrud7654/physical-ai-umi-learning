import type { LoginResponse } from '@/types';

/** 네트워크 지연을 흉내 낸다. URL에 ?mockError 가 있으면 통신 실패를 흉내 낸다. */
export async function delay(ms = 350): Promise<void> {
  await new Promise((resolve) => setTimeout(resolve, ms));
  if (typeof window !== 'undefined' && new URLSearchParams(window.location.search).has('mockError')) {
    throw new Error('서버와 통신하지 못했어요.');
  }
}

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? 'http://localhost:8080/api/v1';
let accessToken: string | null = null;
let sessionExpirationNotified = false;
let accessExpiresAt = 0;
let sessionExpiresAt = 0;
let absoluteExpiresAt = 0;
let sessionExpected = false;
let sessionEpoch = 0;
let latestSession: LoginResponse | null = null;
let refreshPromise: Promise<LoginResponse> | null = null;
let channel: BroadcastChannel | null = null;
let clearedAt = 0;
const SESSION_SIGNAL_KEY = 'umi:session-signal';
export const SESSION_UPDATED_EVENT = 'umi:session-updated';
export const SESSION_LOGGED_OUT_EVENT = 'umi:session-logged-out';

export const SESSION_EXPIRED_EVENT = 'umi:session-expired';

export class ApiError extends Error {
  constructor(public readonly status: number, public readonly code: string, message: string) {
    super(message);
  }
}

/* 한글 문구지만 개발 용어(배포·스킬 버전·프로파일·taskKey)가 섞인 서버 메시지 — 코드로 항상 바꾼다. */
const JARGON_MESSAGES: Record<string, string> = {
  TASK_DEPLOYED: '로봇에 적용된 작업은 삭제할 수 없어요.',
  TASK_KEY_ALREADY_EXISTS: '같은 작업이 이미 있어요. 다시 시도해 주세요.',
  MODEL_INCOMPATIBLE: '이 로봇에서 실행할 수 없는 동작이 포함되어 있어요.',
  SKILL_VERSION_GATE_FAILED: '검증을 통과하지 못한 동작이 포함되어 있어요.',
  SKILL_VERSION_NOT_READY: '아직 실행할 수 없는 동작이 포함되어 있어요.',
  SKILL_VERSION_REQUIRED: '학습된 동작을 선택해 주세요.',
};
/* 서버 원문이 날것일 때(영어 예외·Spring 검증 문구·내부 클래스명) 쓰는 코드별 문구. */
const RAW_FALLBACK_MESSAGES: Record<string, string> = {
  VALIDATION_ERROR: '입력한 내용을 다시 확인해 주세요.',
  RESOURCE_NOT_FOUND: '요청한 정보를 찾을 수 없어요. 새로고침 후 다시 시도해 주세요.',
  STATE_CONFLICT: '이미 처리되었거나 지금은 할 수 없는 요청이에요. 새로고침 후 다시 확인해 주세요.',
};

function statusFallbackMessage(status: number): string {
  if (status === 400 || status === 422) return '입력한 내용을 다시 확인해 주세요.';
  if (status === 403) return '이 작업을 할 권한이 없어요.';
  if (status === 404) return '요청한 정보를 찾을 수 없어요. 새로고침 후 다시 시도해 주세요.';
  if (status === 409) return '이미 처리되었거나 지금은 할 수 없는 요청이에요. 새로고침 후 다시 확인해 주세요.';
  if (status >= 500) return '서버에 문제가 생겼어요. 잠시 후 다시 시도해 주세요.';
  return '서버 요청을 처리하지 못했어요.';
}

/**
 * 서버·로봇이 준 메시지를 사용자 문구로 거른다. 모든 API 오류와 작업 검증 사유가 여기를 지난다.
 * 날것 판정: 한글이 없거나, 3글자 이상 영단어가 섞였다('displayName: must not be blank', 'Task을(를) 찾을 수 없습니다.',
 * 'execution is already terminal'). 사람이 쓴 한글 문구('로봇이 오프라인입니다.' 등)는 그대로 둔다.
 */
export function friendlyServerMessage(code: string | undefined, message: string | undefined, status = 0): string {
  if (code && JARGON_MESSAGES[code]) return JARGON_MESSAGES[code];
  if (code === 'ROBOT_BUSY' && message?.includes('배포')) return '로봇이 다른 작업을 받는 중이에요. 잠시 후 다시 시도해 주세요.';
  if (message && /[가-힣]/.test(message) && !/[A-Za-z]{3,}/.test(message)) return message;
  return (code && RAW_FALLBACK_MESSAGES[code]) || statusFallbackMessage(status);
}

export function setAccessToken(token: string): void {
  accessToken = token;
  sessionExpirationNotified = false;
}

export function clearAccessToken(): void {
  accessToken = null;
  accessExpiresAt = 0;
  sessionExpiresAt = 0;
  absoluteExpiresAt = 0;
  latestSession = null;
  sessionExpected = false;
  sessionEpoch++;
  clearedAt = Date.now();
}

function sessionChannel(): BroadcastChannel | null {
  if (typeof window === 'undefined' || typeof BroadcastChannel === 'undefined') return null;
  if (!channel) {
    channel = new BroadcastChannel('umi:auth');
    channel.onmessage = ({ data }) => {
      if (data?.type === 'session' && data.sentAt > clearedAt) acceptSession(data.session as LoginResponse, false);
      if (data?.type === 'deadlines' && latestSession && data.userId === latestSession.user.id
          && data.sentAt > clearedAt) updateSessionDeadlines(data.session, false);
      if (data?.type === 'logout') {
        clearAccessToken();
        window.dispatchEvent(new Event(SESSION_LOGGED_OUT_EVENT));
      }
    };
  }
  return channel;
}

if (typeof window !== 'undefined') {
  window.addEventListener('storage', (event) => {
    if (event.key === SESSION_SIGNAL_KEY && event.newValue) {
      clearAccessToken();
      window.dispatchEvent(new Event(SESSION_LOGGED_OUT_EVENT));
    }
  });
}

export async function withSessionLock<T>(work: () => Promise<T>): Promise<T> {
  sessionChannel();
  if (typeof navigator !== 'undefined' && navigator.locks) {
    return navigator.locks.request('umi:auth-cookie', work);
  }
  return work();
}

export function acceptSession(session: LoginResponse, broadcast = true): void {
  if (latestSession && (latestSession.user.id !== session.user.id
      || latestSession.absoluteExpiresAt !== session.absoluteExpiresAt)) {
    sessionEpoch++;
    clearedAt = Date.now();
  }
  setAccessToken(session.accessToken);
  sessionExpected = true;
  accessExpiresAt = Date.now() + session.expiresIn * 1000;
  latestSession = session;
  updateSessionDeadlines(session, false);
  if (typeof window !== 'undefined') {
    // Remove legacy persisted credentials; retain only non-secret display information.
    localStorage.removeItem('umi:auth-session');
    localStorage.setItem('umi:auth-user', JSON.stringify(session.user));
    window.dispatchEvent(new CustomEvent(SESSION_UPDATED_EVENT, { detail: session.user }));
  }
  if (broadcast) sessionChannel()?.postMessage({ type: 'session', session, sentAt: Date.now() });
}

export function updateSessionDeadlines(session: { sessionExpiresAt: string; absoluteExpiresAt: string }, broadcast = true): void {
  if (latestSession && session.absoluteExpiresAt !== latestSession.absoluteExpiresAt) return;
  sessionExpiresAt = Date.parse(session.sessionExpiresAt);
  absoluteExpiresAt = Date.parse(session.absoluteExpiresAt);
  if (broadcast) sessionChannel()?.postMessage({ type: 'deadlines', session, userId: latestSession?.user.id, sentAt: Date.now() });
}

export function getSessionDeadline(): number {
  return Math.min(sessionExpiresAt, absoluteExpiresAt);
}

export function broadcastLogout(): void {
  sessionChannel()?.postMessage({ type: 'logout' });
  if (typeof window !== 'undefined') localStorage.setItem(SESSION_SIGNAL_KEY, crypto.randomUUID());
}

function sessionExpired(): void {
  if (sessionExpirationNotified) return;
  sessionExpirationNotified = true;
  clearAccessToken();
  if (typeof window !== 'undefined') window.dispatchEvent(new Event(SESSION_EXPIRED_EVENT));
  broadcastLogout();
}

/** Cookie-only calls must not attach a possibly expired Authorization header. */
export async function sessionRequest<T>(path: string, body?: unknown): Promise<T> {
  return send<T>(path, {
    method: 'POST', credentials: 'include', cache: 'no-store',
    headers: { 'Content-Type': 'application/json', 'X-Session-Request': '1' },
    ...(body === undefined ? {} : { body: JSON.stringify(body) }),
  });
}

export function refreshAuthentication(failedToken?: string, force = false): Promise<LoginResponse> {
  if (refreshPromise) return refreshPromise;
  const epoch = sessionEpoch;
  const hadSession = !!latestSession || (typeof window !== 'undefined' && !!localStorage.getItem('umi:auth-user'));
  sessionExpected = true;
  refreshPromise = withSessionLock(async () => {
    if (epoch !== sessionEpoch) throw new ApiError(401, 'AUTH_SESSION_EXPIRED', '로그인이 종료되었습니다.');
    if (!force && latestSession && accessExpiresAt > Date.now() + 30_000
        && (failedToken === undefined || failedToken !== accessToken)) return latestSession;
    try {
      const result = await sessionRequest<LoginResponse>('/auth/refresh');
      if (epoch !== sessionEpoch) throw new ApiError(401, 'AUTH_SESSION_EXPIRED', '로그인이 종료되었습니다.');
      acceptSession(result);
      return result;
    } catch (error) {
      if (error instanceof ApiError && error.status === 401) {
        if (hadSession) sessionExpired();
        else clearAccessToken();
      }
      throw error;
    }
  }).finally(() => { refreshPromise = null; });
  return refreshPromise;
}

export async function apiRequest<T>(path: string, init: RequestInit = {}): Promise<T> {
  const requestEpoch = sessionEpoch;
  const publicAuth = path.startsWith('/auth/login') || path.startsWith('/auth/signup') || path.startsWith('/auth/email-availability');
  if (!publicAuth && sessionExpected && (!accessToken || accessExpiresAt <= Date.now() + 30_000)) {
    await refreshAuthentication();
  }
  if (!publicAuth && requestEpoch !== sessionEpoch) throw new ApiError(401, 'AUTH_SESSION_CHANGED', '로그인 정보가 변경되었습니다. 다시 시도해 주세요.');
  const sentToken = accessToken;
  const headers = new Headers(init.headers);
  if (init.body && !headers.has('Content-Type')) headers.set('Content-Type', 'application/json');
  if (sentToken && !publicAuth) headers.set('Authorization', `Bearer ${sentToken}`);

  try {
    return await send<T>(path, { ...init, headers });
  } catch (error) {
    if (!(error instanceof ApiError) || error.status !== 401 || publicAuth || !sessionExpected) throw error;
    if (requestEpoch !== sessionEpoch) throw error;
    await refreshAuthentication(sentToken ?? undefined);
    if (requestEpoch !== sessionEpoch) throw error;
    const retryHeaders = new Headers(headers);
    if (accessToken) retryHeaders.set('Authorization', `Bearer ${accessToken}`);
    try {
      // Preserve method, body and Idempotency-Key; retry authentication failures only once.
      return await send<T>(path, { ...init, headers: retryHeaders });
    } catch (retryError) {
      if (retryError instanceof ApiError && retryError.status === 401) sessionExpired();
      throw retryError;
    }
  }
}

async function send<T>(path: string, init: RequestInit): Promise<T> {

  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}${path}`, init);
  } catch (cause) {
    /* 호출부가 취소한 요청(이메일 중복 확인 등)은 그대로 넘긴다. 그 밖의 fetch 실패는 브라우저 원문('Failed to fetch') 대신 안내 문구로. */
    if (cause instanceof DOMException && cause.name === 'AbortError') throw cause;
    throw new ApiError(0, 'NETWORK_ERROR', '서버에 연결하지 못했어요. 네트워크 상태를 확인해 주세요.');
  }
  const body = await response.json().catch(() => null) as { data?: T; code?: string; message?: string } | null;

  if (!response.ok) {
    throw new ApiError(
      response.status,
      body?.code ?? 'API_ERROR',
      friendlyServerMessage(body?.code, body?.message, response.status),
    );
  }
  if (response.status === 204) return undefined as T;
  if (!body || body.data === undefined) throw new ApiError(response.status, 'INVALID_RESPONSE', '서버 응답 형식이 올바르지 않아요.');
  return body.data;
}
