import type { LoginResponse } from '@/types';
import { emailValidationError, normalizeEmail } from '../lib/authValidation';
import { apiRequest, clearAccessToken, acceptSession, refreshAuthentication, sessionRequest, withSessionLock, broadcastLogout, ApiError } from './client';

const AUTH_SESSION_KEY = 'umi:auth-session';

export async function checkEmailAvailability(email: string, signal?: AbortSignal): Promise<{ email: string; available: boolean }> {
  const error = emailValidationError(email);
  if (error) throw new Error(error);
  return apiRequest(`/auth/email-availability?email=${encodeURIComponent(normalizeEmail(email))}`, { signal, cache: 'no-store' });
}

export async function signup(name: string, email: string, password: string): Promise<LoginResponse['user']> {
  const error = emailValidationError(email);
  if (error) throw new Error(error);
  return apiRequest('/auth/signup', {
    method: 'POST',
    body: JSON.stringify({ name: name.trim(), email: normalizeEmail(email), password }),
  });
}

export async function login(email: string, password: string): Promise<LoginResponse> {
  return withSessionLock(async () => {
    const result = await sessionRequest<LoginResponse>('/auth/login', { email, password });
    acceptSession(result);
    return result;
  });
}

export async function restoreAuthentication(): Promise<LoginResponse['user'] | null> {
  if (typeof window === 'undefined') return null;
  // Legacy localStorage access tokens are deliberately no longer restored.
  localStorage.removeItem(AUTH_SESSION_KEY);
  try {
    return (await refreshAuthentication()).user;
  } catch (error) {
    if (error instanceof ApiError && error.status === 401) return null;
    throw error;
  }
}

export function clearAuthentication(): void {
  clearAccessToken();
  if (typeof window !== 'undefined') {
    localStorage.removeItem(AUTH_SESSION_KEY);
    localStorage.removeItem('umi:auth-user');
  }
}

export async function logout(): Promise<void> {
  await withSessionLock(async () => {
    await sessionRequest<void>('/auth/logout');
    clearAuthentication();
    broadcastLogout();
  });
}
