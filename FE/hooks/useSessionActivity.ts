'use client';

import { useEffect } from 'react';
import { apiRequest, getSessionDeadline, refreshAuthentication, updateSessionDeadlines } from '@/services/client';

/** Polling and merely leaving a tab open do not count as user activity. */
export function useSessionActivity(authenticated: boolean): void {
  useEffect(() => {
    if (!authenticated) return;
    let active = true;
    let pendingActivity = false;
    let busy = false;
    let lastAttempt = 0;
    let lastInteractionAt = 0;
    const flush = async () => {
      if (busy || !active || Date.now() - lastAttempt < 30_000) return;
      const expired = getSessionDeadline() > 0 && getSessionDeadline() <= Date.now();
      if (!pendingActivity && !expired) return;
      if (!expired && document.visibilityState !== 'visible') return;
      busy = true;
      lastAttempt = Date.now();
      const report = pendingActivity && Date.now() - lastInteractionAt < 60_000;
      pendingActivity = false;
      try {
        if (expired) {
          // Confirm the deadline with the server; another tab may have extended it.
          await refreshAuthenticationForExpiry();
        }
        if (report && active) {
          const result = await apiRequest<{ sessionExpiresAt: string; absoluteExpiresAt: string }>(
            '/auth/session/activity', { method: 'POST' });
          if (active) updateSessionDeadlines(result);
        }
      } catch {
        // Auth failures are handled centrally; network/503 failures preserve the session.
        if (active && report) pendingActivity = true;
      } finally {
        busy = false;
      }
    };
    const interaction = (event: Event) => {
      if (!event.isTrusted || document.visibilityState !== 'visible') return;
      pendingActivity = true;
      lastInteractionAt = Date.now();
      void flush();
    };
    const events = ['pointerdown', 'pointermove', 'keydown', 'wheel', 'touchstart'];
    for (const event of events) window.addEventListener(event, interaction, { passive: true });
    const timer = window.setInterval(() => { void flush(); }, 15_000);
    return () => {
      active = false;
      window.clearInterval(timer);
      for (const event of events) window.removeEventListener(event, interaction);
    };
  }, [authenticated]);
}

async function refreshAuthenticationForExpiry(): Promise<void> {
  await refreshAuthentication(undefined, true);
}
