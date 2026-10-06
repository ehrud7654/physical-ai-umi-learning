'use client';

import { createContext, useCallback, useContext, useEffect, useMemo, useState } from 'react';

type ToastTone = 'success' | 'error';

interface ToastState {
  showToast: (message: string, tone?: ToastTone) => void;
}

const ToastContext = createContext<ToastState | null>(null);

export function ToastProvider({ children }: { children: React.ReactNode }) {
  const [toast, setToast] = useState('');
  const [tone, setTone] = useState<ToastTone>('success');

  const showToast = useCallback((message: string, nextTone: ToastTone = 'success') => {
    setToast(message);
    setTone(nextTone);
  }, []);

  useEffect(() => {
    if (!toast) return;
    const timer = window.setTimeout(() => setToast(''), 2200);
    return () => window.clearTimeout(timer);
  }, [toast]);

  const value = useMemo<ToastState>(() => ({ showToast }), [showToast]);

  return (
    <ToastContext.Provider value={value}>
      {children}
      {toast && <div className={`toast${tone === 'error' ? ' error' : ''}`} role="status"><span>{tone === 'error' ? '!' : '✓'}</span>{toast}</div>}
    </ToastContext.Provider>
  );
}

export function useToast(): ToastState {
  const context = useContext(ToastContext);
  if (!context) throw new Error('useToast는 ToastProvider 안에서만 사용할 수 있어요.');
  return context;
}
