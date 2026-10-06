'use client';

import { AppStoreProvider } from '@/hooks/useAppStore';
import { AuthProvider } from '@/hooks/useAuth';
import { ToastProvider } from '@/hooks/useToast';

/* 순서 고정: AuthProvider가 useAppStore·useToast에 의존하므로 둘보다 안쪽에 둔다. */
export function Providers({ children }: { children: React.ReactNode }) {
  return (
    <AppStoreProvider>
      <ToastProvider>
        <AuthProvider>{children}</AuthProvider>
      </ToastProvider>
    </AppStoreProvider>
  );
}
