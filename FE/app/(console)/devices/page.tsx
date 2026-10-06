'use client';

import { Suspense } from 'react';
import { useRouter, useSearchParams } from 'next/navigation';
import { DevicesView } from '@/components/views/DevicesView';
import { useToast } from '@/hooks/useToast';

export default function DevicesPage() {
  return <Suspense fallback={null}><DevicesInner /></Suspense>;
}

function DevicesInner() {
  const router = useRouter();
  const { showToast } = useToast();
  const tab = useSearchParams().get('tab') === 'robot' ? 'robot' : 'umi';
  return <DevicesView key={tab} initialTab={tab} onSelectRobot={(id) => router.push(`/devices/${encodeURIComponent(id)}`)} onToast={showToast} />;
}
