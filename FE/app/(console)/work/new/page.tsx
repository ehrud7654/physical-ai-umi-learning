'use client';

import { Suspense } from 'react';
import { useRouter, useSearchParams } from 'next/navigation';
import { WorkView } from '@/components/views/WorkView';
import { useToast } from '@/hooks/useToast';
import { viewToPath } from '@/lib/routes';

/* useSearchParams는 정적 렌더 시 Suspense 경계가 필요하다. */
export default function NewWorkPage() {
  return <Suspense fallback={null}><NewWorkInner /></Suspense>;
}

function NewWorkInner() {
  const router = useRouter();
  const { showToast } = useToast();
  const suggestedSkillName = useSearchParams().get('skill');
  return <WorkView editingWorkId={null} suggestedSkillName={suggestedSkillName} onNavigate={(view) => router.push(viewToPath(view))} onToast={showToast} />;
}
