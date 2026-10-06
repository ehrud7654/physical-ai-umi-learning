'use client';

import { useRouter } from 'next/navigation';
import { WorkflowLibraryView } from '@/components/views/WorkflowLibraryView';
import { useToast } from '@/hooks/useToast';

export default function WorkLibraryPage() {
  const router = useRouter();
  const { showToast } = useToast();
  return <WorkflowLibraryView onEdit={(workId) => router.push(workId ? `/work/${encodeURIComponent(workId)}/edit` : '/work/new')} onOpen={(workId, robotId) => router.push(`/work/${encodeURIComponent(workId)}${robotId ? `?robotId=${encodeURIComponent(robotId)}` : ''}`)} onToast={showToast} />;
}
