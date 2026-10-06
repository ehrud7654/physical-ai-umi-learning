'use client';

import { useParams, useRouter } from 'next/navigation';
import { WorkView } from '@/components/views/WorkView';
import { useToast } from '@/hooks/useToast';
import { viewToPath } from '@/lib/routes';

export default function EditWorkPage() {
  const router = useRouter();
  const { showToast } = useToast();
  const { workId } = useParams<{ workId: string }>();
  return <WorkView editingWorkId={workId} suggestedSkillName={null} onNavigate={(view) => router.push(viewToPath(view))} onToast={showToast} />;
}
