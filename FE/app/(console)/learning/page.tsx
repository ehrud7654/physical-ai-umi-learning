'use client';

import { useRouter } from 'next/navigation';
import { TrainingStatusView } from '@/components/views/TrainingStatusView';
import { useToast } from '@/hooks/useToast';
import { viewToPath } from '@/lib/routes';

export default function LearningPage() {
  const router = useRouter();
  const { showToast } = useToast();
  return <TrainingStatusView onNavigate={(view) => router.push(viewToPath(view))} onToast={showToast} onOpen={(trainingId) => router.push(`/learning/${encodeURIComponent(trainingId)}`)} />;
}
