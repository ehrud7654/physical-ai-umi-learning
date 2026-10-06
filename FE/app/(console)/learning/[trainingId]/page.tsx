'use client';

import { useParams, useRouter } from 'next/navigation';
import { TrainingDetailView } from '@/components/views/TrainingDetailView';
import { useToast } from '@/hooks/useToast';

export default function TrainingDetailPage() {
  const router = useRouter();
  const { showToast } = useToast();
  const { trainingId: rawId } = useParams<{ trainingId: string }>();
  const trainingId = decodeURIComponent(rawId);
  return <TrainingDetailView
    trainingId={trainingId}
    onBack={() => router.back()}
    onUseResult={(skillName) => router.push(`/work/new?skill=${encodeURIComponent(skillName)}`)}
    onCancelled={() => router.replace('/learning')}
    onToast={showToast}
  />;
}
