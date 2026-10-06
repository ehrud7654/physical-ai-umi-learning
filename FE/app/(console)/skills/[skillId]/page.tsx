'use client';

import { useParams, useRouter } from 'next/navigation';
import { SkillDetailView } from '@/components/views/SkillDetailView';
import { useToast } from '@/hooks/useToast';

export default function SkillDetailPage() {
  const router = useRouter();
  const { showToast } = useToast();
  const { skillId: rawId } = useParams<{ skillId: string }>();
  const skillId = decodeURIComponent(rawId);
  return <SkillDetailView
    skillId={skillId}
    onBack={() => router.push('/skills')}
    onStartTraining={(skillKey) => router.push(`/learning/new?skill=${encodeURIComponent(skillKey)}`)}
    onOpenTraining={(trainingId) => router.push(`/learning/${encodeURIComponent(trainingId)}`)}
    onToast={showToast}
  />;
}
