'use client';

import { useRouter } from 'next/navigation';
import { SkillListView } from '@/components/views/SkillListView';
import { useToast } from '@/hooks/useToast';

export default function SkillsPage() {
  const router = useRouter();
  const { showToast } = useToast();
  return <SkillListView onToast={showToast} onOpen={(skillId) => router.push(`/skills/${encodeURIComponent(skillId)}`)} />;
}
