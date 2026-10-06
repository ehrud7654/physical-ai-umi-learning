'use client';

import { useParams, useRouter } from 'next/navigation';
import { RobotDetailView } from '@/components/views/RobotDetailView';
import { useToast } from '@/hooks/useToast';
import { viewToPath } from '@/lib/routes';

export default function RobotDetailPage() {
  const router = useRouter();
  const { showToast } = useToast();
  const { robotId } = useParams<{ robotId: string }>();
  return <RobotDetailView robotId={robotId} onBack={() => router.push('/devices?tab=robot')} onNavigate={(view) => router.push(viewToPath(view))} onToast={showToast} />;
}
