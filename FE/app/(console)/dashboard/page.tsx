'use client';

import { useRouter } from 'next/navigation';
import { DashboardView } from '@/components/views/DashboardView';
import { viewToPath } from '@/lib/routes';

export default function DashboardPage() {
  const router = useRouter();
  return <DashboardView onNavigate={(view) => router.push(viewToPath(view))} onOpenDevices={(tab) => router.push(`/devices?tab=${tab}`)} onSelectRobot={(id) => router.push(`/devices/${encodeURIComponent(id)}`)} onOpenWork={(workId, robotId) => router.push(`/work/${encodeURIComponent(workId)}?robotId=${encodeURIComponent(robotId)}`)} />;
}
