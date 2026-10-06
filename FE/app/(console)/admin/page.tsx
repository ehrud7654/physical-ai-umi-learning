'use client';

import { AdminView } from '@/components/views/AdminView';
import { useToast } from '@/hooks/useToast';

export default function AdminPage() {
  const { showToast } = useToast();
  return <AdminView onToast={showToast} />;
}
