import type { DashboardData } from '@/types';
import { apiRequest } from './client';

export function fetchDashboard(): Promise<DashboardData> {
  return apiRequest<DashboardData>('/dashboard');
}
