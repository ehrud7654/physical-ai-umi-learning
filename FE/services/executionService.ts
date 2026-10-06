import type { RobotExecution, TaskReadinessResult } from '@/types';
import { apiRequest } from './client';

export interface Deployment {
  id: string;
  robotId: string;
  taskVersionId: string;
  status: 'PENDING' | 'DOWNLOADING' | 'VERIFYING' | 'APPLYING' | 'SUCCEEDED' | 'FAILED' | 'CANCELED';
  progressPercent: number;
  error: { code: string; message: string } | null;
  createdAt: string;
  updatedAt: string;
}

export function fetchTaskReadiness(taskId: string, taskVersionId: string, robotId: string): Promise<TaskReadinessResult> {
  return apiRequest<TaskReadinessResult>(`/tasks/${taskId}/versions/${taskVersionId}/validate`, {
    method: 'POST',
    body: JSON.stringify({ robotId }),
  });
}

export function fetchDeployments(robotId: string): Promise<Deployment[]> {
  return apiRequest<Deployment[]>(`/robots/${robotId}/deployments`);
}

export function createDeployment(robotId: string, taskVersionId: string): Promise<Deployment> {
  return apiRequest<Deployment>(`/robots/${robotId}/deployments`, {
    method: 'POST',
    headers: { 'Idempotency-Key': crypto.randomUUID() },
    body: JSON.stringify({ taskVersionId }),
  });
}

export function fetchDeployment(robotId: string, deploymentId: string): Promise<Deployment> {
  return apiRequest<Deployment>(`/robots/${robotId}/deployments/${deploymentId}`);
}

export function createRobotExecution(input: { robotId: string; taskVersionId: string; deploymentId: string }): Promise<RobotExecution> {
  return apiRequest<RobotExecution>(`/robots/${input.robotId}/executions`, {
    method: 'POST',
    headers: { 'Idempotency-Key': crypto.randomUUID() },
    body: JSON.stringify({ taskVersionId: input.taskVersionId, deploymentId: input.deploymentId }),
  });
}

export async function fetchRobotExecutions(robotId: string): Promise<{ data: RobotExecution[]; page: { nextCursor: null; hasNext: false; totalCount: number } }> {
  const data = await apiRequest<RobotExecution[]>(`/robots/${robotId}/executions`);
  return { data, page: { nextCursor: null, hasNext: false, totalCount: data.length } };
}

export function fetchRobotExecution(robotId: string, executionId: string): Promise<RobotExecution> {
  return apiRequest<RobotExecution>(`/robots/${robotId}/executions/${executionId}`);
}

export function cancelRobotExecution(robotId: string, executionId: string): Promise<RobotExecution> {
  return apiRequest<RobotExecution>(`/robots/${robotId}/executions/${executionId}/cancel`, {
    method: 'POST',
    headers: { 'Idempotency-Key': crypto.randomUUID() },
  });
}
