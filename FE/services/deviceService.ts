import type { DeviceType, Robot, RobotDetail, Umi } from '@/types';
import { apiRequest } from './client';

type ApiDeviceType = 'UMI_CAMERA' | 'JETSON';

interface ApiDevice {
  id: string;
  ownerId: string;
  deviceType: ApiDeviceType;
  handleNumber: string;
  name: string;
  location: string | null;
  connectionStatus: 'ONLINE' | 'OFFLINE' | 'UNKNOWN';
  registrationStatus: 'PENDING' | 'REGISTERED';
  lastSeenAt: string | null;
  createdAt: string;
}

interface ApiRobot {
  id: string;
  jetsonDeviceId: string;
  name: string;
  connectionStatus: 'ONLINE' | 'OFFLINE' | 'UNKNOWN';
  busy: boolean;
  currentDeploymentId: string | null;
  currentTaskVersionId: string | null;
  lastSeenAt: string | null;
  createdAt: string;
}

export interface RegisterDeviceInput {
  type: DeviceType;
  id: string;
  name: string;
  location: string;
}

export interface DeviceEnrollment {
  enrollmentId: string;
  deviceId: string;
  qrPayload: string;
  expiresAt: string;
}

export async function fetchRobots(): Promise<Robot[]> {
  const [robots, devices] = await Promise.all([
    apiRequest<ApiRobot[]>('/robots'),
    apiRequest<ApiDevice[]>('/devices?deviceType=JETSON'),
  ]);
  const robotsByDeviceId = new Map(robots.map((robot) => [robot.jetsonDeviceId, robot]));

  // 장치 목록을 기준으로 병합해 과거 버전에서 devices에만 저장된 Jetson도 빠뜨리지 않는다.
  return devices.map((device) => {
    const robot = robotsByDeviceId.get(device.id);
    return robot ? toRobot(robot, device) : toRobotDevice(device);
  });
}

export async function fetchRobotDetail(robotId: string): Promise<RobotDetail> {
  const robot = await apiRequest<ApiRobot>(`/robots/${robotId}`);
  return {
    id: robot.id,
    jetsonDeviceId: robot.jetsonDeviceId,
    name: robot.name,
    status: robot.connectionStatus === 'UNKNOWN' ? '상태 확인 불가' : robot.busy ? '작업 중' : robot.connectionStatus === 'ONLINE' ? '온라인' : '오프라인',
    currentDeploymentId: robot.currentDeploymentId,
    currentTaskVersionId: robot.currentTaskVersionId,
    lastSeenAt: robot.lastSeenAt,
    createdAt: robot.createdAt,
  };
}

export async function fetchUmis(): Promise<Umi[]> {
  const devices = await apiRequest<ApiDevice[]>('/devices?deviceType=UMI_CAMERA');
  return devices.map(toUmi);
}

export async function registerDevice(
  input: RegisterDeviceInput,
): Promise<{ robots: Robot[]; umis: Umi[]; registeredDeviceId: string }> {
  let registeredDeviceId: string;
  if (input.type === 'robot') {
    const robot = await apiRequest<ApiRobot>('/robots', {
      method: 'POST',
      body: JSON.stringify({
        handleNumber: input.id.trim(),
        name: input.name.trim(),
        location: input.location.trim(),
      }),
    });
    registeredDeviceId = robot.jetsonDeviceId;
  } else {
    const device = await apiRequest<ApiDevice>('/devices', {
      method: 'POST',
      body: JSON.stringify({
        deviceType: toApiDeviceType(input.type),
        handleNumber: input.id.trim(),
        name: input.name.trim(),
        location: input.location.trim(),
      }),
    });
    registeredDeviceId = device.id;
  }
  const [robots, umis] = await Promise.all([fetchRobots(), fetchUmis()]);
  return { robots, umis, registeredDeviceId };
}

export function startDeviceEnrollment(deviceId: string): Promise<DeviceEnrollment> {
  return apiRequest<DeviceEnrollment>(`/devices/${deviceId}/enrollment`, { method: 'POST' });
}

export async function fetchDeviceRegistrationStatus(deviceId: string): Promise<ApiDevice['registrationStatus']> {
  const device = await apiRequest<ApiDevice>(`/devices/${deviceId}`);
  return device.registrationStatus;
}

export async function deleteDevices(
  deviceIds: string[],
): Promise<{ robots: Robot[]; umis: Umi[] }> {
  await Promise.all(deviceIds.map((id) => apiRequest<void>(`/devices/${id}`, { method: 'DELETE' })));
  const [robots, umis] = await Promise.all([fetchRobots(), fetchUmis()]);
  return { robots, umis };
}

/** devices.id를 사용해 Jetson 장치의 이름과 설치 위치를 수정한다. */
export async function updateRobot(
  deviceId: string,
  patch: { name: string; location: string },
): Promise<Robot[]> {
  await apiRequest<ApiDevice>(`/devices/${deviceId}`, {
    method: 'PATCH',
    body: JSON.stringify({ name: patch.name.trim(), location: patch.location.trim() }),
  });
  return fetchRobots();
}

function toApiDeviceType(type: DeviceType): ApiDeviceType {
  return type === 'robot' ? 'JETSON' : 'UMI_CAMERA';
}

function toRobot(robot: ApiRobot, device?: ApiDevice): Robot {
  return {
    id: robot.id,
    robotId: robot.id,
    jetsonDeviceId: robot.jetsonDeviceId,
    name: robot.name,
    location: device?.location ?? '위치 미지정',
    status: robot.connectionStatus === 'UNKNOWN' ? '상태 확인 불가' : robot.busy ? '작업 중' : robot.connectionStatus === 'ONLINE' ? '온라인' : '오프라인',
    model: 'API로 호출할 수 없습니다',
  };
}

function toRobotDevice(device: ApiDevice): Robot {
  return {
    id: device.id,
    jetsonDeviceId: device.id,
    name: device.name,
    location: device.location ?? '위치 미지정',
    status: device.connectionStatus === 'UNKNOWN' ? '상태 확인 불가' : device.connectionStatus === 'ONLINE' ? '온라인' : '오프라인',
    model: 'API로 호출할 수 없습니다',
  };
}

function toUmi(device: ApiDevice): Umi {
  return {
    id: device.handleNumber,
    deviceId: device.id,
    place: device.location ?? '위치 미지정',
    status: device.connectionStatus === 'UNKNOWN' ? '상태 확인 불가' : device.connectionStatus === 'ONLINE' ? '온라인' : '오프라인',
    collected: device.lastSeenAt ? formatLastSeen(device.lastSeenAt) : '사용 기록 없음',
    data: 0,
  };
}

function formatLastSeen(value: string): string {
  return new Intl.DateTimeFormat('ko-KR', {
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
  }).format(new Date(value));
}
