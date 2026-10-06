'use client';

import { createContext, useCallback, useContext, useEffect, useMemo, useState } from 'react';
import type { AdminUser, Dataset, LearnedSkill, Robot, SavedWork, Training, Umi, LearningDraft } from '@/types';
import { initialPermissions, initialUsers } from '@/mocks';
import {
  cancelTraining,
  deleteDevices,
  deleteTrainings,
  deleteWorks,
  fetchRobots,
  fetchDatasets,
  fetchSavedWorks,
  fetchLearnedSkills,
  fetchTrainings,
  fetchUmis,
  registerDevice,
  retryTraining,
  saveWork,
  updateRobot,
  type RegisterDeviceInput,
} from '@/services';

interface AppStore {
  learningDraft: LearningDraft;
  updateLearningDraft: (patch: Partial<LearningDraft>) => void;
  resetLearningDraft: () => void;
  /** 초기 데이터를 불러오는 동안 true */
  loading: boolean;
  /** 초기 데이터 로드 실패 여부 */
  loadError: boolean;
  /** 초기 데이터를 다시 불러온다. */
  refresh: () => Promise<void>;
  robots: Robot[];
  umis: Umi[];
  trainings: Training[];
  trainingRefreshing: boolean;
  trainingRefreshError: boolean;
  trainingUpdatedAt: number | null;
  refreshTrainings: () => Promise<boolean>;
  skills: LearnedSkill[];
  savedWorks: SavedWork[];
  users: AdminUser[];
  permissions: Record<string, string[]>;
  datasets: Dataset[];
  datasetRefreshing: boolean;
  datasetRefreshError: boolean;
  refreshDatasets: () => Promise<boolean>;
  /** 장치 등록. 중복 ID면 Error를 던진다. */
  addDevice: (input: RegisterDeviceInput) => Promise<string>;
  refreshDevices: () => Promise<void>;
  deleteDevicesById: (deviceIds: string[]) => Promise<void>;
  updateRobotInfo: (robotId: string, patch: { name: string; location: string }) => Promise<void>;
  cancelTrainingById: (trainingJobId: string) => Promise<void>;
  deleteTrainingsById: (trainingJobIds: string[]) => Promise<void>;
  deleteWorksById: (workIds: string[]) => Promise<void>;
  retryTraining: (training: Training) => Promise<void>;
  saveWorkflow: (work: SavedWork) => Promise<SavedWork>;
}

const emptyDraft: LearningDraft = { step: 1, skillKey: 'pick_place', method: null, umiId: 'UMI-024', expectedEpisodes: 10, selected: [], progress: 12, startedAt: null, name: '' };
const DRAFT_KEY = 'umi-learning-draft-v1';
const AppStoreContext = createContext<AppStore | null>(null);

export function AppStoreProvider({ children }: { children: React.ReactNode }) {
  const [skills, setSkills] = useState<LearnedSkill[]>([]);
  /* 초기값은 비워두고 서비스 계층에서 불러온다 — 로딩·오류 UI는 공통 게이트(project-task)가 담당한다. */
  const [learningDraft, setLearningDraft] = useState<LearningDraft>(emptyDraft);
  const [draftReady, setDraftReady] = useState(false);
  const updateLearningDraft = useCallback((patch: Partial<LearningDraft>) => setLearningDraft(current => ({ ...current, ...patch })), []);
  const resetLearningDraft = useCallback(() => setLearningDraft({ ...emptyDraft }), []);
  /* eslint-disable react-hooks/set-state-in-effect -- restore browser-local draft after hydration */
  useEffect(() => {
    try {
      const saved = JSON.parse(localStorage.getItem(DRAFT_KEY) || 'null');
      if (saved && typeof saved.skillKey === 'string' && Array.isArray(saved.selected)
          && saved.step >= 5 && saved.step <= 6 && typeof saved.trainingJobId === 'string') {
        setLearningDraft({ ...emptyDraft, ...saved });
      }
    } catch { /* Invalid or unavailable storage starts a fresh draft. */ }
    setDraftReady(true);
  }, []);
  useEffect(() => {
    if (!draftReady) return;
    try {
      if (learningDraft.step >= 5 && learningDraft.trainingJobId) localStorage.setItem(DRAFT_KEY, JSON.stringify(learningDraft));
      else localStorage.removeItem(DRAFT_KEY);
    } catch { /* 저장소를 쓸 수 없는 환경에서는 초안을 보존하지 않는다. */ }
  }, [learningDraft, draftReady]);
  /* eslint-enable react-hooks/set-state-in-effect */
  const [robots, setRobots] = useState<Robot[]>([]);
  const [umis, setUmis] = useState<Umi[]>([]);
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [datasetRefreshing, setDatasetRefreshing] = useState(false);
  const [datasetRefreshError, setDatasetRefreshError] = useState(false);
  const [trainings, setTrainings] = useState<Training[]>([]);
  const [trainingRefreshing, setTrainingRefreshing] = useState(false);
  const [trainingRefreshError, setTrainingRefreshError] = useState(false);
  const [trainingUpdatedAt, setTrainingUpdatedAt] = useState<number | null>(null);
  const [savedWorks, setSavedWorks] = useState<SavedWork[]>([]);
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState(false);

  const refresh = useCallback(async () => {
    setLoading(true);
    setLoadError(false);
    try {
      const [nextRobots, nextUmis, nextDatasets, nextTrainings, nextSkills, nextWorks] = await Promise.all([
        fetchRobots(),
        fetchUmis(),
        fetchDatasets(),
        fetchTrainings(),
        fetchLearnedSkills(),
        fetchSavedWorks(),
      ]);
      setRobots(nextRobots);
      setUmis(nextUmis);
      setDatasets(nextDatasets);
      setTrainings(nextTrainings);
      setSkills(nextSkills);
      setTrainingUpdatedAt(Date.now());
      setSavedWorks(nextWorks);
    } catch {
      setLoadError(true);
    } finally {
      setLoading(false);
    }
  }, []);

  const refreshTrainings = useCallback(async () => {
    setTrainingRefreshing(true);
    setTrainingRefreshError(false);
    try {
      setTrainings(await fetchTrainings());
      setTrainingUpdatedAt(Date.now());
      return true;
    } catch {
      setTrainingRefreshError(true);
      return false;
    } finally {
      setTrainingRefreshing(false);
    }
  }, []);

  /* eslint-disable react-hooks/set-state-in-effect -- mount 시 초기 로드 1회 */
  useEffect(() => {
    void refresh();
  }, [refresh]);
  /* eslint-enable react-hooks/set-state-in-effect */

  const addDevice = useCallback(async (input: RegisterDeviceInput) => {
    const next = await registerDevice(input);
    setRobots(next.robots);
    setUmis(next.umis);
    return next.registeredDeviceId;
  }, []);

  const refreshDatasets = useCallback(async () => {
    setDatasetRefreshing(true);
    setDatasetRefreshError(false);
    try {
      setDatasets(await fetchDatasets());
      return true;
    } catch {
      setDatasetRefreshError(true);
      return false;
    } finally {
      setDatasetRefreshing(false);
    }
  }, []);

  const refreshDevices = useCallback(async () => {
    const [nextRobots, nextUmis] = await Promise.all([fetchRobots(), fetchUmis()]);
    setRobots(nextRobots);
    setUmis(nextUmis);
  }, []);

  const deleteDevicesById = useCallback(async (deviceIds: string[]) => {
    const next = await deleteDevices(deviceIds);
    setRobots(next.robots);
    setUmis(next.umis);
  }, []);

  const updateRobotInfo = useCallback(async (robotId: string, patch: { name: string; location: string }) => {
    setRobots(await updateRobot(robotId, patch));
  }, []);

  const cancelTrainingById = useCallback(async (trainingJobId: string) => {
    setTrainings(await cancelTraining(trainingJobId));
    if (learningDraft.trainingJobId === trainingJobId) resetLearningDraft();
  }, [learningDraft.trainingJobId, resetLearningDraft]);

  const deleteTrainingsById = useCallback(async (trainingJobIds: string[]) => {
    setTrainings(await deleteTrainings(trainingJobIds));
    if (learningDraft.trainingJobId && trainingJobIds.includes(learningDraft.trainingJobId)) {
      resetLearningDraft();
    }
  }, [learningDraft.trainingJobId, resetLearningDraft]);

  const deleteWorksById = useCallback(async (workIds: string[]) => {
    setSavedWorks(await deleteWorks(workIds));
  }, []);

  const retryFailedTraining = useCallback(async (training: Training) => {
    setTrainings(await retryTraining(training));
  }, []);

  const saveWorkflow = useCallback(async (work: SavedWork) => {
    const result = await saveWork(work);
    setSavedWorks(result.works);
    const saved = result.works.find((item) => item.id === result.savedId);
    if (!saved) throw new Error('저장된 작업을 다시 조회하지 못했어요.');
    return saved;
  }, []);

  const value = useMemo<AppStore>(
    () => ({
      loading,
      loadError,
      refresh,
      robots,
      umis,
      trainings,
      trainingRefreshing,
      trainingRefreshError,
      trainingUpdatedAt,
      refreshTrainings,
      learningDraft, updateLearningDraft, resetLearningDraft,
      skills,
      savedWorks,
      users: initialUsers,
      permissions: initialPermissions,
      datasets,
      datasetRefreshing,
      datasetRefreshError,
      refreshDatasets,
      addDevice,
      refreshDevices,
      deleteDevicesById,
      updateRobotInfo,
      cancelTrainingById,
      deleteTrainingsById,
      deleteWorksById,
      retryTraining: retryFailedTraining,
      saveWorkflow,
    }),
    [loading, loadError, refresh, robots, umis, datasets, datasetRefreshing, datasetRefreshError, refreshDatasets, trainings, trainingRefreshing, trainingRefreshError, trainingUpdatedAt, refreshTrainings, learningDraft, updateLearningDraft, resetLearningDraft, skills, savedWorks, addDevice, refreshDevices, deleteDevicesById, updateRobotInfo, cancelTrainingById, deleteTrainingsById, deleteWorksById, retryFailedTraining, saveWorkflow],
  );

  return <AppStoreContext.Provider value={value}>{children}</AppStoreContext.Provider>;
}

export function useAppStore(): AppStore {
  const store = useContext(AppStoreContext);
  if (!store) throw new Error('useAppStore는 AppStoreProvider 안에서만 사용할 수 있어요.');
  return store;
}
