'use client';

import { Suspense, useEffect, useRef } from 'react';
import { useRouter, useSearchParams } from 'next/navigation';
import { LearnView } from '@/components/views/LearnView';
import { useAppStore } from '@/hooks/useAppStore';
import { useToast } from '@/hooks/useToast';
import { viewToPath } from '@/lib/routes';

/* useSearchParams는 정적 렌더 시 Suspense 경계가 필요하다. */
export default function NewLearningPage() {
  return <Suspense fallback={null}><NewLearningInner /></Suspense>;
}

function NewLearningInner() {
  const router = useRouter();
  const { showToast } = useToast();
  const { learningDraft, resetLearningDraft } = useAppStore();
  /* 동작 상세에서 ‘학습 시작’으로 들어오면 ?skill=로 동작이 정해져 있다 — 동작 선택 단계를 건너뛴다. */
  const presetSkillKey = useSearchParams().get('skill');

  /* 학습 시작 전(step<5)에 화면을 떠나면 미완성 draft를 버린다 — 예전 navigateTo/onHome/logout 가드를 route 이탈로 통일. */
  const stepRef = useRef(learningDraft.step);
  useEffect(() => {
    stepRef.current = learningDraft.step;
  }, [learningDraft.step]);
  useEffect(() => () => {
    if (stepRef.current < 5) resetLearningDraft();
  }, [resetLearningDraft]);

  return <LearnView
    onNavigate={(view) => router.push(viewToPath(view))}
    onToast={showToast}
    presetSkillKey={presetSkillKey}
    onDone={(trainingId) => router.replace(trainingId ? `/learning/${encodeURIComponent(trainingId)}` : viewToPath('training'))}
  />;
}
