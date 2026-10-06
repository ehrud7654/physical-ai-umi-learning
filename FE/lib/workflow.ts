import type { BlockKind, LearnedSkill, SavedWork, WorkflowBlock } from '@/types';
import { blockCatalog, motorRanges } from '@/lib/blocks';

let nextBlockId = 20;

export function makeBlock(kind: BlockKind): WorkflowBlock {
  const item = blockCatalog.find((block) => block.kind === kind)!;
  return {
    id: nextBlockId++, kind, title: item.title,
    ...(kind === 'motor' ? { motor: 'J1', angle: 45 } : {}),
    ...(kind === 'repeat' ? { repeat: 3, children: [] } : {}),
  };
}

export function makeSkillBlock(skill: LearnedSkill): WorkflowBlock {
  return { id: nextBlockId++, kind: skill.kind, title: skill.name, skillId: skill.id };
}

export const isSkillBlock = (block: WorkflowBlock) => block.kind === 'umi' || block.kind === 'vlm';

export function flattenWorkflowBlocks(blocks: WorkflowBlock[]): WorkflowBlock[] {
  return blocks.flatMap((block) => [block, ...flattenWorkflowBlocks(block.children ?? [])]);
}

export function estimateWorkflowSeconds(blocks: WorkflowBlock[]): number {
  return blocks.reduce((total, block) => {
    if (block.kind === 'repeat') {
      return total + estimateWorkflowSeconds(block.children ?? []) * (block.repeat ?? 1);
    }
    if (block.kind === 'umi') return total + 25;
    if (block.kind === 'vlm') return total + 35;
    return total + 12;
  }, 0);
}

export function formatWorkDuration(seconds: number): string {
  const minutes = Math.floor(seconds / 60);
  const rest = seconds % 60;
  if (!minutes) return `약 ${rest}초`;
  return rest ? `약 ${minutes}분 ${rest}초` : `약 ${minutes}분`;
}

export function formatUpdatedAt(value: string): string {
  return new Intl.DateTimeFormat('ko-KR', { year: 'numeric', month: '2-digit', day: '2-digit' }).format(new Date(value));
}

export function formatApiDateTimeParts(value: string): { date: string; time: string } {
  const date = new Date(value);
  return {
    date: new Intl.DateTimeFormat('ko-KR', {
      year: 'numeric', month: '2-digit', day: '2-digit',
    }).format(date),
    time: new Intl.DateTimeFormat('ko-KR', {
      hour: '2-digit', minute: '2-digit', second: '2-digit',
    }).format(date),
  };
}

export function getWorkflowIssues(work: SavedWork): string[] {
  const issues: string[] = [];
  const blocks = flattenWorkflowBlocks(work.blocks);
  if (!work.name.trim()) issues.push('작업명이 비어 있어요.');
  if (!work.blocks.length) issues.push('실행할 블록이 없어요.');
  blocks.forEach((block) => {
    if (isSkillBlock(block) && !block.skillId) issues.push(`${block.title}에 학습 버전 정보가 없어요.`);
    if (block.kind === 'motor') {
      const range = motorRanges[block.motor ?? ''];
      if (!range || block.angle === undefined || block.angle < range[0] || block.angle > range[1]) issues.push(`${block.title} 설정을 확인해 주세요.`);
    }
    if (block.kind === 'repeat' && ((block.repeat ?? 0) < 1 || (block.repeat ?? 0) > 100 || !block.children?.length)) issues.push('반복 블록에는 실행할 동작과 1~100의 반복 횟수가 필요해요.');
  });
  return issues;
}

/**
 * 저장본과 편집 중 블록이 실행 내용상 같은지. 제목은 비교하지 않는다 — 표시용이라 서버 왕복 시 달라진다
 * (모터: 팔레트 '모터 움직이기' ↔ 서버 '모터 제어', 반복: '반복하기' ↔ '반복 수행'). 학습 동작 블록도 서버는 종류를
 * 'umi' 하나로 돌려주므로 umi/vlm을 같은 것으로 본다. 제목까지 비교하면 모터·반복이 든 작업은 저장 직후에도
 * '저장 안 됨'으로 판정돼, 다시 누르면 자기 자신과 이름이 겹친다며 막혔다.
 */
export function workflowContentEquals(left: WorkflowBlock[], right: WorkflowBlock[]): boolean {
  if (left.length !== right.length) return false;
  const kindOf = (block: WorkflowBlock) => (isSkillBlock(block) ? 'skill' : block.kind);
  return left.every((block, index) => {
    const other = right[index];
    return kindOf(block) === kindOf(other)
      && block.skillId === other.skillId
      && block.motor === other.motor
      && block.angle === other.angle
      && block.repeat === other.repeat
      && workflowContentEquals(block.children ?? [], other.children ?? []);
  });
}

/** 불러온 블록 ID와 새 블록 ID가 겹치지 않게 카운터를 끌어올린다. */
export function syncNextBlockId(min: number): void {
  nextBlockId = Math.max(nextBlockId, min);
}
