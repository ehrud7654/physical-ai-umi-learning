export const DEFAULT_PAGE_SIZE = 12;

/** 클라이언트 페이지네이션 순수 계산. page는 1부터 시작하며 범위를 벗어나면 안전하게 보정한다. */
export function paginate<T>(items: T[], page: number, pageSize = DEFAULT_PAGE_SIZE): { page: number; pageCount: number; pageItems: T[] } {
  const pageCount = Math.max(1, Math.ceil(items.length / pageSize));
  const safePage = Math.min(Math.max(1, Math.floor(page) || 1), pageCount);
  const start = (safePage - 1) * pageSize;
  return { page: safePage, pageCount, pageItems: items.slice(start, start + pageSize) };
}
