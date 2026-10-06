'use client';

import { useMemo, useState } from 'react';
import { DEFAULT_PAGE_SIZE, paginate } from '@/lib/paginate';

/** 목록을 페이지 단위로 자른다. 필터 등으로 항목이 줄어 현재 페이지가 범위를 벗어나면 paginate가 마지막 페이지로 보정한다. */
export function usePagination<T>(items: T[], pageSize = DEFAULT_PAGE_SIZE) {
  const [requestedPage, setPage] = useState(1);
  const result = useMemo(() => paginate(items, requestedPage, pageSize), [items, requestedPage, pageSize]);
  return { ...result, setPage };
}
