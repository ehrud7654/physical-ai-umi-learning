'use client';

/* ponytail: 페이지 번호를 전부 렌더한다. 수십 페이지가 넘어가면 현재 페이지 주변만 보이는 윈도잉으로 바꾼다. */
export function Pagination({ page, pageCount, onChange }: { page: number; pageCount: number; onChange: (page: number) => void }) {
  if (pageCount <= 1) return null;
  const pages = Array.from({ length: pageCount }, (_, index) => index + 1);
  return (
    <nav className="pagination" aria-label="페이지 이동">
      <button className="secondary-button" type="button" disabled={page === 1} onClick={() => onChange(page - 1)}>이전</button>
      {pages.map((number) => (
        <button key={number} className={number === page ? 'primary-button' : 'secondary-button'} type="button" aria-current={number === page ? 'page' : undefined} onClick={() => onChange(number)}>{number}</button>
      ))}
      <button className="secondary-button" type="button" disabled={page === pageCount} onClick={() => onChange(page + 1)}>다음</button>
    </nav>
  );
}
