export function normalizeEmail(value: string): string {
  return value.trim().toLowerCase();
}

export function emailValidationError(value: string): string | null {
  const email = normalizeEmail(value);
  if (!email) return '이메일을 입력해 주세요.';
  if (email.length > 320 || !/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)) {
    return '올바른 이메일 형식으로 입력해 주세요.';
  }
  return null;
}
