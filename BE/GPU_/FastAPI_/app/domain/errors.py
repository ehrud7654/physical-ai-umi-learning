class DomainError(Exception):
    def __init__(self, status_code: int, code: str, message: str):
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message


class RunNotFound(DomainError):
    def __init__(self):
        super().__init__(404, "RUN_NOT_FOUND", "학습 실행을 찾을 수 없습니다.")

