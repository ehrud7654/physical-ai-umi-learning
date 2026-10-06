# UMI Studio Frontend

UMI Studio(Physical AI Robot Learning)는 UMI로 수집한 동작 데이터를 학습하고, 학습된 동작과 제어 블록을 조합해 로봇 작업을 만드는 웹 플랫폼입니다.

프론트엔드는 **React 19 + TypeScript + Next.js App Router(vinext/Vite 런타임)** 기반이며, 화면 데이터는 모두 백엔드 REST API(`/api/v1`)에서 가져옵니다. 관리자 화면은 아직 UI 목업입니다.

## 사용자 흐름

```text
랜딩·로그인
→ 대시보드에서 장치와 학습 상태 확인
→ UMI 데이터(Episode) 확인 또는 새 학습 생성
→ 동작과 데이터 준비 방식 선택 후 학습 요청
→ 학습 진행·결과 확인
→ 학습 동작과 제어 블록으로 작업 구성·저장
→ 로봇을 선택해 작업 배포·실행·중지
```

## 화면과 라우트

| 라우트 | 화면 | 주요 기능 |
|---|---|---|
| `/` | 랜딩 | 소개, 튜토리얼, 로그인 모달. 로그인 상태면 대시보드로 이동 |
| `/dashboard` | 대시보드 | 장치·학습·작업 요약, 현재 실행 중인 작업 카드 |
| `/data` | 데이터 | Episode 상태 필터·정렬, 상세와 미리보기 |
| `/skills` | 동작 목록 | 동작 검색(이름·설명), 동작 추가, 설명 수정, 최신 버전·배포 상태 |
| `/learning` | 학습 목록 | 상태별 요약·필터, 새로고침, 상세, 취소·재시도·선택 삭제 |
| `/learning/new` | 학습 생성 | 동작·데이터 방식·Episode 선택 → 학습 요청 → 진행 확인(초안 보존) |
| `/work` | 작업 목록 | 작업명·블록 검색, 실행 중 작업 카드, 상세·편집, 선택 삭제 |
| `/work/new`, `/work/[workId]/edit` | 작업 생성·편집 | 학습 동작·모터·반복 블록 조립, 저장, 배포와 실행 |
| `/devices`, `/devices/[robotId]` | 장치 관리 | UMI·로봇 목록·정렬, 장치 등록(UMI는 QR 등록), 선택 삭제, 로봇 상세·정보 수정 |
| `/admin` | 관리자 | 사용자별 로봇 접근 권한 **(UI 목업)** |

목록 화면(동작·학습·작업)은 12개 단위 클라이언트 페이지네이션을 쓰고, 추가 버튼은 페이지 제목 우측에 통일되어 있습니다.

## 인증과 세션

- 로그인: `POST /auth/login`(이메일·비밀번호) → `accessToken`, `expiresIn`, `user`
- 저장: `localStorage['umi:auth-session']`에 토큰·만료 시각·사용자 정보 보존 → 새로고침해도 유지, 만료 시 자동 삭제
- 요청: `Authorization: Bearer {accessToken}` 헤더를 `services/client.ts`가 자동 부착
- 만료·401: 세션 만료 이벤트를 발생시켜 랜딩으로 이동
- 로그아웃: `POST /auth/logout`으로 Redis 세션을 삭제한 뒤 로컬 토큰을 정리합니다. 네트워크·503 실패 시 로그인 상태를 유지하고 재시도 안내를 표시합니다.
- 회원가입은 이메일 형식 검증 후 `/auth/email-availability`로 자동 중복 확인하고, `/auth/signup`으로 계정을 생성합니다. 가입 완료 후 로그인합니다
- 알려진 한계: 토큰이 `localStorage`에 있어 XSS에 노출될 수 있습니다. httpOnly 쿠키 전환은 BE와 함께 결정할 항목으로 남겨 두었습니다

## 데이터 동기화

SSE는 사용하지 않고 REST 재조회로 맞춥니다.

| 대상 | 방식 |
|---|---|
| 초기 데이터(장치·Episode·학습·동작·작업) | 콘솔 진입 시 `useAppStore`가 한 번에 조회, 화면별 새로고침 버튼 |
| 실행 중인 작업 카드 | 로봇별 실행 목록 5초 폴링 |
| 학습 상세 | 3초 폴링 |
| UMI QR 등록 상태 | 2초 폴링 |

## Episode 상태

| API 상태 | 화면 표시 | 화면 필터 |
|---|---|---|
| `UPLOADING` | 업로드 중 | 처리 중 |
| `VALIDATING` | 확인 중 | 처리 중 |
| `READY` | 학습 가능 | 학습 가능 |
| `INVALID` | 확인 실패 | 확인 실패 |
| `DELETING` / `DELETED` / `DELETE_FAILED` | 삭제 중 / 삭제됨 / 삭제 실패 | — |

## 핵심 용어

| 도메인 용어 | 의미 | 화면 표현 |
|---|---|---|
| UMI | 사용자가 보여주는 동작을 수집하는 장치 | UMI, 데이터 수집 장치 |
| Episode | UMI 동작 1회와 결과 파일 묶음 | 데이터 |
| Skill | 로봇이 학습한 단일 동작 | 동작, 학습 동작 |
| Task | 학습 동작과 제어 블록을 순서대로 조합한 흐름 | 작업 |
| Deployment | 저장된 작업 버전을 로봇에 준비하는 과정 | 배포 |
| Execution | 배포된 작업의 실제 실행 | 작업 실행 |

## 프로젝트 구조

```text
FE/
├─ app/
│  ├─ page.tsx                 # 랜딩·로그인
│  ├─ layout.tsx, providers.tsx # 전역 폰트·Provider(인증·스토어·토스트)
│  ├─ globals.css              # 디자인 토큰, 공통·반응형 스타일
│  ├─ Tutorial.tsx             # 랜딩 튜토리얼
│  └─ (console)/               # 로그인 필요 구간
│     ├─ layout.tsx            #   인증 가드, 사이드바, 로딩·오류 게이트, 모달 스크롤 잠금
│     ├─ dashboard/ data/ skills/ learning/ work/ devices/ admin/
│     └─ …/page.tsx            #   라우트 → components/views 연결만 담당
├─ components/
│  ├─ views/                   # 화면 단위 컴포넌트(DashboardView, SkillListView …)
│  ├─ GlobalHeader.tsx         # 사이드바·모바일 메뉴
│  ├─ ActiveRobotExecutionsCard.tsx
│  └─ Pagination.tsx
├─ hooks/
│  ├─ useAuth.tsx              # 로그인·세션 복원·만료 처리
│  ├─ useAppStore.tsx          # 공통 도메인 상태와 서비스 호출
│  ├─ useTaskDeployment.ts     # 배포·실행 상태 흐름
│  └─ usePagination.ts
├─ lib/                        # 순수 로직(라우트 매핑, 블록 카탈로그, 페이지 계산, 상태 라벨)
├─ services/                   # 도메인별 REST 래퍼(client.ts가 base URL·토큰·401 처리)
├─ types/                      # 공통 TypeScript 타입
├─ mocks/users.ts              # 관리자 화면 목업 데이터
├─ public/                     # 로고와 이미지
├─ docs/                       # UI·UX 명세, 팀 개발 안내, BE 전달사항
└─ Dockerfile                  # pnpm build → pnpm start
```

## 공개본 로컬 실행

요구 환경: Node.js 22.13 이상, pnpm 11.

```bash
cd FE
pnpm install
pnpm dev
```

기본 주소는 [http://localhost:3000](http://localhost:3000)입니다. 로그인 폼에 계정 정보를 직접 입력해 로그인합니다.

### API 주소

`services/client.ts`는 `NEXT_PUBLIC_API_BASE_URL`(기본값 `http://localhost:8080/api/v1`)로 요청합니다. 백엔드를 로컬에 띄우지 않고 배포 서버를 쓰려면 `FE/.env.local`(git 제외)에 다음을 두면 dev 서버가 `/api` 요청을 프록시하고 CORS를 우회합니다.

```env
NEXT_PUBLIC_API_BASE_URL=/api/v1
DEV_API_PROXY=http://<배포 서버 호스트>
```

### 검사·빌드

```bash
pnpm lint    # ESLint (오류 0 유지, 경고는 기존 <img> 6건)
pnpm build   # vinext 프로덕션 빌드 → dist/
pnpm start   # dist/를 Node 서버로 실행(PORT, HOST 환경 변수)
```

vinext는 빌드 시 타입 검사를 하지 않습니다. `npx tsc --noEmit`으로 확인할 수 있으며, 기존 코드에 남아 있는 타입 오류 23건은 알려진 부채입니다.

## 배포

`FE/Dockerfile`이 `pnpm install --frozen-lockfile` → `pnpm build` → `pnpm start`(3000 포트)를 수행합니다. 이 공개 저장소에는 원본 프로젝트의 API 서버와 통합 Compose가 포함되지 않습니다.

## 아직 목업인 범위

- 관리자 화면의 사용자·권한 데이터(`mocks/users.ts`)와 저장 동작
- Episode 미리보기 세션 URL(`createEpisodePreviewSession`) — BE 엔드포인트 확정 전
- 동작·작업의 삭제 API(목록 UI는 BE 삭제 API 이후 추가 예정)

### 문서 바로가기

- [UI·UX 명세](docs/DESIGN_SPEC.md)

브랜드 표기와 이미지는 공개용으로 일반화했습니다. 로그인·데이터·학습 기능에는 별도 백엔드가 필요합니다.
