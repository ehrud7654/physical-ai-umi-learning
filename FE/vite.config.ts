import tailwindcss from '@tailwindcss/postcss';
import vinext from 'vinext';
import { defineConfig, loadEnv, type ProxyOptions } from 'vite';

// 실서버 CORS 허용목록엔 프로덕션 origin만 있어 로컬 origin은 403이 난다.
// dev 프록시에서 Origin/Referer를 떼어 same-origin 요청처럼 보이게 한다.
const stripBrowserOrigin: NonNullable<ProxyOptions['configure']> = (proxy) => {
  proxy.on('proxyReq', (proxyReq) => {
    proxyReq.removeHeader('origin');
    proxyReq.removeHeader('referer');
  });
};

export default defineConfig(({ mode }) => {
  // 로컬 FE에서 원격 BE를 쓸 때(CORS 회피용, dev 서버 전용): `.env.local`에
  // DEV_API_PROXY=http://<be-host> 와 NEXT_PUBLIC_API_BASE_URL=/api/v1 을 두면
  // /api 요청을 그 호스트로 프록시한다. 미설정 시 아무 동작도 하지 않는다.
  const proxyTarget = process.env.DEV_API_PROXY ?? loadEnv(mode, process.cwd(), '').DEV_API_PROXY;

  return {
    css: { postcss: { plugins: [tailwindcss()] } },
    server: proxyTarget
      ? { proxy: { '/api': { target: proxyTarget, changeOrigin: true, configure: stripBrowserOrigin } } }
      : {},
    plugins: [vinext()],
  };
});
