import type { Metadata } from 'next';
import './globals.css';
import { Providers } from './providers';

export const metadata: Metadata = {
  metadataBase: new URL('https://example.invalid'),
  title: 'UMI Studio · 로봇 학습 플랫폼',
  description: '누구나 쉽게 가르치는 로봇 학습 플랫폼',
  openGraph: {
    title: 'UMI Studio · 로봇 학습 플랫폼',
    description: '누구나 쉽게 가르치는 로봇 학습 플랫폼',
    images: [{ url: '/og.png', width: 1200, height: 630, alt: 'UMI Studio 로봇 학습 플랫폼' }],
  },
  twitter: {
    card: 'summary_large_image',
    title: 'UMI Studio · 로봇 학습 플랫폼',
    description: '누구나 쉽게 가르치는 로봇 학습 플랫폼',
    images: ['/og.png'],
  },
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="ko"><head><link rel="preconnect" href="https://cdn.jsdelivr.net" crossOrigin="anonymous" /><link rel="stylesheet" href="https://cdn.jsdelivr.net/gh/sun-typeface/SUIT@2/fonts/variable/woff2/SUIT-Variable.css" /><script dangerouslySetInnerHTML={{ __html: "if(location.hash.includes('figmacapture='))document.documentElement.classList.add('figma-desktop-capture')" }} /></head><body><Providers>{children}</Providers><script src="https://mcp.figma.com/mcp/html-to-design/capture.js" async /></body></html>;
}
