'use client';

import { useState } from 'react';
import { useAppStore } from '@/hooks/useAppStore';
export function AdminView({ onToast }: { onToast: (message: string) => void }) {
  const { robots, users, permissions: storedPermissions } = useAppStore();
  const [userId, setUserId] = useState('user01');
  const [permissions, setPermissions] = useState<Record<string, string[]>>(storedPermissions);
  const allowed = permissions[userId] ?? [];
  const user = users.find((item) => item.id === userId)!;
  const toggleRobot = (robotId: string) => setPermissions((current) => ({ ...current, [userId]: allowed.includes(robotId) ? allowed.filter((id) => id !== robotId) : [...allowed, robotId] }));

  return <><div className="page-heading"><div><p className="eyebrow">관리자 전용</p><h1>사용자별 로봇 접근 권한</h1><p>각 사용자가 작업 탭에서 선택할 수 있는 로봇을 관리하세요.</p></div><span className="admin-badge">관리자</span></div><div className="admin-layout"><aside className="card user-list"><div className="admin-section-heading"><h2>일반 사용자</h2><span>{users.length}명</span></div><input className="search-input" placeholder="이름 또는 아이디 검색" />{users.map((item) => <button key={item.id} className={userId === item.id ? 'selected' : ''} onClick={() => setUserId(item.id)}><span className="avatar">{item.name[0]}</span><div><strong>{item.name}</strong><small>{item.id} · {item.team}</small></div><em>›</em></button>)}</aside><section className="card permission-panel"><div className="permission-heading"><div><span className="avatar large">{user.name[0]}</span><div><p className="eyebrow">선택한 사용자</p><h2>{user.name} <small>{user.id}</small></h2><p>{user.team} · 현재 {allowed.length}대 접근 가능</p></div></div><button className="primary-button" onClick={() => onToast(`${user.name}님의 로봇 권한을 저장했어요.`)}>변경 사항 저장</button></div><div className="permission-note"><span>i</span>권한을 해제해도 서버에 등록된 로봇은 삭제되지 않아요.</div><div className="robot-list-heading"><div><h3>서버에 등록된 로봇</h3><p>사용자가 작업 탭에서 선택할 로봇을 체크하세요.</p></div><span>{allowed.length}/{robots.length}대 허용</span></div><div className="robot-permission-list">{robots.map((robot) => <label key={robot.id} className={allowed.includes(robot.id) ? 'allowed' : ''}><input type="checkbox" checked={allowed.includes(robot.id)} onChange={() => toggleRobot(robot.id)} /><span className="robot-tile">R</span><div><strong>{robot.name}</strong><p>{robot.location}</p></div><span className={`status-pill status-${robot.status.replace(' ', '-')}`}><i />{robot.status}</span><em>{allowed.includes(robot.id) ? '접근 허용' : '접근 안 함'}</em></label>)}</div></section></div></>;
}
