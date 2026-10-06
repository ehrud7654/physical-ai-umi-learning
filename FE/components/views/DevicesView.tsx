'use client';

import { type FormEvent, useState, useEffect, useRef } from 'react';
import QRCode from 'qrcode';
import type { DeviceType } from '@/types';
import { useAppStore } from '@/hooks/useAppStore';
import { fetchDeviceRegistrationStatus, startDeviceEnrollment, type DeviceEnrollment } from '@/services';
export function DevicesView({ initialTab, onSelectRobot, onToast }: { initialTab: DeviceType; onSelectRobot: (id: string) => void; onToast: (message: string) => void }) {
  const [tab, setTab] = useState<DeviceType>(initialTab);
  const [sort, setSort] = useState<'recent' | 'oldest'>('recent');
  const [registerOpen, setRegisterOpen] = useState(false);
  const [selecting, setSelecting] = useState(false);
  const [selectedIds, setSelectedIds] = useState<string[]>([]);
  const [deleteOpen, setDeleteOpen] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const { robots, umis: umi, deleteDevicesById } = useAppStore();
  const recentOrder = (value: string) => value === '방금 전' ? -1 : value.startsWith('오늘') ? 0 : value.startsWith('어제') ? 1 : Number.parseInt(value, 10) || 99;
  const sortedRobots = sort === 'recent' ? [...robots] : [...robots].reverse();
  const sortedUmi = [...umi].sort((a, b) => sort === 'recent' ? recentOrder(a.collected) - recentOrder(b.collected) : recentOrder(b.collected) - recentOrder(a.collected));
  const visibleDeviceIds = (tab === 'robot' ? sortedRobots.map(item => item.jetsonDeviceId) : sortedUmi.map(item => item.deviceId)).filter((id): id is string => typeof id === 'string' && id.length > 0);
  const allSelected = visibleDeviceIds.length > 0 && visibleDeviceIds.every(id => selectedIds.includes(id));
  const toggleDevice = (id: string | undefined) => { if (id) setSelectedIds(current => current.includes(id) ? current.filter(item => item !== id) : [...current, id]); };
  const changeTab = (nextTab: DeviceType) => { setTab(nextTab); setSort('recent'); setSelecting(false); setSelectedIds([]); };
  const handleDelete = () => {
    if (!selecting) { setSelecting(true); return; }
    if (!selectedIds.length) { setSelecting(false); return; }
    setDeleteOpen(true);
  };
  const deleteConfirmed = async () => {
    setDeleting(true);
    try {
      await deleteDevicesById(selectedIds);
      onToast(`${selectedIds.length}개 장치를 삭제했어요.`);
      setSelectedIds([]); setSelecting(false); setDeleteOpen(false);
    } catch (cause) {
      onToast(cause instanceof Error ? cause.message : '장치를 삭제하지 못했어요.');
    } finally { setDeleting(false); }
  };
  return <><div className="page-heading list-page-heading title-divider"><div><h1>장치 관리</h1></div><button className="primary-button page-create-button" type="button" onClick={() => setRegisterOpen(true)}>장치 추가</button></div><div className="device-toolbar"><div className="device-tabs"><button className={tab === 'umi' ? 'active' : ''} onClick={() => changeTab('umi')}>수집 핸들 <span>{umi.length}</span></button><button className={tab === 'robot' ? 'active' : ''} onClick={() => changeTab('robot')}>로봇 <span>{robots.length}</span></button></div><div className="device-toolbar-actions">{selecting && <label className="device-select-all" aria-label="장치 전체 선택"><input className="list-select-checkbox" type="checkbox" disabled={!visibleDeviceIds.length} checked={allSelected} onChange={() => setSelectedIds(allSelected ? [] : visibleDeviceIds)} /></label>}<button className={selecting ? 'danger-button device-delete-action' : 'secondary-button device-delete-action'} onClick={handleDelete}>삭제</button><label className="device-sort"><select aria-label="장치 정렬" value={sort} onChange={(event) => setSort(event.target.value as typeof sort)}><option value="recent">최근 사용순</option><option value="oldest">오래된 순</option></select></label></div></div>
    {tab === 'robot' && !robots.length && <p className="table-empty">등록된 로봇이 없어요. ‘장치 추가’ 버튼으로 추가해 보세요.</p>}
    {tab === 'umi' && !umi.length && <p className="table-empty">등록된 수집 핸들이 없어요. ‘장치 추가’ 버튼으로 추가해 보세요.</p>}
    {tab === 'robot' ? <div className="device-grid">{sortedRobots.map(robot => <article className={`card device-card${selecting ? ' selecting' : ''}`} key={robot.id}>{selecting && <input className="list-select-checkbox device-card-checkbox" type="checkbox" aria-label={`${robot.name} 선택`} disabled={!robot.jetsonDeviceId} checked={!!robot.jetsonDeviceId && selectedIds.includes(robot.jetsonDeviceId)} onChange={() => toggleDevice(robot.jetsonDeviceId)} />}<button className="device-card-content" onClick={() => selecting ? toggleDevice(robot.jetsonDeviceId) : onSelectRobot(robot.id)}><div className="device-card-head"><span className="device-visual">R</span><span className={`status-pill status-${robot.status.replace(' ', '-')}`}><i />{robot.status}</span></div><h2>{robot.name}</h2><p>{robot.location}</p><span className="detail-link">상세 정보 보기 →</span></button></article>)}</div> : <div className="device-grid">{sortedUmi.map(item => <article className={`card device-card${selecting ? ' selecting' : ''}`} key={item.id}>{selecting && <input className="list-select-checkbox device-card-checkbox" type="checkbox" aria-label={`${item.id} 선택`} checked={selectedIds.includes(item.deviceId)} onChange={() => toggleDevice(item.deviceId)} />}<div className="device-card-content" onClick={() => selecting && toggleDevice(item.deviceId)}><div className="device-card-head"><span className="device-visual umi">U</span></div><h2>{item.id}</h2><p>{item.place}</p><dl><div><dt>최근 사용</dt><dd>{item.collected}</dd></div><div><dt>보유 데이터</dt><dd>{item.data}개</dd></div></dl></div></article>)}</div>}
    {registerOpen && <DeviceRegisterModal onClose={() => setRegisterOpen(false)} onRegistered={(type, label) => { setTab(type); onToast(`‘${label}’ 장치를 등록했어요.`); }} />}
    {deleteOpen && <div className="modal-backdrop" onMouseDown={() => !deleting && setDeleteOpen(false)}><section className="confirm-modal" role="dialog" aria-modal="true" aria-label="장치 삭제 확인" onMouseDown={(event) => event.stopPropagation()}><div className="modal-heading"><div><h2>선택한 장치를 삭제할까요?</h2><p>{selectedIds.length}개 장치가 목록에서 삭제됩니다.</p></div><button className="close-button" disabled={deleting} onClick={() => setDeleteOpen(false)} aria-label="닫기">×</button></div><div className="card-actions"><button className="secondary-button" disabled={deleting} onClick={() => setDeleteOpen(false)}>계속 사용</button><button className="danger-button" disabled={deleting} onClick={() => void deleteConfirmed()}>{deleting ? '삭제 중...' : '삭제'}</button></div></section></div>}
  </>;
}

function DeviceRegisterModal({ onClose, onRegistered }: { onClose: () => void; onRegistered: (type: DeviceType, label: string) => void }) {
  const { addDevice, refreshDevices } = useAppStore();
  const [type, setType] = useState<DeviceType>('robot');
  const [name, setName] = useState('');
  const [deviceId, setDeviceId] = useState('');
  const [location, setLocation] = useState('');
  const [error, setError] = useState('');
  const [saving, setSaving] = useState(false);
  const [registeredUmi, setRegisteredUmi] = useState<{ id: string; name: string; deviceId: string } | null>(null);
  const [enrollment, setEnrollment] = useState<DeviceEnrollment | null>(null);
  const [qrDataUrl, setQrDataUrl] = useState('');
  const [qrLoading, setQrLoading] = useState(false);
  const [qrError, setQrError] = useState('');
  const [now, setNow] = useState(() => Date.now());
  const [registrationComplete, setRegistrationComplete] = useState(false);
  const completionNotified = useRef(false);

  const issueEnrollment = async (targetDeviceId: string) => {
    setQrLoading(true);
    setQrError('');
    try {
      const issued = await startDeviceEnrollment(targetDeviceId);
      const image = await QRCode.toDataURL(issued.qrPayload, {
        width: 328,
        margin: 2,
        errorCorrectionLevel: 'M',
        color: { dark: '#000000', light: '#FFFFFF' },
      });
      setEnrollment(issued);
      setQrDataUrl(image);
      setRegistrationComplete(false);
      completionNotified.current = false;
      setNow(Date.now());
    } catch (cause) {
      setQrError(cause instanceof Error ? cause.message : 'QR 코드를 발급하지 못했어요.');
    } finally {
      setQrLoading(false);
    }
  };

  useEffect(() => {
    if (!enrollment) return;
    const timer = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(timer);
  }, [enrollment]);

  useEffect(() => {
    if (!registeredUmi || !enrollment || registrationComplete) return;
    let active = true;
    const checkRegistration = async () => {
      try {
        const status = await fetchDeviceRegistrationStatus(registeredUmi.deviceId);
        if (!active || status !== 'REGISTERED') return;
        setRegistrationComplete(true);
        await refreshDevices();
        if (!completionNotified.current) {
          completionNotified.current = true;
          onRegistered('umi', registeredUmi.id);
        }
      } catch {
        // 일시적인 조회 실패는 다음 polling 주기에 다시 확인한다.
      }
    };
    void checkRegistration();
    const timer = window.setInterval(() => void checkRegistration(), 2000);
    return () => {
      active = false;
      window.clearInterval(timer);
    };
  }, [enrollment, onRegistered, refreshDevices, registeredUmi, registrationComplete]);

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    if (!name.trim() || !deviceId.trim() || !location.trim()) {
      setError('이름, ID, 설치 위치를 모두 입력해 주세요.');
      return;
    }
    setError('');
    setSaving(true);
    try {
      const registeredDeviceId = await addDevice({ type, id: deviceId, name, location });
      if (type === 'umi') {
        setRegisteredUmi({ id: deviceId.trim(), name: name.trim(), deviceId: registeredDeviceId });
        await issueEnrollment(registeredDeviceId);
        setSaving(false);
      } else {
        onRegistered(type, name.trim());
        onClose();
      }
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : '등록에 실패했어요. 잠시 후 다시 시도해 주세요.');
      setSaving(false);
    }
  };

  if (registeredUmi) return <div className="modal-backdrop" onMouseDown={onClose}>
    <section className="register-modal umi-qr-modal" role="dialog" aria-modal="true" aria-label="수집 핸들 등록 QR" onMouseDown={(event) => event.stopPropagation()}>
      <div className="modal-heading"><div><p className="eyebrow">수집 핸들 등록</p><h2>UMI Collect 앱에서 QR 코드를 스캔해 주세요</h2><p>UMI Collect 앱으로 스캔하면 이 장치와 연결됩니다.</p></div><button type="button" className="close-button" onClick={onClose} aria-label="닫기">×</button></div>
      <div className="umi-qr-content">
        <div className="umi-qr-placeholder" aria-label={qrDataUrl ? "수집 핸들 등록 QR 코드" : "등록 QR 코드 준비 중"}>
          {qrDataUrl ? <img src={qrDataUrl} alt="수집 핸들 등록 QR 코드" /> : <span>{qrLoading ? '발급 중' : 'QR'}</span>}
        </div>
        <div className="umi-qr-state"><span>{registrationComplete ? '등록 완료' : enrollment ? '연동 대기 중' : '연동 준비 중'}</span><strong>{registeredUmi.name}</strong><p>{registeredUmi.id}</p>{enrollment && !registrationComplete && <p>남은 시간 {Math.max(0, Math.ceil((new Date(enrollment.expiresAt).getTime() - now) / 1000))}초</p>}</div>
      </div>
      <div className="umi-qr-guide"><strong>{registrationComplete ? '장치 등록이 완료됐어요' : qrError ? 'QR 코드 발급 실패' : 'UMI Collect 앱에서 QR 코드를 스캔해 주세요'}</strong><p>{registrationComplete ? '수집 핸들이 장치 목록에 추가되었습니다.' : qrError || 'QR 코드는 일회용이며 유효시간이 지나면 다시 발급해야 합니다.'}</p></div>
      <div className="card-actions umi-qr-actions"><button type="button" className="secondary-button" disabled={qrLoading || registrationComplete} onClick={() => void issueEnrollment(registeredUmi.deviceId)}>{qrLoading ? '발급 중...' : 'QR 다시 발급'}</button><button type="button" className="primary-button" disabled={!registrationComplete} onClick={onClose}>확인</button></div>
    </section>
  </div>;

  return <div className="modal-backdrop" onMouseDown={onClose}>
    <form className="register-modal" role="dialog" aria-modal="true" aria-label="장치 등록" onMouseDown={(event) => event.stopPropagation()} onSubmit={submit}>
      <div className="modal-heading"><div><h2>새 장치 등록</h2><p>작업 로봇 또는 수집 핸들을 플랫폼에 연결하세요.</p></div><button type="button" className="close-button" onClick={onClose} aria-label="닫기">×</button></div>
      <div className="form-grid">
        <label>장치 종류<select value={type} disabled={saving} onChange={(event) => setType(event.target.value as DeviceType)}><option value="robot">로봇</option><option value="umi">수집 핸들</option></select><small>등록 후 이 탭에서 바로 확인할 수 있어요.</small></label>
        <label>장치 이름<input value={name} disabled={saving} onChange={(event) => setName(event.target.value)} placeholder={type === 'robot' ? '예: 5번 피킹 로봇' : '예: 실습실 C 핸들'} /></label>
        <label>장치 ID<input value={deviceId} disabled={saving} onChange={(event) => setDeviceId(event.target.value)} placeholder={type === 'robot' ? '예: RB-30' : '예: UMI-030'} /><small>중복되지 않는 ID를 입력해 주세요.</small></label>
        <label>설치 위치<input value={location} disabled={saving} onChange={(event) => setLocation(event.target.value)} placeholder="예: 물류 창고 2층" /></label>
      </div>
      {error && <p className="form-error" role="alert">{error}</p>}
      <div className="card-actions"><button type="button" className="secondary-button" disabled={saving} onClick={onClose}>취소</button><button className="primary-button" disabled={saving}>{saving ? '등록 중...' : '장치 등록'}</button></div>
    </form>
  </div>;
}
