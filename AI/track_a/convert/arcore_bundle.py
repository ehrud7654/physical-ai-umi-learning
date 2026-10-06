"""Strict bundle decoder; explicit calibration and quality policy, no guessed summary grammar."""
import csv
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import zipfile
import numpy as np
from umi.raw import RawEpisode, RawMeta, validate_raw
from umi.ik import quat_to_matrix, matrix_to_quat


def rigid(value):
    m = np.array(value, dtype=float, copy=True)
    if (m.shape != (4, 4) or not np.isfinite(m).all()
            or not np.allclose(m[3], [0, 0, 0, 1], atol=1e-8, rtol=0)
            or not np.allclose(m[:3, :3].T @ m[:3, :3], np.eye(3), atol=1e-6, rtol=0)
            or not np.isclose(np.linalg.det(m[:3, :3]), 1, atol=1e-6, rtol=0)):
        raise ValueError('invalid rigid calibration transform')
    return m


ARCORE_CAMERA_AXES = "+X right, +Y up, -Z forward (ARCore/OpenGL)"


def decode_bundle(*, bundle, t_camera_pinch, t_camera_pinch_axes,
                  t_base_world, calibration_id,
                  recording_id, skill_id, tracking_valid_values, warmup_s,
                  usable_segments, frames_dropped, image_decoder,
                  stabilized_at_s=None, demo_start_s=None,
                  pre_stabilized=False):
    camera_pinch, base_world = rigid(t_camera_pinch), rigid(t_base_world)
    if t_camera_pinch_axes != ARCORE_CAMERA_AXES:
        raise ValueError(
            "t_camera_pinch must be expressed in ARCore/OpenGL camera axes; "
            "convert OpenCV calibration with diag(1,-1,-1,1) @ T_cv first"
        )
    if not isinstance(calibration_id, str) or not calibration_id.strip():
        raise ValueError('calibration_id required')
    if type(pre_stabilized) is not bool:
        raise ValueError('pre_stabilized must be an explicit boolean')
    if (not tracking_valid_values or isinstance(tracking_valid_values, str)
            or warmup_s is None or not np.isfinite(warmup_s) or warmup_s < 0
            or (warmup_s < 3 and not pre_stabilized)
            or usable_segments is None or type(frames_dropped) is not int or frames_dropped < 0
            or not callable(image_decoder)):
        raise ValueError('explicit tracking, justified warmup, pose-row segments, drop count and RGB decoder required')
    for name, value in (('stabilized_at_s', stabilized_at_s), ('demo_start_s', demo_start_s)):
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                                  or not np.isfinite(value) or value < 0):
            raise ValueError(f'{name} must be a non-negative offset in seconds')
    start_offsets = [float(warmup_s)]
    start_offsets.extend(float(value) for value in (stabilized_at_s, demo_start_s)
                         if value is not None)
    effective_start_s = max(start_offsets)
    bundle = Path(bundle)
    archive = None if bundle.is_dir() else zipfile.ZipFile(bundle)
    hashes = {}
    try:
        if archive and len(archive.namelist()) != len(set(archive.namelist())):
            raise ValueError('duplicate ZIP entries')
        def read(name):
            if PurePosixPath(name).is_absolute() or '..' in PurePosixPath(name).parts or '\\' in name or ':' in name:
                raise ValueError('unsafe member path')
            if archive:
                if archive.getinfo(name).file_size > 32*1024*1024:
                    raise ValueError('member too large')
                data = archive.read(name)
            else:
                path = (bundle/name).resolve()
                path.relative_to(bundle.resolve())
                if path.stat().st_size > 32*1024*1024:
                    raise ValueError('member too large')
                data = path.read_bytes()
            hashes[name] = hashlib.sha256(data).hexdigest()
            return data
        def rows(name, required):
            reader = csv.DictReader(io.StringIO(read(name).decode('utf-8-sig')))
            if not reader.fieldnames or not set(required.split(',')).issubset(reader.fieldnames):
                raise ValueError('missing CSV columns: '+name)
            return list(reader)
        meta = json.loads(read('episode.json'))
        if meta.get('schema') != 'umi_raw/0.1.0':
            raise ValueError('unsupported schema')
        for key, value in dict(pose_is='T_world_camera', frame='arcore_world',
                handedness='right', units='m', quaternion_order='xyzw').items():
            if meta['pose'].get(key) != value:
                raise ValueError('unsupported pose convention')
        if meta['clock'].get('source') != 'elapsedRealtimeNanos' or meta['clock'].get('all_streams_same_clock') is not True:
            raise ValueError('unverified clocks')
        cam = meta['camera']
        for k, v in dict(ois='off', vdis='off', focus_mode='fixed', ae='locked', awb='locked').items():
            if cam.get(k) != v:
                raise ValueError('camera contract mismatch')
        gripper_meta = meta.get('gripper', {})
        producer = gripper_meta.get('producer')
        if producer is not None:
            if (not isinstance(producer, dict)
                    or not all(isinstance(producer.get(k), str) and producer[k].strip()
                               for k in ('name', 'version', 'sha256'))
                    or len(producer['sha256']) != 64
                    or any(c not in '0123456789abcdefABCDEF' for c in producer['sha256'])):
                raise ValueError('invalid gripper producer provenance')
        summary = read('summary.txt').decode('utf-8-sig')
        read('imu.csv')  # provenance only; no IMU fusion
        poses = rows('poses.csv', 'index,timestamp_ns,tracking,x,y,z,qx,qy,qz,qw')
        frames = rows('frames.csv', 'frame_index,timestamp_ns,image')
        grips = rows('gripper.csv', 'frame_index,gap_m,status')
        stamps = [int(p['timestamp_ns']) for p in poses]
        if len(stamps) < 2 or min(stamps) < 0 or any(b <= a for a,b in zip(stamps,stamps[1:])):
            raise ValueError('invalid pose timestamps')
        ft = [int(f['timestamp_ns']) for f in frames]
        fi = [int(f['frame_index']) for f in frames]
        if (len(set(fi)) != len(fi) or any(b <= a for a,b in zip(ft,ft[1:]))
                or not set(ft).issubset(stamps) or len(stamps)-len(ft) != frames_dropped):
            raise ValueError('frame join/drop count mismatch')
        by_time = dict(zip(ft, frames)); by_id = {}
        for g in grips:
            index, status = int(g['frame_index']), g['status']
            if index in by_id or index not in fi or status not in ('D','M','T','X'):
                raise ValueError('invalid gripper frame/status')
            gap = float(g['gap_m']) if g['gap_m'].strip() else np.nan
            if (status in ('D','M') and (not np.isfinite(gap) or not 0 <= gap <= .09)
                    or status in ('T','X') and not np.isnan(gap)):
                raise ValueError('invalid measured/missing gap')
            by_id[index] = (gap, status)
        if set(by_id) != set(fi):
            raise ValueError('missing gripper frame')
        allowed = np.zeros(len(poses), dtype=bool)
        segment_ids = np.full(len(poses), -1)
        for segment_id, segment in enumerate(usable_segments):
            if len(segment) != 2 or any(type(x) is not int for x in segment):
                raise ValueError('segments must be pose-row offsets')
            a,b = segment
            if not 0 <= a < b <= len(poses) or allowed[a:b].any():
                raise ValueError('invalid usable segment')
            allowed[a:b] = True
            segment_ids[a:b] = segment_id
        candidates = [i for i,p in enumerate(poses) if allowed[i] and
            (stamps[i]-stamps[0])/1e9 >= effective_start_s and
            p['tracking'] in tracking_valid_values and stamps[i] in by_time]
        runs = []
        for i in candidates:
            if (not runs or i != runs[-1][-1]+1
                    or segment_ids[i] != segment_ids[runs[-1][-1]]):
                runs.append([])
            runs[-1].append(i)
        chosen = max(runs, key=len, default=[])
        if len(chosen) < 2:
            raise ValueError('no contiguous usable run')
        xyzs, quats, gaps, statuses, images = [],[],[],[],[]
        for i in chosen:
            p, f = poses[i], by_time[stamps[i]]
            xyz = np.array([float(p[k]) for k in ('x','y','z')])
            q = np.array([float(p[k]) for k in ('qw','qx','qy','qz')])
            if not np.isfinite(xyz).all() or not np.isfinite(q).all() or abs(np.linalg.norm(q)-1)>1e-6:
                raise ValueError('invalid pose')
            world_camera = np.eye(4)
            world_camera[:3,:3], world_camera[:3,3] = quat_to_matrix(q), xyz
            t = base_world @ world_camera @ camera_pinch
            xyzs.append(t[:3,3]); quats.append(matrix_to_quat(t[:3,:3]))
            gap,status = by_id[int(f['frame_index'])]
            gaps.append(gap); statuses.append(status)
            image = np.asarray(image_decoder(read(f['image'])))
            if image.dtype != np.uint8 or image.shape != (cam['height'],cam['width'],3):
                raise ValueError('decoder must preserve RGB resolution')
            images.append(image)
        times = np.array([(stamps[i]-stamps[0])/1e9 for i in chosen], dtype=np.float64)
        result = RawEpisode(RawMeta(recording_id=recording_id or meta['episode_id'],
            skill_id=skill_id or meta.get('skill_id',''), source='arcore', frame='robot_base',
            n_steps=len(chosen), pose_rate_hz=float(1/np.median(np.diff(times))), cameras=['cam_wrist'],
            calibration_id=calibration_id, bundle_schema=meta['schema'], frames_dropped=frames_dropped,
            usable_segments=[[0,len(chosen)]], notes={'source_pose_rows':chosen,
                'source_usable_segments':usable_segments, 'source_clock_origin_ns':stamps[0],
                'warmup_s':warmup_s, 'stabilized_at_s':stabilized_at_s,
                'demo_start_s':demo_start_s, 'effective_start_s':effective_start_s,
                'pre_stabilized':pre_stabilized,
                'gripper_producer':producer, 'summary_original':summary,
                'source_files_sha256':hashes,
                't_base_world':base_world.tolist(), 't_camera_pinch':camera_pinch.tolist(),
                't_camera_pinch_axes':t_camera_pinch_axes,
                'logical_camera':'cam_wrist'}), np.array(xyzs),np.array(quats),np.array(gaps),
            np.array(statuses,dtype='<U1'),times,{'cam_wrist':np.stack(images)},{'cam_wrist':times.copy()})
        problems = validate_raw(result)
        if problems:
            raise ValueError('; '.join(problems))
        return result
    finally:
        if archive:
            archive.close()
