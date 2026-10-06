"""Local synthetic ARCore ZIP, trajectory and conversion provenance checks."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import io
import json
import tempfile
import unittest
import zipfile
import numpy as np
from PIL import Image
from track_a.convert.arcore import to_raw
from umi.provenance import conversion_digest
from umi.trajectory import TrajectoryLimits, continuous_mask
from tools.run_umi_regression import is_shared_gpu_server
from tools.umi_ingest_arcore import pilot_summary


class Tests(unittest.TestCase):
    def fixture(self):
        meta = dict(schema='umi_raw/0.1.0', episode_id='fixture', skill_id='pick_place',
            pose=dict(pose_is='T_world_camera',frame='arcore_world',handedness='right',
                      units='m',quaternion_order='xyzw'),
            clock=dict(source='elapsedRealtimeNanos',all_streams_same_clock=True),
            camera=dict(width=8,height=6,ois='off',vdis='off',focus_mode='fixed',ae='locked',awb='locked'),
            gripper=dict(method='marker_scale',marker_mm=16,
                         producer=dict(name='fixture-gap',version='1.0.0',sha256='a'*64)))
        members = {'episode.json':json.dumps(meta), 'summary.txt':'grammar not assumed', 'imu.csv':'timestamp_ns,sensor,x,y,z\n'}
        poses=['index,timestamp_ns,tracking,x,y,z,qx,qy,qz,qw']
        frames=['frame_index,timestamp_ns,image']; gaps=['frame_index,gap_m,status']
        for i in range(8):
            poses.append(f'{i},{1000000000+i*1000000000},TRACKING,1,0,0,{2**-.5},0,0,{2**-.5}')
            frames.append(f'{i},{1000000000+i*1000000000},frames/{i}.jpg')
            gaps.append(f'{i},'+(',X' if i==5 else '0.04,D'))
            encoded=io.BytesIO()
            Image.fromarray(np.full((6,8,3),i*20,dtype=np.uint8)).save(encoded,format='JPEG')
            members[f'frames/{i}.jpg']=encoded.getvalue()
        members.update({'poses.csv':'\n'.join(poses),'frames.csv':'\n'.join(frames),'gripper.csv':'\n'.join(gaps)})
        return members

    def decode(self, members, **overrides):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'fixture.zip'
            with zipfile.ZipFile(path,'w') as archive:
                for name,data in members.items(): archive.writestr(name,data)
            base=np.eye(4); base[:3,:3]=[[0,-1,0],[1,0,0],[0,0,1]]; base[1,3]=2
            pinch=np.eye(4); pinch[1,3]=.1
            def decoder(data):
                with Image.open(io.BytesIO(data)) as img: return np.array(img.convert('RGB'))
            args=dict(bundle=path,t_cam_to_pinch=pinch,t_arcore_world_to_base=base,
                t_cam_to_pinch_axes='+X right, +Y up, -Z forward (ARCore/OpenGL)',
                calibration_id='synthetic-only',tracking_valid_values=['TRACKING'],warmup_s=3.,
                usable_segments=[[0,8]],frames_dropped=0,image_decoder=decoder)
            args.update(overrides)
            return to_raw(**args)

    def test_zip_transform_join_gap_and_original_images(self):
        raw=self.decode(self.fixture())
        self.assertEqual(raw.meta.n_steps,5)
        np.testing.assert_allclose(raw.eef_pos, np.tile([0,3,.1],(5,1)),atol=1e-12)
        np.testing.assert_array_equal(raw.pose_timestamp,[3,4,5,6,7])
        self.assertTrue(np.isnan(raw.gripper_gap_m[2]))
        self.assertEqual(raw.images['cam_wrist'].shape,(5,6,8,3))
        self.assertEqual(set(raw.images),{'cam_wrist'})
        self.assertIn('frames/3.jpg',raw.meta.notes['source_files_sha256'])

    def test_paused_pose_does_not_bridge(self):
        members=self.fixture()
        lines=members['poses.csv'].splitlines(); lines[6]=lines[6].replace('TRACKING','PAUSED')
        members['poses.csv']='\n'.join(lines)
        raw=self.decode(members)
        self.assertEqual(raw.meta.notes['source_pose_rows'],[3,4])

    def test_adjacent_reviewed_segments_do_not_merge(self):
        raw=self.decode(self.fixture(),usable_segments=[[0,5],[5,8]])
        self.assertEqual(raw.meta.notes['source_pose_rows'],[5,6,7])

    def test_stabilization_demo_start_and_gripper_provenance(self):
        raw=self.decode(self.fixture(),stabilized_at_s=4.,demo_start_s=5.)
        self.assertEqual(raw.meta.notes['source_pose_rows'],[5,6,7])
        self.assertEqual(raw.meta.notes['effective_start_s'],5.)
        self.assertEqual(raw.meta.notes['gripper_producer']['sha256'],'a'*64)
        with self.assertRaises(ValueError):
            self.decode(self.fixture(),demo_start_s=-1.)
        members=self.fixture(); meta=json.loads(members['episode.json'])
        meta['gripper']['producer']['sha256']='not-a-sha'; members['episode.json']=json.dumps(meta)
        with self.assertRaises(ValueError): self.decode(members)
        with self.assertRaises(ValueError):
            self.decode(self.fixture(),demo_start_s='5')

    def test_pilot_summary_reports_status_gap_and_duration(self):
        raw=self.decode(self.fixture()); report=pilot_summary(raw)
        self.assertEqual(report['status_counts'],{'D':4,'M':0,'T':0,'X':1})
        self.assertEqual(report['longest_tx_run'],1)
        self.assertEqual(report['gap_min_m'],.04)
        self.assertFalse(report['pilot_duration_20s_pass'])
        self.assertTrue(report['gripper_producer_recorded'])
        self.assertEqual(report['pilot_issues'],['usable_duration_under_20s'])
        self.assertFalse(report['pilot_pass'])

        raw.pose_timestamp=np.linspace(0,20,raw.meta.n_steps)
        report=pilot_summary(raw)
        self.assertTrue(report['pilot_duration_20s_pass'])
        self.assertTrue(report['pilot_pass'])

        raw.gripper_status[:]='X'; raw.gripper_gap_m[:]=np.nan
        report=pilot_summary(raw)
        self.assertIn('no_valid_gripper_gap',report['pilot_issues'])
        self.assertFalse(report['pilot_pass'])

    def test_pilot_summary_flags_missing_gripper_producer(self):
        members=self.fixture(); meta=json.loads(members['episode.json'])
        del meta['gripper']['producer']; members['episode.json']=json.dumps(meta)
        report=pilot_summary(self.decode(members))
        self.assertFalse(report['gripper_producer_recorded'])
        self.assertIn('gripper_producer_missing',report['pilot_issues'])
        self.assertFalse(report['pilot_pass'])

    def test_duplicate_timestamp_and_nonunit_quaternion(self):
        members=self.fixture()
        lines=members['poses.csv'].splitlines()
        lines[5]=lines[5].replace('5000000000','4000000000')
        members['poses.csv']='\n'.join(lines)
        with self.assertRaises(ValueError): self.decode(members)
        members=self.fixture(); lines=members['poses.csv'].splitlines()
        fields=lines[4].split(','); fields[-1]='9'; lines[4]=','.join(fields)
        members['poses.csv']='\n'.join(lines)
        with self.assertRaises(ValueError): self.decode(members)

    def test_missing_frame_splits_and_requires_matching_drop_count(self):
        members=self.fixture()
        members['frames.csv']='\n'.join(x for x in members['frames.csv'].splitlines() if not x.startswith('5,'))
        members['gripper.csv']='\n'.join(x for x in members['gripper.csv'].splitlines() if not x.startswith('5,'))
        with self.assertRaises(ValueError): self.decode(members)
        raw=self.decode(members,frames_dropped=1)
        self.assertEqual(raw.meta.notes['source_pose_rows'],[3,4])

    def test_invalid_gap_and_unknown_status(self):
        for value in ['0.1,D','0.04,DEAD','0.04,X']:
            members=self.fixture(); members['gripper.csv']=members['gripper.csv'].replace('3,0.04,D','3,'+value)
            with self.subTest(value=value),self.assertRaises(ValueError): self.decode(members)

    def test_unsafe_path_and_camera_stabilization(self):
        members=self.fixture(); members['frames.csv']=members['frames.csv'].replace('frames/3.jpg','../outside.jpg')
        with self.assertRaises(ValueError): self.decode(members)
        members=self.fixture(); meta=json.loads(members['episode.json']); meta['camera']['ois']='on'
        members['episode.json']=json.dumps(meta)
        with self.assertRaises(ValueError): self.decode(members)

    def test_calibration_and_policy_required(self):
        for overrides in [dict(calibration_id=''),dict(warmup_s=0),dict(tracking_valid_values=None),
                          dict(t_cam_to_pinch_axes='+X right, +Y down, +Z forward (OpenCV)'),
                          dict(t_cam_to_pinch=np.zeros((4,4))),dict(usable_segments=[[0,99]])]:
            with self.subTest(overrides=overrides),self.assertRaises(ValueError): self.decode(self.fixture(),**overrides)

        raw=self.decode(self.fixture(),warmup_s=0,pre_stabilized=True)
        self.assertEqual(raw.meta.n_steps,8)
        self.assertTrue(raw.meta.notes['pre_stabilized'])
        with self.assertRaises(ValueError):
            self.decode(self.fixture(),warmup_s=0,pre_stabilized='yes')

    def test_velocity_and_acceleration_use_actual_dt(self):
        arm=np.zeros((4,5)); arm[:,0]=[0,.1,.2,.5]
        limits=TrajectoryLimits((1,)*6,(2,)*6,(.5,)*6)
        keep,reasons=continuous_mask(arm,np.zeros(4),[0,1,2,2.1],[True]*4,limits)
        self.assertFalse(keep[-1]); self.assertEqual(reasons['trajectory_velocity'],1)
        limits=TrajectoryLimits((1,)*6,(10,)*6,(.5,)*6)
        keep,reasons=continuous_mask(arm,np.zeros(4),[0,1,2,2.1],[True]*4,limits)
        self.assertFalse(keep[-1]); self.assertEqual(reasons['trajectory_acceleration'],1)

    def test_gap_is_metres_not_hinge_radians(self):
        limits=TrajectoryLimits((1,)*5+(.01,),(10,)*6,(10,)*6)
        keep,reasons=continuous_mask(np.zeros((3,5)),[.01,.04,.04],[0,1,2],[True]*3,limits)
        self.assertEqual(keep.tolist(),[True,False,True]); self.assertEqual(reasons['trajectory_step'],1)

    def test_tool_hash_covers_conversion_and_is_newline_stable(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); (root/'umi').mkdir(); (root/'tools').mkdir()
            p=root/'umi/convert.py'; p.write_bytes(b'a\n'); tool=root/'tools/convert_umi.py'; tool.write_bytes(b'b\n')
            first=conversion_digest(root); p.write_bytes(b'a\r\n')
            self.assertEqual(first,conversion_digest(root))
            p.write_bytes(b'changed\n'); self.assertNotEqual(first,conversion_digest(root))
            second=conversion_digest(root); tool.write_bytes(b'changed\n')
            self.assertNotEqual(second,conversion_digest(root))


if __name__=='__main__':
    if is_shared_gpu_server(): raise SystemExit('Local-only checks')
    unittest.main(verbosity=2)
