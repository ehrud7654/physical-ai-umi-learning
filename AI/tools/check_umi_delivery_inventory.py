"""Small, format-neutral fixtures for the read-only UMI delivery inventory."""

from __future__ import annotations

import hashlib
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parent))

from inspect_umi_delivery import IntakeError, inspect_delivery


class DeliveryInventoryTests(unittest.TestCase):
    def test_zip_inventory_is_a_fingerprint_not_a_quality_pass(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'episode.zip'
            with zipfile.ZipFile(path, 'w') as archive:
                archive.writestr('episode/metadata.json', '{"schema":"unconfirmed"}')
                archive.writestr('episode/poses.csv', 'timestamp,tx\n1,0\n2,1\n')
                archive.writestr('episode/frames/0001.jpg', b'jpeg fixture')
            report = inspect_delivery(path)
            self.assertEqual(report['status'], 'INVENTORY_ONLY_SCHEMA_UNVERIFIED')
            self.assertFalse(report['schema_verified'])
            self.assertFalse(report['quality_verified'])
            self.assertEqual(report['file_count'], 3)
            self.assertEqual(report['artifact_sha256'],
                             hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertEqual(report['extension_counts'],
                             {'.csv': 1, '.jpg': 1, '.json': 1})
            self.assertIn({'path': 'episode/poses.csv',
                           'csv_columns': ['timestamp', 'tx']},
                          report['metadata_structure'])

    def test_directory_and_tar_inventory_have_same_member_digests(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            episode = root / 'episode'
            episode.mkdir()
            (episode / 'frame.jpg').write_bytes(b'frame')
            (episode / 'pose.json').write_text('{"T":"unknown"}',
                                               encoding='utf-8')
            tar_path = root / 'episode.tar'
            with tarfile.open(tar_path, 'w') as archive:
                archive.add(episode / 'frame.jpg', arcname='frame.jpg')
                archive.add(episode / 'pose.json', arcname='pose.json')
            directory = inspect_delivery(episode)
            packed = inspect_delivery(tar_path)
            self.assertEqual(directory['artifact_type'], 'directory')
            self.assertEqual(packed['artifact_type'], 'tar')
            self.assertEqual(directory['files'], packed['files'])
            self.assertEqual(directory['metadata_structure'][0]['json_top_keys'],
                             ['T'])

    def test_unsafe_and_duplicate_archive_members_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ('../outside.json', '/absolute.json', 'C:/drive.json'):
                path = root / 'unsafe.zip'
                with zipfile.ZipFile(path, 'w') as archive:
                    archive.writestr(name, '{}')
                with self.subTest(name=name), self.assertRaises(IntakeError):
                    inspect_delivery(path)
            path = root / 'duplicate.zip'
            with zipfile.ZipFile(path, 'w') as archive:
                archive.writestr('./metadata.json', '{}')
                archive.writestr('metadata.json', '{}')
            with self.assertRaisesRegex(IntakeError, 'duplicate'):
                inspect_delivery(path)

    def test_tar_symlink_rejected_without_extraction(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'episode.tar'
            with tarfile.open(path, 'w') as archive:
                link = tarfile.TarInfo('linked_pose.csv')
                link.type = tarfile.SYMTYPE
                link.linkname = '../outside.csv'
                archive.addfile(link)
            with self.assertRaisesRegex(IntakeError, 'non-regular'):
                inspect_delivery(path)


if __name__ == '__main__':
    unittest.main()
