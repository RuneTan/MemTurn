"""Failure and restart tests for reversible orphan checkpoint cleanup."""
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from memturn.config import Config
from memturn.learning.archive import archive_orphan, inventory, restore_archive, verify_archive
from memturn.learning.huggingface import HuggingFaceLearner, SFTConfig
from memturn.memory.store import Store
from memturn.runtime.service import Runtime
from memturn.tokens import OfflineCounter


class AdapterArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / 'adapters'
        self.learner = HuggingFaceLearner(self.root, SFTConfig('model', 'revision', ('q_proj',)), None)

    def tearDown(self):
        self.temp.cleanup()

    def checkpoint(self, digit='a'):
        directory = self.root / ('hf_' + digit * 32)
        directory.mkdir()
        (directory / 'adapter_model.safetensors').write_bytes(b'checkpoint bytes')
        (directory / 'manifest.json').write_text('{"incomplete":true}', encoding='utf-8')
        (directory / 'empty').mkdir()
        return directory

    def test_recovery_frees_capacity_protects_live_and_restores_exact_bytes(self):
        current, retired, orphan = (self.checkpoint(d) for d in 'abc')
        before = inventory(orphan)
        self.assertFalse(self.learner.has_artifact_capacity(3))
        events = self.learner.recover_orphans({current.name, retired.name})
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]['state'], 'archived')
        self.assertTrue(current.exists() and retired.exists())
        self.assertFalse(orphan.exists())
        self.assertTrue(self.learner.has_artifact_capacity(3))
        archive = Path(events[0]['archive'])
        self.assertFalse(archive.is_relative_to(self.root))
        self.assertEqual(verify_archive(archive)['inventory'], before)
        restored = restore_archive(archive, Path(self.temp.name) / 'manual_restore')
        self.assertEqual(inventory(restored), before)
        self.assertEqual(self.learner.recover_orphans({current.name, retired.name}), [])

    def test_registered_and_unowned_directories_are_preserved(self):
        owned = self.checkpoint()
        self.learner.artifacts[owned.name] = object()
        (self.root / 'unknown_checkpoint').mkdir()
        self.assertEqual(self.learner.recover_orphans({'base'}), [])
        self.assertTrue(owned.exists())
        self.assertTrue((self.root / 'unknown_checkpoint').exists())

    def test_copy_failure_keeps_original(self):
        source = self.checkpoint()
        before = inventory(source)
        with patch('memturn.learning.archive.shutil.copytree', side_effect=OSError('disk full')):
            events = self.learner.recover_orphans({'base'})
        self.assertEqual(events[0]['state'], 'recovery_failed')
        self.assertEqual(inventory(source), before)

    def test_corrupt_copy_cannot_trigger_deletion(self):
        source = self.checkpoint()
        before, real_copy = inventory(source), shutil.copytree
        def corrupt(src, dst, *args, **kwargs):
            result = real_copy(src, dst, *args, **kwargs)
            if Path(dst).name == 'payload':
                (Path(dst) / 'adapter_model.safetensors').write_bytes(b'corrupt')
            return result
        with patch('memturn.learning.archive.shutil.copytree', side_effect=corrupt):
            events = self.learner.recover_orphans({'base'})
        self.assertEqual(events[0]['state'], 'recovery_failed')
        self.assertEqual(inventory(source), before)

    def test_interrupted_cleanup_retains_complete_recoverable_archive(self):
        source = self.checkpoint()
        before, unlink = inventory(source), Path.unlink
        def interrupt(path, *args, **kwargs):
            if path == source / 'manifest.json':
                raise OSError('interrupted cleanup')
            return unlink(path, *args, **kwargs)
        with patch.object(Path, 'unlink', interrupt):
            with self.assertRaisesRegex(OSError, 'interrupted cleanup'):
                archive_orphan(source, self.root)
        archives = list((self.root.parent / 'research_archive' / 'orphan_adapters').glob('hf_*'))
        self.assertEqual(len(archives), 1)
        self.assertFalse((source / 'adapter_model.safetensors').exists())
        self.assertEqual(verify_archive(archives[0])['inventory'], before)
        restored = restore_archive(archives[0], self.root.parent / 'restored')
        self.assertEqual(inventory(restored), before)
        # A retry archives the leftover subset without replacing the full copy.
        self.learner.recover_orphans({'base'})
        self.assertFalse(source.exists())
        self.assertEqual(verify_archive(archives[0])['inventory'], before)

    def test_changed_source_is_preserved(self):
        source = self.checkpoint()
        real_copy = shutil.copytree
        def change_source(src, dst, *args, **kwargs):
            result = real_copy(src, dst, *args, **kwargs)
            if Path(dst).name == 'payload':
                (source / 'manifest.json').write_text('new contents', encoding='utf-8')
            return result
        with patch('memturn.learning.archive.shutil.copytree', side_effect=change_source):
            events = self.learner.recover_orphans({'base'})
        self.assertEqual(events[0]['state'], 'recovery_failed')
        self.assertTrue((source / 'adapter_model.safetensors').exists())
        self.assertEqual((source / 'manifest.json').read_text(encoding='utf-8'), 'new contents')

    def test_unreadable_tree_is_not_treated_as_empty(self):
        source = self.checkpoint()
        def unreadable(directory, *, followlinks, onerror):
            onerror(PermissionError('unreadable directory'))
            return iter(())
        with patch('memturn.learning.archive.os.walk', side_effect=unreadable):
            with self.assertRaises(PermissionError):
                archive_orphan(source, self.root)
        self.assertTrue((source / 'adapter_model.safetensors').exists())

    def test_restore_refuses_overwrite_and_modified_archive(self):
        source = self.checkpoint()
        archived = archive_orphan(source, self.root)
        with self.assertRaises(FileExistsError):
            restore_archive(archived, self.root)
        (archived / 'payload' / 'manifest.json').write_text('corrupted', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'checksum'):
            restore_archive(archived, self.root.parent / 'restore')

    def test_outside_root_cannot_be_cleaned(self):
        outside = self.root.parent / 'outside'
        outside.mkdir()
        (outside / 'data').write_bytes(b'preserve')
        with self.assertRaisesRegex(ValueError, 'outside artifact root'):
            archive_orphan(outside, self.root)
        self.assertEqual((outside / 'data').read_bytes(), b'preserve')

    def test_link_is_never_followed(self):
        source = self.checkpoint()
        outside = self.root.parent / 'outside'
        outside.mkdir()
        try:
            (source / 'linked').symlink_to(outside, target_is_directory=True)
        except OSError:
            # Windows may deny creating symlinks; inject the equivalent lstat bit.
            import stat
            from types import SimpleNamespace
            from memturn.learning.archive import _plain
            with patch.object(Path, 'lstat', return_value=SimpleNamespace(
                    st_mode=stat.S_IFDIR, st_file_attributes=0x400)):
                with self.assertRaisesRegex(ValueError, 'reparse'):
                    _plain(source, directory=True)
        else:
            with self.assertRaisesRegex(ValueError, 'links'):
                archive_orphan(source, self.root)
        self.assertTrue(source.exists() and outside.exists())

    def test_runtime_recovers_only_in_write_mode_and_protects_published_version(self):
        calls = []
        class Learner:
            def recover_orphans(self, versions):
                calls.append(versions)
                return []
            def retain(self, versions): pass
        store = Store(self.root.parent / 'state.sqlite', Config(), OfflineCounter())
        try:
            readonly = Runtime(store, None, None, None, Learner())
            readonly.close()
            self.assertEqual(calls, [])
            runtime = Runtime(store, object(), object(), object(), Learner())
            runtime.close()
            self.assertTrue(calls and all(live == {'base'} for live in calls))
        finally:
            store.close()


if __name__ == '__main__':
    unittest.main()
