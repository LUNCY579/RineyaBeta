import importlib.util
from pathlib import Path
import subprocess
import tempfile
import unittest
spec = importlib.util.spec_from_file_location('retire', Path(__file__).with_name('retire.py'))
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)

class PlanTests(unittest.TestCase):
    def initial(self):
        return {r.branch_ref(k): v for k, v in {**r.EXPECTED, r.CONTROL: 'a' * 40}.items()}
    def test_all_pass_changes_only_recommended_refs(self):
        initial = self.initial()
        changes, tags = r.plan(initial, dict.fromkeys(r.TARGETS, True), 'a' * 40)
        self.assertEqual(len(changes), 6)
        self.assertEqual(len(tags), 5)
        touched = {x[0] for x in changes}
        self.assertNotIn('refs/heads/main', touched)
        self.assertNotIn('refs/heads/research/cornerstone2-v1', touched)
        self.assertNotIn('refs/heads/research/query-refinement-official-family-v1', touched)
        result = r.expected_after(initial, changes, {})
        self.assertEqual(sum(k.startswith('refs/heads/') for k in result), 5)
    def test_failed_experiment_gates_retain_both_branches(self):
        changes, _ = r.plan(self.initial(), {'query': True, 'composition': False, 'hybrid': False}, 'a' * 40)
        self.assertEqual(len(changes), 2)  # query predecessor and temporary maintenance only
        touched = {row[0] for row in changes}
        self.assertFalse(any('composition' in name or 'hybrid' in name or '/agent/' in name for name in touched))
    def test_concurrent_ref_movement_aborts_plan(self):
        for name in r.EXPECTED:
            state = self.initial()
            state[r.branch_ref(name)] = 'b' * 40
            with self.assertRaises(RuntimeError):
                r.plan(state, dict.fromkeys(r.TARGETS, True), 'a' * 40)
    def test_existing_archive_is_never_overwritten(self):
        state = self.initial()
        state['refs/tags/retired/research/query-refinement-attention-v1'] = 'c' * 40
        with self.assertRaises(RuntimeError):
            r.plan(state, {'query': True}, 'a' * 40)
    def test_every_mutation_has_exact_lease(self):
        changes, _ = r.plan(self.initial(), dict.fromkeys(r.TARGETS, True), 'a' * 40)
        args = r.push_args(changes)
        self.assertIn('--atomic', args)
        self.assertNotIn('--force', args)
        for ref, old, new in changes:
            self.assertIn(f'--force-with-lease={ref}:{old}', args)
            self.assertIn(f'{new or ""}:{ref}', args)

class GitTransactionTests(unittest.TestCase):
    def test_atomic_retire_and_race_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            remote, work = root / 'remote.git', root / 'work'
            def git(*args, okay=True):
                proc = subprocess.run(['git', *args], cwd=work if work.exists() else root,
                                      text=True, capture_output=True)
                if okay:
                    self.assertEqual(proc.returncode, 0, proc.stderr)
                return proc
            git('init', '--bare', str(remote))
            git('init', str(work))
            git('config', 'user.name', 'test')
            git('config', 'user.email', 'test@example.invalid')
            git('commit', '--allow-empty', '-m', 'old')
            old = git('rev-parse', 'HEAD').stdout.strip()
            git('commit', '--allow-empty', '-m', 'new')
            new = git('rev-parse', 'HEAD').stdout.strip()
            git('remote', 'add', 'origin', str(remote))
            git('push', 'origin', f'{old}:refs/heads/old', f'{old}:refs/heads/canonical')
            git('tag', '-a', 'retired/old', old, '-m', 'preserved')
            git('push', 'origin', 'refs/tags/retired/old')
            changes = [('refs/heads/old', old, None), ('refs/heads/canonical', old, new)]
            # A competing update arrives after the preflight and archive creation.
            git('push', 'origin', f'{new}:refs/heads/old')
            outcome = git(*r.push_args(changes), okay=False)
            self.assertNotEqual(outcome.returncode, 0)
            refs = git('ls-remote', '--heads', 'origin').stdout
            self.assertIn(new + '\trefs/heads/old', refs)
            self.assertIn(old + '\trefs/heads/canonical', refs)  # atomic: no partial FF
            # Fresh reviewed expected value allows the normal transaction.
            changes[0] = ('refs/heads/old', new, None)
            git('tag', '-a', 'retired/new-tip', new, '-m', 'competing work preserved')
            git('push', 'origin', 'refs/tags/retired/new-tip')
            git(*r.push_args(changes))
            refs = git('ls-remote', '--heads', '--tags', 'origin').stdout
            self.assertNotIn('\trefs/heads/old\n', refs)
            self.assertIn(new + '\trefs/heads/canonical', refs)
            self.assertIn(old + '\trefs/tags/retired/old^{}', refs)
            self.assertIn(new + '\trefs/tags/retired/new-tip^{}', refs)

if __name__ == '__main__':
    unittest.main(verbosity=2)
