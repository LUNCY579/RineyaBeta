"""Persist the completed retirement report, then remove only this temporary branch.

The research refs are read-only. The first workflow completed its ref transaction
but failed to publish the extra audit tag afterwards. Do not rerun retirement.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess

REPO = 'LUNCY579/RineyaBeta'
BRANCH = 'maintenance/branch-retirement-audit-20260909'
REPORT_HASH = '577f2272b9cc3bdbec1e2fd37a893dcdab60db7f934a9ec13d3a9b7118abca47'


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def git(*args):
    result = subprocess.run(['git', *args], text=True, capture_output=True)
    if result.returncode:
        # Git commands here never contain a credential; checkout handles auth.
        raise RuntimeError(f'Git {args[0]} failed: {result.stderr}')
    return result.stdout.strip()


def snapshot():
    return {ref: sha for line in git('ls-remote', '--heads', '--tags', 'origin').splitlines()
            for sha, ref in [line.split('\t')]}


def main():
    ref, initial = 'refs/heads/' + BRANCH, os.environ['GITHUB_SHA']
    require(os.environ['GITHUB_REPOSITORY'] == REPO and os.environ['GITHUB_REF'] == ref, 'Wrong execution scope')
    require(git('rev-parse', 'HEAD') == initial, 'Unexpected checkout')
    require(not git('status', '--porcelain', '--untracked-files=no'), 'Modified source')
    raw = Path('audit-input/retirement-report.json').read_bytes()
    require(hashlib.sha256(raw).hexdigest() == REPORT_HASH, 'Original report artifact hash differs')
    report = json.loads(raw)
    require(report['status'] == 'complete' and report['live_branch_count'] == 5, 'Retirement did not finish')
    require(all(g['passed'] is True and g['source_clean'] is True and g['licenses_training'] is False
                for g in report['gates'].values()), 'Original mechanical gate failed')
    before = snapshot()
    require(before == {**report['after'], ref: initial}, 'Remote differs from verified five-branch state; stop')
    tags = ['retired/audit/branch-retirement-20260909', 'retired/' + BRANCH]
    require(all('refs/tags/' + t not in before for t in tags), 'Never overwrite an audit or archive tag')
    root = Path('maintenance/branch_retirement_20260909')
    (root / 'EXECUTION_REPORT.json').write_bytes(raw)
    rows = '\n'.join(f'| {name} | {g["tested_commit"][:7]} | {g["passed_tests"]} | {g["skipped"]} |'
                     for name, g in report['gates'].items())
    text = '# Completed RineyaBeta branch retirement — 2026-09-09\n\n'
    text += 'The requested branch transaction completed with five live branches and verified annotated recovery tags.\n\n'
    text += '| Mechanical gate | Exact tested head | Passed | Skipped |\n|---|---|---:|---:|\n' + rows + '\n\n'
    text += 'Original execution: ' + report['run_url'] + '\n\n'
    text += ('The original workflow ended in failure only when pushing its additional audit tag, after its '\
             'atomic branch transaction and post-transaction verification had completed. This separate audit-only '\
             'workflow persists the exact downloaded report (SHA-256 ' + REPORT_HASH + '), creates and verifies '\
             'its archival tags before deleting its own temporary branch, and does not update any research branch.\n\n')
    text += ('No pretrained checkpoint was executed. No training, sealed evaluation, quality improvement or '\
             'approval of historical hybrid protocol amendments is implied. Main, Cornerstone2 and the '\
             'eleven preexisting retired tags were unchanged by the retirement transaction.\n')
    (root / 'EXECUTION_SUMMARY.md').write_text(text)
    git('config', 'user.name', 'github-actions[bot]')
    git('config', 'user.email', '41898282+github-actions[bot]@users.noreply.github.com')
    git('add', '--', str(root / 'EXECUTION_REPORT.json'), str(root / 'EXECUTION_SUMMARY.md'))
    git('commit', '-m', 'docs: preserve verified retirement result and mechanical gate receipts')
    final_commit = git('rev-parse', 'HEAD')
    git('push', f'--force-with-lease={ref}:{initial}', 'origin', f'{final_commit}:{ref}')
    expected = {**before, ref: final_commit}
    require(snapshot() == expected, 'Audit branch update drift')
    for tag in tags:
        message = ('Verified branch-retirement audit, 2026-09-09.\n'
                   'Five live research/baseline branches; original heads retained in annotated recovery tags.\n'
                   'Full report and passing mechanical counts: maintenance/branch_retirement_20260909/EXECUTION_REPORT.json\n'
                   'Original run: ' + report['run_url'] + '\nNo pretrained run, training or scientific-gate promotion.\n')
        git('tag', '-a', tag, final_commit, '-m', message)
        expected['refs/tags/' + tag] = git('rev-parse', 'refs/tags/' + tag)
        expected['refs/tags/' + tag + '^{}'] = final_commit
    git('push', '--atomic', 'origin', *[f'refs/tags/{tag}:refs/tags/{tag}' for tag in tags])
    require(snapshot() == expected, 'Audit tags were not preserved exactly; temporary branch retained')
    # LAST write: all evidence and recovery refs already exist and were verified.
    git('push', '--atomic', f'--force-with-lease={ref}:{final_commit}', 'origin', ':' + ref)
    expected.pop(ref)
    require(snapshot() == expected, 'Unexpected final refs after temporary-branch cleanup')
    result = {'status': 'complete', 'live_branches': 5, 'audit_commit': final_commit,
              'audit_tag': tags[0], 'original_report_sha256': REPORT_HASH,
              'research_refs_changed_by_this_finalizer': False}
    Path('audit-finalization-result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
