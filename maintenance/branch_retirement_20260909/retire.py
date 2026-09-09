"""One-shot, user-authorized branch retirement; no experiments or training here.

Test jobs run separately with read-only credentials. This job creates and verifies
annotated recovery tags first, then performs one atomic, exact-lease ref update.
Any concurrent movement of a retired/updated branch rejects the entire update.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
import subprocess
import urllib.request

REPO = 'LUNCY579/RineyaBeta'
CONTROL = 'maintenance/branch-retirement-20260909'
EXPECTED = {
    'main': '3532c7bc8dbdd48ef8801ba6080cb80b0e90379a',
    'research/cornerstone2-v1': '1cc2d8e2f53d937ae74641b3851b1f80b05518cd',
    'research/query-refinement-attention-v1': 'e5794507c4e4236068b3124ce6efe61ed8dc9e56',
    'research/query-refinement-official-family-v1': '5500e875c5fa80c8e00b831702a020e57cc1145d',
    'research/composition-qrr-v1': '690dcf27ee48512369fc2f3ca02603dee06ddfba',
    'research/composition-qrr-official-family-v1': 'ffb0e4be250b684aada0bd4e82bb7a8a51fa0635',
    'research/hybrid-local-global-attention-v1': 'a196318ec75cacccc8ac9ec446e4c4b6a29c2802',
    'agent/hlga-continuation-20260908': '2b1131ea5b94d7ffe29225c15a51b74060bbb42c',
}
CANDIDATE = '5785ef1af91359547da62136a7326e23b9183b57'
TARGETS = {
    'query': EXPECTED['research/query-refinement-official-family-v1'],
    'composition': CANDIDATE,
    'hybrid': EXPECTED['agent/hlga-continuation-20260908'],
}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def git(*args):
    # No shell, credentials, broad --force, or guessed branch names.
    return subprocess.run(['git', *args], check=True, text=True, capture_output=True).stdout.strip()


def snapshot():
    return {ref: sha for line in git('ls-remote', '--heads', '--tags', 'origin').splitlines()
            for sha, ref in [line.split('\t')]}


def branch_ref(name):
    return 'refs/heads/' + name


def plan(before, gates, control_sha):
    for name, sha in {**EXPECTED, CONTROL: control_sha}.items():
        require(before.get(branch_ref(name)) == sha, f'Branch changed or absent: {name}; stop and re-audit')
    changes, tags = [], {}
    if gates.get('query'):
        old = 'research/query-refinement-attention-v1'
        changes.append((branch_ref(old), EXPECTED[old], None))
        tags['retired/' + old] = EXPECTED[old]
    if gates.get('composition'):
        new = 'research/composition-qrr-official-family-v1'
        old = 'research/composition-qrr-v1'
        changes.extend([(branch_ref(new), EXPECTED[new], CANDIDATE),
                        (branch_ref(old), EXPECTED[old], None)])
        tags['retired/' + old] = EXPECTED[old]
    if gates.get('hybrid'):
        canonical = 'research/hybrid-local-global-attention-v1'
        continuation = 'agent/hlga-continuation-20260908'
        changes.extend([(branch_ref(canonical), EXPECTED[canonical], EXPECTED[continuation]),
                        (branch_ref(continuation), EXPECTED[continuation], None)])
        tags['retired/research/hybrid-local-global-attention-v1-before-20260909-consolidation'] = EXPECTED[canonical]
        tags['retired/' + continuation] = EXPECTED[continuation]
    # The temporary maintenance branch is itself recoverable and removed last in
    # the same transaction; it does not become permanent branch-list clutter.
    tags['retired/' + CONTROL] = control_sha
    changes.append((branch_ref(CONTROL), control_sha, None))
    for tag in tags:
        require('refs/tags/' + tag not in before, f'Archive tag already exists: {tag}; never overwrite it')
    require('refs/tags/retired/audit/branch-retirement-20260909' not in before, 'Audit tag already exists')
    return changes, tags


def push_args(changes):
    # Deletions and fast-forwards are both compare-and-swap guarded. Ancestor
    # checks are performed separately before building this atomic transaction.
    return ['push', '--atomic', *[f'--force-with-lease={ref}:{old}' for ref, old, _ in changes],
            'origin', *[f'{new or ""}:{ref}' for ref, _, new in changes]]


def expected_after(before, changes, tag_objects):
    after = dict(before)
    for tag, (obj, commit) in tag_objects.items():
        after['refs/tags/' + tag] = obj
        after['refs/tags/' + tag + '^{}'] = commit
    for ref, _, new in changes:
        if new is None:
            after.pop(ref)
        else:
            after[ref] = new
    return after


def ensure_ancestor(old, new):
    subprocess.run(['git', 'merge-base', '--is-ancestor', old, new], check=True,
                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def archive(name, commit, message):
    git('tag', '-a', name, commit, '-m', message)
    return git('rev-parse', 'refs/tags/' + name), commit


def open_pr_guard(affected):
    token = os.environ['GH_TOKEN']
    page = 1
    while True:
        req = urllib.request.Request(f'https://api.github.com/repos/{REPO}/pulls?state=open&per_page=100&page={page}',
            headers={'Authorization': 'Bearer ' + token, 'Accept': 'application/vnd.github+json',
                     'X-GitHub-Api-Version': '2022-11-28'})
        with urllib.request.urlopen(req, timeout=30) as response:
            prs = json.load(response)
        require(isinstance(prs, list), 'Unexpected pull request response')
        for pr in prs:
            for side in ('head', 'base'):
                repo = pr[side].get('repo') or {}
                require(not (repo.get('full_name') == REPO and pr[side]['ref'] in affected),
                        f'Open PR #{pr["number"]} depends on a changing branch; stop for review')
        if len(prs) < 100:
            return
        page += 1


def gate_receipts(control_sha):
    results, details = {}, {}
    for name, sha in TARGETS.items():
        path = Path('gate-artifacts') / name / 'gate.json'
        if not path.exists():
            results[name] = False
            details[name] = {'passed': False, 'reason': 'No gate artifact; branch retained'}
            continue
        receipt = json.loads(path.read_text())
        require(receipt['kind'] == name and receipt['tested_commit'] == sha, 'Wrong checkpoint-independent test source')
        require(receipt['workflow_sha'] == control_sha and receipt['run_id'] == os.environ['GITHUB_RUN_ID'],
                'Gate artifact belongs to another run/source')
        require(receipt['pretrained_model_executed'] is False and receipt['licenses_training'] is False,
                'Unexpected evidence or training scope')
        details[name] = receipt
        results[name] = receipt.get('passed') is True and receipt.get('source_clean') is True
    return results, details


def main():
    require(os.environ.get('GITHUB_REPOSITORY') == REPO, 'Wrong repository')
    require(os.environ.get('GITHUB_REF') == branch_ref(CONTROL), 'This one-shot workflow must run on its exact maintenance branch')
    control_sha = os.environ['GITHUB_SHA']
    require(git('rev-parse', 'HEAD') == control_sha, 'Maintenance checkout drift')
    require(not git('status', '--porcelain', '--untracked-files=no'), 'Modified maintenance source')
    gates, details = gate_receipts(control_sha)
    before = snapshot()
    changes, tags = plan(before, gates, control_sha)
    # Full history exists via checkout fetch-depth=0; fetch explicit objects too.
    for sha in set(EXPECTED.values()) | {CANDIDATE}:
        git('fetch', '--no-tags', '--filter=blob:none', 'origin', sha)
    if gates['query']:
        ensure_ancestor(EXPECTED['research/query-refinement-attention-v1'], TARGETS['query'])
    if gates['composition']:
        old = EXPECTED['research/composition-qrr-official-family-v1']
        ensure_ancestor(old, CANDIDATE)
        ensure_ancestor(EXPECTED['research/composition-qrr-v1'], CANDIDATE)
        path = '.github/workflows/official-family.yml'
        require(git('diff', '--name-only', old, CANDIDATE) == path, 'Composition repair changed more than CI discovery')
        previous, candidate = git('show', f'{old}:{path}'), git('show', f'{CANDIDATE}:{path}')
        needle = 'python -m pytest -c /dev/null -q '
        require(previous.count(needle) == 1 and candidate == previous.replace(needle,
                'python -m pytest -c /dev/null --import-mode=importlib -q ', 1), 'Unexpected composition fix')
    if gates['hybrid']:
        ensure_ancestor(EXPECTED['research/hybrid-local-global-attention-v1'], TARGETS['hybrid'])
        protocol = json.loads(git('show', TARGETS['hybrid'] + ':experiments/hybrid_attention_v1/protocol.json'))
        require(all(protocol.get(key) is False for key in
                ('licenses_training', 'licenses_sealed_evaluation', 'established_quality_improvement')),
                'Hybrid continuation unexpectedly opens training/claims')
    for _, old, new in changes:
        if new:
            ensure_ancestor(old, new)
    open_pr_guard({ref.removeprefix('refs/heads/') for ref, _, _ in changes})
    require(snapshot() == before, 'Remote changed during preflight; stop and re-audit')
    git('config', 'user.name', 'github-actions[bot]')
    git('config', 'user.email', '41898282+github-actions[bot]@users.noreply.github.com')
    run_url = f'https://github.com/{REPO}/actions/runs/{os.environ["GITHUB_RUN_ID"]}'
    report = {'schema': 'rineya.branch-retirement/1', 'status': 'tags_pending', 'repository': REPO,
              'run_url': run_url, 'maintenance_commit': control_sha, 'gates': details,
              'before': before, 'changes': changes, 'archive_targets': tags,
              'training_authorized': False, 'quality_claim_authorized': False,
              'note': 'Branch organization and mechanical checks only. Hybrid protocol amendments and historical evidence are preserved, not newly approved. No pretrained execution.'}
    Path('retirement-report.json').write_text(json.dumps(report, indent=2) + '\n')
    objects = {}
    for tag, sha in tags.items():
        text = f'User-authorized archival reference, 2026-09-09.\nOriginal commit: {sha}\nWorkflow: {run_url}\n'
        text += 'No commits discarded; no training, performance claim or scientific-gate approval.\n'
        text += 'Gate outcomes: ' + json.dumps(gates, sort_keys=True)
        objects[tag] = archive(tag, sha, text)
    # First push creates recovery tags only. No branch is deleted before these
    # remote tag objects and their peeled commit IDs have been verified.
    git('push', '--atomic', 'origin', *[f'refs/tags/{tag}:refs/tags/{tag}' for tag in tags])
    with_tags = expected_after(before, [], objects)
    require(snapshot() == with_tags, 'Tag verification failed or concurrent change; no branches deleted')
    report.update(status='archives_verified', archive_objects=objects)
    Path('retirement-report.json').write_text(json.dumps(report, indent=2) + '\n')
    # Dry run plus real atomic push; exact leases protect against the intervening race.
    arguments = push_args(changes)
    git(*arguments[:2], '--dry-run', *arguments[2:])
    git(*arguments)
    after = snapshot()
    require(after == expected_after(before, changes, objects), 'Post-transaction remote differs; inspect preserved tags')
    report.update(status='complete', after=after,
                  live_branch_count=sum(ref.startswith('refs/heads/') for ref in after))
    Path('retirement-report.json').write_text(json.dumps(report, indent=2) + '\n')
    # Persist the full execution result in an annotated audit tag as well as the
    # downloadable workflow artifact. Ordinary artifact expiry loses no audit.
    audit_name = 'retired/audit/branch-retirement-20260909'
    obj, sha = archive(audit_name, control_sha, json.dumps(report, indent=2, sort_keys=True))
    git('push', 'origin', f'refs/tags/{audit_name}:refs/tags/{audit_name}')
    final = snapshot()
    expected = dict(after)
    expected['refs/tags/' + audit_name] = obj
    expected['refs/tags/' + audit_name + '^{}'] = sha
    require(final == expected, 'Audit-tag verification drift')
    print(json.dumps({'status': 'complete', 'live_branch_count': report['live_branch_count'],
                      'gates': gates, 'changes': changes, 'audit_tag': audit_name}, indent=2))
    if os.environ.get('GITHUB_STEP_SUMMARY'):
        with open(os.environ['GITHUB_STEP_SUMMARY'], 'a') as stream:
            stream.write('## Verified branch retirement\n\n```json\n' + json.dumps(report, indent=2) + '\n```\n')


if __name__ == '__main__':
    main()
