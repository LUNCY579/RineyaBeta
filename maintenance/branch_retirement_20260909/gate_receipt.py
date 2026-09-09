"""Record a commit-bound mechanical CI gate, including failure, never model efficacy."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

kind, expected = sys.argv[1:]
head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
dirty = subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], text=True).strip()
rc_path = Path('gate-exit-code.txt')
rc = int(rc_path.read_text()) if rc_path.exists() else None
result = {'schema': 'rineya.retirement-mechanical-gate/1', 'kind': kind,
          'tested_commit': head, 'expected_commit': expected, 'source_clean': not dirty,
          'workflow_sha': os.environ['GITHUB_SHA'], 'run_id': os.environ['GITHUB_RUN_ID'],
          'exit_code': rc, 'passed': False, 'pretrained_model_executed': False,
          'licenses_training': False, 'quality_claim_authorized': False}
try:
    if head != expected or dirty or rc != 0:
        raise ValueError('Missing/failed test execution or source drift')
    if kind == 'hybrid':
        source = Path('experiments/hybrid_attention_v1/runs/branch-retirement-native/receipt.json')
        raw = json.loads(source.read_text())
        if (raw.get('passed') is not True or raw.get('source_unchanged') is not True
                or raw.get('native_hf_available') is not True
                or raw.get('real_rineya_checkpoint_loaded') is not False
                or raw.get('licenses_training') is not False or raw.get('skipped')
                or raw.get('failures') or raw.get('errors') or raw.get('tests_run', 0) < 1):
            raise ValueError('Native hybrid qualification failed, skipped, or has wrong scope')
        result.update(tests=raw['tests_run'], passed_tests=raw['tests_passed'], skipped=0)
    else:
        source = Path('family-tests.xml')
        tree = ET.parse(source).getroot()
        cases = list(tree.iter('testcase'))
        failures = [c for c in cases if c.find('failure') is not None or c.find('error') is not None]
        skipped = [c for c in cases if c.find('skipped') is not None]
        names = {c.attrib['name'] for c in cases if c not in skipped and c not in failures}
        required = {'test_native_safe_roundtrip_and_tied_head', 'test_native_missing_parameter_is_not_accepted',
                    'test_query_native_integration_on_its_branch' if kind == 'query'
                    else 'test_frozen_native_integration_on_its_branch'}
        if failures or len(cases) < 100 or not required.issubset(names):
            raise ValueError('Incomplete/failed inherited-plus-family suite or missing native tests')
        result.update(tests=len(cases), passed_tests=len(cases)-len(skipped), skipped=len(skipped),
                      required_native_tests_passed=sorted(required))
    result.update(passed=True, source_receipt_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
except Exception as exc:
    result['error'] = f'{type(exc).__name__}: {exc}'
Path('gate.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
if not result['passed']:
    sys.exit(1)
