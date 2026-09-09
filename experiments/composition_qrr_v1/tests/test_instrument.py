import copy
import tempfile
import unittest
from pathlib import Path
from experiments.composition_qrr_v1.identity import (ContractError, digest, inventory,
    read_json, safe_run_dir, write_new)
from experiments.composition_qrr_v1.instrument import (ARMS, OPS, audit_bank, build_bank,
    make_world, prompt_for, reference_solve, score, solve)
from experiments.composition_qrr_v1.analysis import compare
from experiments.composition_qrr_v1.bias import inspect_trace
from experiments.composition_qrr_v1.native import validate_contract, check_receipt


class InstrumentTests(unittest.TestCase):
    def setUp(self):
        self.bank = build_bank(3)
        self.case = self.bank['cases'][0]

    def test_oracles_across_200_worlds(self):
        for i in range(200):
            w = make_world(29, i)
            for op in OPS:
                self.assertEqual(solve(w, op), reference_solve(w, op))

    def test_bank_reproducible(self):
        self.assertEqual(self.bank, build_bank(3))
        self.assertEqual(audit_bank(self.bank)['cases'], 36)

    def test_distinct_seeds(self):
        self.assertNotEqual(self.bank['cases_sha256'], build_bank(3, seed=43)['cases_sha256'])

    def test_reject_changed_bank(self):
        self.bank['cases'][0]['gold'] = 'wrong'
        with self.assertRaises(ContractError): audit_bank(self.bank)

    def test_reject_rehashed_wrong_gold(self):
        self.bank['cases'][0]['gold'] = 'wrong'
        self.bank['cases_sha256'] = digest(self.bank['cases'])
        with self.assertRaises(ContractError): audit_bank(self.bank)

    def test_reject_incomplete_pack(self):
        self.bank['cases'].pop()
        self.bank['cases_sha256'] = digest(self.bank['cases'])
        with self.assertRaises(ContractError): audit_bank(self.bank)

    def test_reject_sealed_bank(self):
        self.bank['split'] = 'sealed'
        with self.assertRaises(ContractError): audit_bank(self.bank)

    def test_counterfactuals_change_answer(self):
        w = self.case['world']
        self.assertEqual(len({solve(w, op) for op in ('lookup', 'next', 'next2')}), 3)
        self.assertNotEqual(solve(w, 'count_red'), solve(w, 'count_blue'))

    def test_gold_does_not_enter_prompt(self):
        before = [prompt_for(self.case, arm) for arm in ARMS]
        self.case['gold'] = 'SENTINEL_LEAK'
        self.assertEqual(before, [prompt_for(self.case, arm) for arm in ARMS])

    def test_oracle_has_explicit_label(self):
        self.assertNotIn('Oracle diagnostic', prompt_for(self.case))
        self.assertIn('Oracle diagnostic', prompt_for(self.case, 'oracle_evidence'))

    def test_same_world_across_length_position(self):
        variants = [build_bank(3, padding_lines=10, placement=p) for p in ('start', 'middle', 'end')]
        for v in variants:
            audit_bank(v)
            self.assertEqual([c['cluster_id'] for c in v['cases']], [c['cluster_id'] for c in self.bank['cases']])
            self.assertEqual([c['gold'] for c in v['cases']], [c['gold'] for c in self.bank['cases']])
        self.assertNotEqual(variants[0]['cases'][0]['prompt_sha256'], variants[2]['cases'][0]['prompt_sha256'])

    def test_missing_code_not_constant(self):
        self.assertNotEqual(make_world(17, 1)['absent'], make_world(17, 2)['absent'])

    def test_format_gain_is_not_semantic_gain(self):
        plain = score(self.case, self.case['gold'])
        carrier = score(self.case, 'vault ' + self.case['gold'])
        self.assertTrue(plain['literal_exact'])
        self.assertFalse(carrier['literal_exact'])
        self.assertTrue(carrier['semantic_correct'])

    def test_false_abstention_is_error(self):
        r = score(self.case, 'UNKNOWN')
        self.assertFalse(r['semantic_correct'])
        self.assertEqual(r['error_category'], 'false_abstention')

    def test_wrong_cue_detection(self):
        c = next(c for c in self.bank['cases'] if c['op'] == 'next')
        r = score(c, solve(c['world'], 'lookup'))
        self.assertTrue(r['cue_substitution'])
        self.assertFalse(r['semantic_correct'])

    def test_substring_not_accepted(self):
        self.assertFalse(score(self.case, 'Maybe ' + self.case['gold'])['semantic_correct'])
        self.assertFalse(score(self.case, self.case['gold'] + ' or Vffffffff')['semantic_correct'])

    def test_boundaries(self):
        for args in ({'worlds': 0}, {'worlds': 65}, {'worlds': True}, {'padding_lines': -1}):
            with self.assertRaises(ContractError): build_bank(**args)


class IdentityTests(unittest.TestCase):
    def test_missing_checkpoint_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            with self.assertRaises(ContractError): inventory(t)

    def test_lfs_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t)
            for name in ('config.json', 'tokenizer.json'): (p / name).write_text('{}')
            (p / 'model.safetensors').write_text('version https://git-lfs.github.com/spec/v1\n')
            with self.assertRaises(ContractError): inventory(t)

    def test_inventory_changes_with_bytes(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t)
            for name in ('config.json', 'tokenizer.json'): (p / name).write_text('{}')
            (p / 'model.safetensors').write_bytes(b'x' * 128)
            old = inventory(t)
            (p / 'model.safetensors').write_bytes(b'y' * 128)
            self.assertNotEqual(old['identity'], inventory(t)['identity'])
            self.assertFalse(old['lineage_verified'])

    def test_whole_snapshot_mutation_rejected(self):
        from unittest.mock import patch
        from experiments.composition_qrr_v1 import identity
        with tempfile.TemporaryDirectory() as t:
            p = Path(t)
            for name in ('config.json', 'tokenizer.json'): (p / name).write_text('{}')
            (p / 'model.safetensors').write_bytes(b'x' * 128)
            real_hash = identity.hash_file
            def changing_hash(path):
                result = real_hash(path)
                if path.name == 'model.safetensors':
                    (p / 'config.json').write_text('{"changed":true}')
                return result
            with patch.object(identity, 'hash_file', changing_hash):
                with self.assertRaises(ContractError): inventory(t)

    def test_invalid_json_does_not_create_partial_record(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / 'a.json'
            with self.assertRaises(ValueError): write_new(p, {'a': float('nan')})
            self.assertFalse(p.exists())

    def test_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t)
            (p / 'real').write_text('data')
            (p / 'link').symlink_to(p / 'real')
            with self.assertRaises(ContractError): inventory(t)

    def test_duplicate_json_key_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / 'a.json'; p.write_text('{"a":1,"a":2}')
            with self.assertRaises(ContractError): read_json(p)

    def test_nonfinite_json_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / 'a.json'; p.write_text('{"a":NaN}')
            with self.assertRaises(ContractError): read_json(p)

    def test_no_overwrite(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / 'a.json'; write_new(p, {'a': 1})
            with self.assertRaises(FileExistsError): write_new(p, {'a': 2})

    def test_unsafe_run_id_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            for name in ('../escape', '/absolute', 'a/b', '.', ''):
                with self.assertRaises(ContractError): safe_run_dir(t, name)

    def test_native_contract_cannot_be_template(self):
        with self.assertRaises(ContractError): validate_contract({'schema': 'qrr.native-contract/1'})

    def test_arbitrary_receipt_rejected(self):
        with self.assertRaises(ContractError): check_receipt({'passed': True}, {'runtime': {}}, 'x')


class AnalysisTests(unittest.TestCase):
    def rows(self, n=3, variants=2):
        return [dict(case_id=f'{i}-{j}', cluster_id=str(i), op='next', renderer=str(j),
                     bank_identity='bank', model_identity='model', semantic_correct=False)
                for i in range(n) for j in range(variants)]

    def test_cluster_unit(self):
        a = self.rows(); b = copy.deepcopy(a)
        for r in b: r['semantic_correct'] = True
        r = compare(a, b, draws=100)
        self.assertEqual(r['worlds'], 3)
        self.assertEqual(r['mean_world_delta'], 1)
        self.assertNotIn('exact_mcnemar_two_sided', r)

    def test_exact_only_for_singletons(self):
        a = self.rows(3, 1); b = copy.deepcopy(a)
        for r in b: r['semantic_correct'] = True
        self.assertEqual(compare(a, b, draws=100)['exact_mcnemar_two_sided'], .25)

    def test_unmatched_case_rejected(self):
        with self.assertRaises(ContractError): compare(self.rows(), self.rows()[:-1])

    def test_bool_string_rejected(self):
        a = self.rows(); b = copy.deepcopy(a); b[0]['semantic_correct'] = 'false'
        with self.assertRaises(ContractError): compare(a, b)

    def test_model_mismatch_rejected(self):
        a = self.rows(); b = copy.deepcopy(a); b[0]['model_identity'] = 'other'
        with self.assertRaises(ContractError): compare(a, b)

    def test_late_bias_divergence(self):
        r = inspect_trace([[4, 0, -1], [.5, 0, -1]], 1., 1)
        self.assertEqual(r['step'], 1)
        self.assertFalse(r['later_saved_prefixes_valid'])

    def test_bias_tie_rule(self):
        self.assertEqual(inspect_trace([[1., 0.]], 1., 1)['status'], 'UNCHANGED_ON_SUPPLIED_COMPLETE_TRACE')
        self.assertEqual(inspect_trace([[0., 1.]], 1., 0)['status'], 'REGENERATE_FROM_DIVERGENCE')

    def test_nonfinite_bias_rejected(self):
        with self.assertRaises(ContractError): inspect_trace([[0., 1.]], float('nan'), 1)
