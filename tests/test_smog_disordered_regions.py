"""Regression checks for optional SMOG disordered-region filtering."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from openabc.forcefields.parsers import SMOGParser


class TestSMOGDisorderedRegions(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.pdb = Path(self.directory.name) / 'protein.pdb'
        coordinates = [(3.8*i, 1.5*(i % 2), float(i % 3)) for i in range(8)]
        self.pdb.write_text(''.join(
            f'ATOM  {i:5d}  CA  ALA A{i:4d}    {x:8.3f}{y:8.3f}{z:8.3f}'
            '  1.00  0.00           C  \n'
            for i, (x, y, z) in enumerate(coordinates, 1)
        ) + 'END\n')

    def parse(self, disordered_atoms=None, **kwargs):
        parser = SMOGParser(str(self.pdb), str(self.pdb), default_parse=False)
        parser.atoms['charge'] = 0.0
        # Supply an imported native map so this tests topology filtering
        # independently of shadow-map geometry and contact-search dependencies.
        parser.native_pairs = pd.DataFrame({
            'a1': [0, 1, 2, 3], 'a2': [4, 5, 6, 7], 'mu': [0.7]*4,
            'epsilon_G': [2.5]*4, 'sigma_G': [0.05]*4, 'alpha_G': [4.194304e-5]*4,
        })
        parser.parse_mol(get_native_pairs=False, disordered_atoms=disordered_atoms, **kwargs)
        return parser

    @staticmethod
    def exclusion_pairs(parser):
        return set(map(tuple, parser.exclusions[['a1', 'a2']].to_numpy(dtype=int)))

    def test_none_and_empty_preserve_standard_topology(self):
        default = self.parse()
        for selection in ([], np.array([], dtype=int)):
            empty = self.parse(selection)
            for attr in ('protein_bonds', 'protein_angles', 'protein_dihedrals',
                         'native_pairs', 'exclusions', 'protein_exclusions'):
                assert_frame_equal(getattr(default, attr), getattr(empty, attr))
        self.assertEqual(len(default.native_pairs), 4)
        self.assertEqual(len(default.protein_dihedrals), 10)

    def test_removes_native_pairs_and_torsions_but_keeps_bonded_exclusions(self):
        default = self.parse()
        parser = self.parse(np.array([1, 2, 2], dtype=int))
        self.assertEqual(set(map(tuple, parser.native_pairs[['a1', 'a2']].to_numpy())),
                         {(0, 4), (3, 7)})
        tuples = set(map(tuple, parser.protein_dihedrals[['a1', 'a2', 'a3', 'a4']].to_numpy()))
        self.assertEqual(tuples, {(3, 4, 5, 6), (4, 5, 6, 7)})
        self.assertEqual(len(parser.protein_dihedrals), 4)
        for attr in ('protein_bonds', 'protein_angles', 'atoms'):
            assert_frame_equal(getattr(default, attr), getattr(parser, attr))
        expected = {(i, i+delta) for delta in (1, 2, 3) for i in range(8-delta)}
        expected.update({(0, 4), (3, 7)})
        self.assertEqual(self.exclusion_pairs(parser), expected)
        assert_frame_equal(parser.exclusions, parser.protein_exclusions)

    def test_reparsing_exclusions_preserves_special_pairs_and_respects_flags(self):
        parser = self.parse([1, 2])
        before = parser.exclusions.copy()
        parser.parse_exclusions()
        assert_frame_equal(before, parser.exclusions)
        parser.parse_exclusions(exclude14=False)
        self.assertFalse({(i, i+3) for i in range(5)} & self.exclusion_pairs(parser))
        parser.parse_exclusions(exclude12=False, exclude13=False, exclude14=False,
                                exclude_native_pairs=False)
        self.assertTrue(parser.exclusions.empty)
        self.assertTrue(parser.protein_exclusions.empty)

    def test_all_disordered_removes_all_native_and_torsion_terms(self):
        parser = self.parse(range(8))
        self.assertTrue(parser.native_pairs.empty)
        self.assertTrue(parser.protein_dihedrals.empty)
        self.assertEqual(len(parser.protein_bonds), 7)
        self.assertEqual(len(parser.protein_angles), 6)
        self.assertEqual(len(parser.exclusions), 7+6+5)

    def test_reparsing_molecule_resets_cached_removed_dihedrals(self):
        parser = self.parse([1, 2])
        parser.parse_mol(get_native_pairs=False)
        self.assertEqual(parser._disordered_dihedral_exclusions, [])
        self.assertEqual(len(parser.protein_dihedrals), 10)

    def test_rejects_invalid_local_indices(self):
        for selection in ([-1], [8], [1.5], [True]):
            with self.subTest(selection=selection), self.assertRaises(ValueError):
                self.parse(selection)


if __name__ == '__main__':
    unittest.main()
