#!/usr/bin/env python
# Copyright 2026 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

from pyscf import gto
from pyscf import scf
from pyscf import dft
from pyscf.soscf import gdm


def setUpModule():
    global h2, h2o, oh, h2o_s, oh_s
    h2 = gto.M(
        verbose = 0,
        atom = 'H 0 0 0; H 0 0 1.0',
        basis = 'sto-3g')
    h2o = gto.M(
        verbose = 0,
        atom = '''
        O 0 0 0
        H 0 -0.757 0.587
        H 0  0.757 0.587
        ''',
        basis = 'sto-3g')
    oh = gto.M(
        verbose = 0,
        atom = 'O 0 0 0; H 0 0 0.97',
        basis = 'sto-3g',
        spin = 1)
    h2o_s = h2o.copy()
    h2o_s.symmetry = True
    h2o_s.build(False, False)
    oh_s = oh.copy()
    oh_s.symmetry = True
    oh_s.build(False, False)


def tearDownModule():
    global h2, h2o, oh, h2o_s, oh_s
    del h2, h2o, oh, h2o_s, oh_s


def _run_gdm(mf):
    mf = mf.gdm()
    mf.max_cycle = 80
    mf.conv_tol = 1e-9
    mf.conv_tol_grad = 1e-5
    mf.kernel()
    return mf


def _run_ref(mf):
    mf.max_cycle = 80
    mf.conv_tol = 1e-11
    mf.conv_tol_grad = 1e-7
    mf.kernel()
    return mf


class KnownValues(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_grids = dft.radi.ATOM_SPECIFIC_TREUTLER_GRIDS
        dft.radi.ATOM_SPECIFIC_TREUTLER_GRIDS = False

    @classmethod
    def tearDownClass(cls):
        dft.radi.ATOM_SPECIFIC_TREUTLER_GRIDS = cls.original_grids

    def test_api(self):
        mf = scf.RHF(h2).gdm()
        self.assertIs(scf.gdm(mf), mf)
        self.assertTrue(isinstance(mf, gdm._GDM_SCF))
        self.assertFalse(isinstance(mf.remove_gdm(), gdm._GDM_SCF))

    def test_density_fit_api(self):
        mf = scf.RHF(h2).gdm().density_fit()
        self.assertTrue(isinstance(mf, gdm._GDM_SCF))
        self.assertTrue(hasattr(mf, 'with_df'))
        self.assertTrue(mf.converged is False)

    def test_rhf(self):
        ref = _run_ref(scf.RHF(h2o))
        mf = _run_gdm(scf.RHF(h2o))
        self.assertTrue(mf.converged)
        self.assertAlmostEqual(mf.e_tot, ref.e_tot, 8)

    def test_uhf(self):
        ref = _run_ref(scf.UHF(oh))
        mf = _run_gdm(scf.UHF(oh))
        self.assertTrue(mf.converged)
        self.assertAlmostEqual(mf.e_tot, ref.e_tot, 8)

    def test_rohf(self):
        ref = _run_ref(scf.ROHF(oh))
        mf = _run_gdm(scf.ROHF(oh))
        self.assertTrue(mf.converged)
        self.assertAlmostEqual(mf.e_tot, ref.e_tot, 8)

    def test_ghf(self):
        ref = _run_ref(scf.GHF(h2))
        mf = _run_gdm(scf.GHF(h2))
        self.assertTrue(mf.converged)
        self.assertAlmostEqual(mf.e_tot, ref.e_tot, 8)

    def test_symmetry(self):
        for mf0 in (scf.RHF(h2o_s), scf.ROHF(oh_s), scf.UHF(oh_s)):
            ref = _run_ref(mf0)
            mf = _run_gdm(mf0.__class__(mf0.mol))
            self.assertTrue(mf.converged)
            self.assertAlmostEqual(mf.e_tot, ref.e_tot, 8)

    def test_rks(self):
        ref = dft.RKS(h2o)
        ref.xc = 'lda,vwn'
        ref.grids.level = 0
        ref = _run_ref(ref)
        mf = dft.RKS(h2o)
        mf.xc = 'lda,vwn'
        mf.grids.level = 0
        mf = _run_gdm(mf)
        self.assertTrue(mf.converged)
        self.assertAlmostEqual(mf.e_tot, ref.e_tot, 5)

    def test_uks(self):
        ref = dft.UKS(oh)
        ref.xc = 'lda,vwn'
        ref.grids.level = 0
        ref = _run_ref(ref)
        mf = dft.UKS(oh)
        mf.xc = 'lda,vwn'
        mf.grids.level = 0
        mf = _run_gdm(mf)
        self.assertTrue(mf.converged)
        self.assertAlmostEqual(mf.e_tot, ref.e_tot, 5)

    def test_roks(self):
        ref = dft.ROKS(oh)
        ref.xc = 'lda,vwn'
        ref.grids.level = 0
        ref = _run_ref(ref)
        mf = dft.ROKS(oh)
        mf.xc = 'lda,vwn'
        mf.grids.level = 0
        mf = _run_gdm(mf)
        self.assertTrue(mf.converged)
        self.assertAlmostEqual(mf.e_tot, ref.e_tot, 7)

    def test_hybrid_rks(self):
        ref = dft.RKS(h2)
        ref.xc = 'b3lyp'
        ref.grids.level = 0
        ref = _run_ref(ref)
        mf = dft.RKS(h2)
        mf.xc = 'b3lyp'
        mf.grids.level = 0
        mf = _run_gdm(mf)
        self.assertTrue(mf.converged)
        self.assertAlmostEqual(mf.e_tot, ref.e_tot, 7)


if __name__ == "__main__":
    print("Full Tests for GDM")
    unittest.main()
