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
from unittest import mock

import numpy
import scipy.linalg

from pyscf import gto
from pyscf import scf
from pyscf import dft
from pyscf.soscf import gdm


def setUpModule():
    global h2, h2_stretched, h2o, h2o_cation, oh, h2o_s, oh_s
    h2 = gto.M(
        verbose = 0,
        atom = 'H 0 0 0; H 0 0 1.0',
        basis = 'sto-3g')
    h2_stretched = gto.M(
        verbose = 0,
        atom = 'H 0 0 0; H 0 0 3.0',
        basis = 'sto-3g')
    h2o = gto.M(
        verbose = 0,
        atom = '''
        O 0 0 0
        H 0 -0.757 0.587
        H 0  0.757 0.587
        ''',
        basis = 'sto-3g')
    h2o_cation = gto.M(
        verbose = 0,
        atom = '''
        O 0 0 0
        H 0 -0.757 0.587
        H 0  0.757 0.587
        ''',
        basis = 'sto-3g',
        charge = 1,
        spin = 1)
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
    global h2, h2_stretched, h2o, h2o_cation, oh, h2o_s, oh_s
    del h2, h2_stretched, h2o, h2o_cation, oh, h2o_s, oh_s


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


def _directional_derivative_error(mf, complex_rotation=False):
    mf.verbose = 0
    mf.conv_tol = 1e-11
    mf.kernel()
    mf = mf.gdm()
    mo_occ = mf.mo_occ
    mo_coeff = mf.mo_coeff
    h1e = mf.get_hcore()
    s1e = mf.get_ovlp()

    dm = mf.make_rdm1(mo_coeff, mo_occ)
    vhf = mf.get_veff(mf.mol, dm)
    fock = mf.get_fock(h1e, s1e, vhf, dm, level_shift_factor=0)
    g, _ = mf.gen_g_hdiag(mo_coeff, mo_occ, fock, h1e)

    rng = numpy.random.default_rng(12)
    perturbation = rng.standard_normal(g.size)
    if complex_rotation:
        perturbation = perturbation + 1j*rng.standard_normal(g.size)
    perturbation *= .08 / numpy.linalg.norm(perturbation)
    rotation, perturbation = gdm._rotation_path(
        mf, perturbation, mo_occ, mo_coeff)
    mo_coeff = mf.rotate_mo(mo_coeff, rotation(1.))

    dm = mf.make_rdm1(mo_coeff, mo_occ)
    vhf = mf.get_veff(mf.mol, dm)
    fock = mf.get_fock(h1e, s1e, vhf, dm, level_shift_factor=0)
    fock_mo = gdm._fock_mo(mo_coeff, fock)
    g, _ = mf.gen_g_hdiag(
        mo_coeff, mo_occ, fock, h1e, fock_mo)

    direction = rng.standard_normal(g.size)
    if complex_rotation:
        direction = direction + 1j*rng.standard_normal(g.size)
    direction /= numpy.linalg.norm(direction)
    path, direction = gdm._rotation_path(
        mf, direction, mo_occ, mo_coeff)

    def energy(scale):
        mo1 = mf.rotate_mo(mo_coeff, path(scale))
        dm1 = mf.make_rdm1(mo1, mo_occ)
        vhf1 = mf.get_veff(mf.mol, dm1)
        return mf.energy_tot(dm1, h1e, vhf1)

    epsilon = 1e-4
    finite_difference = (energy(epsilon) - energy(-epsilon)) / (2*epsilon)
    analytic = gdm._directional_derivative(g, direction)
    return finite_difference, analytic


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

        mf = scf.RHF(h2).diis_gdm()
        self.assertIs(scf.diis_gdm(mf), mf)
        self.assertTrue(isinstance(mf, gdm._GDM_SCF))
        self.assertTrue(mf.gdm_diis_start)

    def test_decorated_object_owns_state(self):
        mf = dft.RKS(h2).gdm()
        mf.xc = 'pbe0'
        mf.grids.level = 0
        mf.max_cycle = 40
        mf.conv_tol = 1e-10
        mf.conv_tol_grad = 1e-6
        energy = mf.kernel()

        ref = dft.RKS(h2, xc='pbe0')
        ref.grids.level = 0
        ref.conv_tol = 1e-11
        ref_energy = ref.kernel()
        self.assertEqual(mf.undo_gdm().xc, 'pbe0')
        self.assertAlmostEqual(energy, ref_energy, 9)

        mf = dft.UKS(h2o_cation).gdm()
        mf.xc = 'pbe0'
        mf.grids.level = 0
        mf.max_cycle = 80
        mf.conv_tol = 1e-10
        mf.conv_tol_grad = 1e-6
        energy = mf.kernel()

        ref = dft.UKS(h2o_cation, xc='pbe0').gdm()
        ref.grids.level = 0
        ref.max_cycle = 80
        ref.conv_tol = 1e-10
        ref.conv_tol_grad = 1e-6
        ref_energy = ref.kernel()
        self.assertTrue(mf.converged)
        self.assertTrue(ref.converged)
        self.assertEqual(mf.undo_gdm().xc, 'pbe0')
        self.assertAlmostEqual(energy, ref_energy, 9)

    def test_step_scaling(self):
        step = numpy.asarray([1., 2., -4.])
        step1, norm1, capped1 = gdm._scale_step(step, .5, 'inf')
        self.assertTrue(capped1)
        self.assertAlmostEqual(abs(step1).max(), .5, 12)
        self.assertAlmostEqual(norm1, .5, 12)

        step2, norm2, capped2 = gdm._scale_step(step, .5, '2')
        self.assertTrue(capped2)
        self.assertAlmostEqual(numpy.linalg.norm(step2), .5, 12)
        self.assertAlmostEqual(norm2, .5, 12)

    def test_density_fit_api(self):
        mf0 = scf.RHF(h2).gdm()
        mf0.gdm_step_control = 'dogleg'
        mf0.gdm_space = 17
        mf0.gdm_transport_tol = 2e-9
        mf = mf0.density_fit()
        self.assertTrue(isinstance(mf, gdm._GDM_SCF))
        self.assertTrue(hasattr(mf, 'with_df'))
        self.assertTrue(mf.converged is False)
        self.assertEqual(mf.gdm_step_control, 'dogleg')
        self.assertEqual(mf.gdm_space, 17)
        self.assertEqual(mf.gdm_transport_tol, 2e-9)

        ref = _run_ref(scf.RHF(h2).density_fit())
        mf = _run_gdm(scf.RHF(h2).density_fit())
        self.assertTrue(mf.converged)
        self.assertAlmostEqual(mf.e_tot, ref.e_tot, 9)

    def test_energy_weighted_coordinates(self):
        class FakeMF:
            gdm_hdiag_shift = 'auto'

        self.assertEqual(gdm._hdiag_shift(FakeMF(), -.125), -.125)
        numpy.testing.assert_allclose(
            gdm._hdiag_shift_weights_rhf(numpy.array([2., 2., 0.])),
            [2., 2.])
        numpy.testing.assert_allclose(
            gdm._hdiag_shift_weights_uhf(
                numpy.array([[1., 0., 0.], [1., 1., 0.]])),
            [1., 1., 1., 1.])
        numpy.testing.assert_allclose(
            gdm._hdiag_shift_weights_rohf(numpy.array([2., 1., 0.])),
            [1., 2., 1.])
        numpy.testing.assert_allclose(
            gdm._ewc_scale(numpy.array([-.01, 0., 4.]), .1),
            [.1, .1, 2.])

    def test_grassmann_geodesic(self):
        rng = numpy.random.default_rng(7)
        block = (rng.standard_normal((4, 3))
                 + 1j*rng.standard_normal((4, 3)))
        mo_occ = numpy.array([1., 1., 1., 0., 0., 0., 0.])
        kappa = numpy.zeros((7, 7), dtype=complex)
        kappa[3:, :3] = block
        kappa[:3, 3:] = -block.conj().T
        path = gdm._grassmann_rotation_path(block, mo_occ > 0)
        numpy.testing.assert_allclose(
            path(.73), scipy.linalg.expm(.73*kappa), atol=2e-14)
        coefficients = rng.standard_normal((9, 7))
        numpy.testing.assert_allclose(
            path.rotate(coefficients, .73),
            coefficients.dot(scipy.linalg.expm(.73*kappa)), atol=2e-14)

    def test_rohf_parallel_transport(self):
        rng = numpy.random.default_rng(9)
        mo_occ = numpy.array([2., 2., 1., 0., 0.])
        size = numpy.count_nonzero(scf.hf.uniq_var_indices(mo_occ))
        step = (rng.standard_normal(size)
                + 1j*rng.standard_normal(size)) * .04
        v = (rng.standard_normal(size)
             + 1j*rng.standard_normal(size))
        w = (rng.standard_normal(size)
             + 1j*rng.standard_normal(size))
        tv = gdm._transport_rohf_vertical(
            v, step, mo_occ, tol=1e-13, max_cycle=30)
        tw = gdm._transport_rohf_vertical(
            w, step, mo_occ, tol=1e-13, max_cycle=30)
        ts = gdm._transport_rohf_vertical(
            step, step, mo_occ, tol=1e-13, max_cycle=30)
        self.assertAlmostEqual(gdm._dot(v, w), gdm._dot(tv, tw), 11)
        numpy.testing.assert_allclose(ts, step, atol=1e-13)

        kappa = scf.hf.unpack_uniq_var(step, mo_occ)
        tangent = scf.hf.unpack_uniq_var(v, mo_occ)
        commutator = kappa.dot(tangent) - tangent.dot(kappa)
        dense = scf.hf.pack_uniq_var(commutator, mo_occ)
        blocks = gdm._rohf_commutator_tangent(
            gdm._split_rohf(step, mo_occ),
            gdm._split_rohf(v, mo_occ))
        numpy.testing.assert_allclose(
            gdm._join_rohf(blocks, mo_occ), dense, atol=1e-13)

    def test_dogleg_geometry(self):
        p_cauchy = numpy.array([.1, 0.])
        p_newton = numpy.array([1., 1.])
        step, leg = gdm._dogleg_step(p_cauchy, p_newton, .5)
        self.assertEqual(leg, 'dogleg')
        self.assertAlmostEqual(numpy.linalg.norm(step), .5, 13)

    def test_strong_wolfe_zoom(self):
        class FakeMF:
            gdm_wolfe_c1 = 1e-4
            gdm_wolfe_c2 = .9
            gdm_step_control_max_cycle = 12

        def evaluate(*args, **kwargs):
            coordinate = args[7][0]
            energy = .5 * (coordinate - 2.)**2
            gradient = numpy.array([(coordinate - 2.) * .5])
            return (energy, None, None, None, None, gradient,
                    numpy.ones(1), None, {})

        step = numpy.array([4.])
        with mock.patch.object(
                gdm, '_rotation_path', return_value=(lambda scale: scale,
                                                     step)), \
             mock.patch.object(gdm, '_evaluate_step', side_effect=evaluate):
            result = gdm._strong_wolfe_search(
                FakeMF(), None, None, None, None, None, None, 2.,
                numpy.array([-1.]), step, None)
        self.assertEqual(result[4], 'strong-wolfe')
        self.assertAlmostEqual(result[1], .5, 12)

    def test_directional_derivatives(self):
        methods = [scf.RHF(h2o), scf.UHF(oh), scf.ROHF(oh), scf.GHF(h2)]
        rks = dft.RKS(h2o, xc='pbe0')
        uks = dft.UKS(oh, xc='pbe0')
        roks = dft.ROKS(oh, xc='pbe0')
        for mf in (rks, uks, roks):
            mf.grids.level = 0
        methods.extend((rks, uks, roks))
        for mf in methods:
            complex_rotation = isinstance(mf, scf.ghf.GHF)
            with self.subTest(method=mf.__class__.__name__):
                finite_difference, analytic = _directional_derivative_error(
                    mf, complex_rotation)
                self.assertLess(abs(finite_difference-analytic), 2e-6)

        finite_difference, analytic = _directional_derivative_error(
            scf.UHF(oh), complex_rotation=True)
        self.assertLess(abs(finite_difference-analytic), 2e-6)

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
        self.assertLess(abs(mf.e_tot-ref.e_tot), 1e-5)

    def test_dogleg_rhf(self):
        ref = _run_ref(scf.RHF(h2o))
        mf = scf.RHF(h2o).gdm()
        mf.gdm_step_control = 'dogleg'
        mf.max_cycle = 80
        mf.conv_tol = 1e-9
        mf.conv_tol_grad = 1e-5
        mf.kernel()
        self.assertTrue(mf.converged)
        self.assertAlmostEqual(mf.e_tot, ref.e_tot, 8)

    def test_final_noncanonical_orbitals(self):
        mf = scf.RHF(h2o).gdm()
        mf.canonicalization = False
        mf.max_cycle = 80
        mf.conv_tol = 1e-9
        mf.conv_tol_grad = 1e-5
        mf.kernel()
        dm = mf.make_rdm1()
        fock = mf.get_fock(dm=dm, level_shift_factor=0)
        expected = numpy.einsum(
            'pi,pi->i', mf.mo_coeff.conj(), fock.dot(mf.mo_coeff)).real
        self.assertTrue(mf.converged)
        numpy.testing.assert_allclose(mf.mo_energy, expected, atol=1e-12)

    def test_unrestricted_broken_symmetry_controllers(self):
        dm0 = numpy.zeros((2, h2_stretched.nao, h2_stretched.nao))
        dm0[0,0,0] = 1
        dm0[1,1,1] = 1
        methods = ((scf.UHF, -0.9332846583338104),
                   (lambda mol: dft.UKS(mol, xc='pbe0'),
                    -0.9330229956520144))
        for method, reference in methods:
            for controller in ('strong_wolfe', 'dogleg'):
                mf = method(h2_stretched).gdm()
                if isinstance(mf, dft.uks.UKS):
                    mf.grids.level = 0
                mf.gdm_step_control = controller
                mf.max_cycle = 50
                mf.conv_tol = 1e-10
                mf.conv_tol_grad = 1e-6
                with self.subTest(method=mf.__class__.__name__,
                                  controller=controller):
                    energy = mf.kernel(dm0=dm0)
                    self.assertTrue(mf.converged)
                    self.assertAlmostEqual(energy, reference, 9)
                    self.assertGreater(mf.spin_square()[0], .99)

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

    def test_diis_gdm(self):
        ref = _run_ref(scf.RHF(h2o))
        mf = scf.RHF(h2o).diis_gdm()
        mf.gdm_diis_max_cycle = 2
        mf.max_cycle = 80
        mf.conv_tol = 1e-9
        mf.conv_tol_grad = 1e-5
        mf.kernel()
        self.assertTrue(mf.converged)
        self.assertAlmostEqual(mf.e_tot, ref.e_tot, 8)


if __name__ == "__main__":
    print("Full Tests for GDM")
    unittest.main()
