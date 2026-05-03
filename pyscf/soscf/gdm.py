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

'''
Geometric direct minimization SCF solver.
'''

import sys

import numpy
import scipy.linalg

from pyscf import lib
from pyscf import __config__
from pyscf.lib import logger
from pyscf.scf import chkfile
from pyscf.scf import hf
from pyscf.scf import hf_symm, uhf_symm, ghf_symm
# import _response_functions to load gen_response methods in SCF class
from pyscf.scf import _response_functions  # noqa
from pyscf.soscf import newton_ah


def expmat(a):
    return scipy.linalg.expm(a)


def _dot(x, y):
    return numpy.vdot(x, y).real


def _norm_gorb(g):
    norm_gorb = numpy.linalg.norm(g)
    if g.size > 0 and not hf.TIGHT_GRAD_CONV_TOL:
        norm_gorb /= numpy.sqrt(g.size)
    return norm_gorb


def _regularize_hdiag(h_diag, floor):
    h_diag = numpy.asarray(h_diag).real.ravel()
    h_diag = numpy.abs(h_diag)
    h_diag[h_diag < floor] = floor
    return h_diag


def _mo_energy_from_fock(mo_coeff, fock):
    if (isinstance(mo_coeff, (tuple, list)) or
            isinstance(mo_coeff, numpy.ndarray) and mo_coeff.ndim == 3):
        mo_ea = numpy.einsum('pi,pi->i', mo_coeff[0].conj(),
                             fock[0].dot(mo_coeff[0])).real
        mo_eb = numpy.einsum('pi,pi->i', mo_coeff[1].conj(),
                             fock[1].dot(mo_coeff[1])).real
        return numpy.asarray((mo_ea, mo_eb))

    mo_energy = numpy.einsum('pi,pi->i', mo_coeff.conj(),
                             fock.dot(mo_coeff)).real
    if getattr(fock, 'focka', None) is not None:
        mo_ea = numpy.einsum('pi,pi->i', mo_coeff.conj(),
                             fock.focka.dot(mo_coeff)).real
        mo_eb = numpy.einsum('pi,pi->i', mo_coeff.conj(),
                             fock.fockb.dot(mo_coeff)).real
        mo_energy = lib.tag_array(mo_energy, mo_ea=mo_ea, mo_eb=mo_eb)
    return mo_energy


def _sym_forbid_rhf(mf, mo_coeff, occidx, viridx):
    if mf._scf.mol.symmetry:
        orbsym = hf_symm.get_orbsym(mf._scf.mol, mo_coeff)
        return orbsym[viridx,None] != orbsym[occidx]
    return None


def gen_g_hdiag_rhf(mf, mo_coeff, mo_occ, fock_ao, h1e=None):
    occidx = mo_occ > 0
    viridx = ~occidx
    g = mf._scf.get_grad(mo_coeff, mo_occ, fock_ao)

    if g.size == 0:
        return g, g

    fock = mo_coeff.conj().T.dot(fock_ao).dot(mo_coeff)
    e = fock.diagonal().real
    h_diag = (e[viridx,None] - e[occidx]) * 2

    sym_forbid = _sym_forbid_rhf(mf, mo_coeff, occidx, viridx)
    if sym_forbid is not None:
        g = g.copy().reshape(h_diag.shape)
        h_diag = h_diag.copy()
        g[sym_forbid] = 0
        h_diag[sym_forbid] = 0
        g = g.ravel()

    return numpy.asarray(g).ravel(), h_diag.ravel()


def gen_g_hdiag_uhf(mf, mo_coeff, mo_occ, fock_ao, h1e=None):
    occidxa = mo_occ[0] > 0
    occidxb = mo_occ[1] > 0
    viridxa = ~occidxa
    viridxb = ~occidxb
    g = mf._scf.get_grad(mo_coeff, mo_occ, fock_ao)

    if g.size == 0:
        return g, g

    focka = mo_coeff[0].conj().T.dot(fock_ao[0]).dot(mo_coeff[0])
    fockb = mo_coeff[1].conj().T.dot(fock_ao[1]).dot(mo_coeff[1])
    ea = focka.diagonal().real
    eb = fockb.diagonal().real
    h_diaga = ea[viridxa,None] - ea[occidxa]
    h_diagb = eb[viridxb,None] - eb[occidxb]

    if mf._scf.mol.symmetry:
        orbsyma, orbsymb = uhf_symm.get_orbsym(mf._scf.mol, mo_coeff)
        sym_forbida = orbsyma[viridxa,None] != orbsyma[occidxa]
        sym_forbidb = orbsymb[viridxb,None] != orbsymb[occidxb]
        ga = g[:h_diaga.size].copy().reshape(h_diaga.shape)
        gb = g[h_diaga.size:].copy().reshape(h_diagb.shape)
        h_diaga = h_diaga.copy()
        h_diagb = h_diagb.copy()
        ga[sym_forbida] = 0
        gb[sym_forbidb] = 0
        h_diaga[sym_forbida] = 0
        h_diagb[sym_forbidb] = 0
        g = numpy.hstack((ga.ravel(), gb.ravel()))

    h_diag = numpy.hstack((h_diaga.ravel(), h_diagb.ravel()))
    return numpy.asarray(g).ravel(), h_diag


def gen_g_hdiag_rohf(mf, mo_coeff, mo_occ, fock_ao, h1e=None):
    if getattr(fock_ao, 'focka', None) is not None:
        focka = fock_ao.focka
        fockb = fock_ao.fockb
    elif (isinstance(fock_ao, (tuple, list)) or
          getattr(fock_ao, 'ndim', None) == 3):
        focka, fockb = fock_ao
    else:
        focka = fockb = fock_ao

    g = mf._scf.get_grad(mo_coeff, mo_occ, fock_ao)
    if g.size == 0:
        return g, g

    occidxa = mo_occ > 0
    occidxb = mo_occ == 2
    viridxa = ~occidxa
    viridxb = ~occidxb
    uniq_var_a = viridxa[:,None] & occidxa
    uniq_var_b = viridxb[:,None] & occidxb
    uniq_ab = uniq_var_a | uniq_var_b

    focka = mo_coeff.conj().T.dot(focka).dot(mo_coeff)
    fockb = mo_coeff.conj().T.dot(fockb).dot(mo_coeff)
    ea = focka.diagonal().real
    eb = fockb.diagonal().real
    h_diaga = ea[viridxa,None] - ea[occidxa]
    h_diagb = eb[viridxb,None] - eb[occidxb]

    h_diag = numpy.zeros((mo_occ.size, mo_occ.size))
    h_diag[uniq_var_a] = h_diaga.ravel()
    h_diag[uniq_var_b] += h_diagb.ravel()

    if mf._scf.mol.symmetry:
        orbsym = hf_symm.get_orbsym(mf._scf.mol, mo_coeff)
        sym_forbid = orbsym[:,None] != orbsym
        g1 = numpy.zeros_like(h_diag)
        g1[uniq_ab] = g
        g1[sym_forbid] = 0
        h_diag[sym_forbid] = 0
        g = g1[uniq_ab]

    return numpy.asarray(g).ravel(), h_diag[uniq_ab]


def gen_g_hdiag_ghf(mf, mo_coeff, mo_occ, fock_ao, h1e=None):
    occidx = mo_occ > 0
    viridx = ~occidx
    g = mf._scf.get_grad(mo_coeff, mo_occ, fock_ao)

    if g.size == 0:
        return g, g

    fock = mo_coeff.conj().T.dot(fock_ao).dot(mo_coeff)
    e = fock.diagonal().real
    h_diag = e[viridx,None] - e[occidx]

    if mf._scf.mol.symmetry:
        orbsym = ghf_symm.get_orbsym(mf._scf.mol, mo_coeff)
        sym_forbid = orbsym[viridx,None] != orbsym[occidx]
        g = g.copy().reshape(h_diag.shape)
        h_diag = h_diag.copy()
        g[sym_forbid] = 0
        h_diag[sym_forbid] = 0
        g = g.ravel()

    return numpy.asarray(g).ravel(), h_diag.ravel()


def _lbfgs_direction(g_orb, h_diag, history):
    q = g_orb.copy()
    alphas = []
    for s, y, rho in reversed(history):
        alpha = rho * _dot(s, q)
        q -= alpha * y
        alphas.append(alpha)

    z = q / h_diag

    for (s, y, rho), alpha in zip(history, reversed(alphas)):
        beta = rho * _dot(y, z)
        z += s * (alpha - beta)
    return -z


def _pack_uhf(dx, mo_occ):
    occidxa = mo_occ[0] > 0
    occidxb = mo_occ[1] > 0
    viridxa = ~occidxa
    viridxb = ~occidxb
    uniq = numpy.array((viridxa[:,None] & occidxa,
                        viridxb[:,None] & occidxb))
    return dx[uniq]


def _unpack_uhf(dx, mo_occ):
    occidxa = mo_occ[0] > 0
    occidxb = mo_occ[1] > 0
    viridxa = ~occidxa
    viridxb = ~occidxb
    nmo = len(occidxa)
    x = numpy.zeros((2,nmo,nmo), dtype=dx.dtype)
    uniq = numpy.array((viridxa[:,None] & occidxa,
                        viridxb[:,None] & occidxb))
    x[uniq] = dx
    return x - x.conj().transpose(0,2,1)


def _is_uhf_mo(mo_coeff):
    return (isinstance(mo_coeff, (tuple, list)) or
            isinstance(mo_coeff, numpy.ndarray) and mo_coeff.ndim == 3)


def _orbital_transform(mo0, mo1, s1e):
    if _is_uhf_mo(mo0):
        return numpy.asarray((mo0[0].conj().T.dot(s1e).dot(mo1[0]),
                              mo0[1].conj().T.dot(s1e).dot(mo1[1])))
    return mo0.conj().T.dot(s1e).dot(mo1)


def _transport_vec(dx, mo_occ, u):
    if isinstance(u, numpy.ndarray) and u.ndim == 3:
        x = _unpack_uhf(dx, mo_occ)
        x = numpy.asarray((u[0].conj().T.dot(x[0]).dot(u[0]),
                           u[1].conj().T.dot(x[1]).dot(u[1])))
        return _pack_uhf(x, mo_occ)

    x = hf.unpack_uniq_var(dx, mo_occ)
    x = u.conj().T.dot(x).dot(u)
    return hf.pack_uniq_var(x, mo_occ)


def _add_history(history, step, g0, g1, mo_occ, u, max_space, min_curvature):
    step = _transport_vec(step, mo_occ, u)
    g0 = _transport_vec(g0, mo_occ, u)
    y = g1 - g0
    sy = _dot(step, y)
    if sy > min_curvature * max(1., numpy.linalg.norm(step) * numpy.linalg.norm(y)):
        history.append((step.copy(), y.copy(), 1. / sy))
        if len(history) > max_space:
            del history[0]


def _scale_step(step, max_stepsize):
    norm_step = numpy.linalg.norm(step)
    if norm_step > max_stepsize:
        step = step * (max_stepsize / norm_step)
        norm_step = max_stepsize
    return step, norm_step


def _evaluate_step(mf, mo_coeff, mo_occ, dm, vhf, h1e, s1e, step, log):
    mol = mf._scf.mol
    u = mf.update_rotate_matrix(step, mo_occ, mo_coeff=mo_coeff)
    mo1 = mf.rotate_mo(mo_coeff, u, log)
    dm1 = mf.make_rdm1(mo1, mo_occ)
    vhf1 = mf._scf.get_veff(mol, dm1, dm_last=dm, vhf_last=vhf)
    e1 = mf._scf.energy_tot(dm1, h1e, vhf1)
    fock1 = mf.get_fock(h1e, s1e, vhf1, dm1, level_shift_factor=0)
    mo_energy1 = _mo_energy_from_fock(mo1, fock1)
    return e1, mo_energy1, mo1, dm1, vhf1, fock1


def _line_search(mf, mo_coeff, mo_occ, dm, vhf, h1e, s1e, e_tot,
                 g_orb, step, log):
    if not mf.gdm_line_search:
        out = _evaluate_step(mf, mo_coeff, mo_occ, dm, vhf, h1e, s1e,
                             step, log)
        return step, 1., out

    de = _dot(g_orb, step)
    scale = 1.
    for _ in range(mf.gdm_line_search_max_cycle):
        trial_step = step * scale
        out = _evaluate_step(mf, mo_coeff, mo_occ, dm, vhf, h1e, s1e,
                             trial_step, log)
        e1 = out[0]
        if numpy.isfinite(e1) and e1 <= e_tot + mf.gdm_armijo * scale * de:
            return trial_step, scale, out
        log.debug('GDM line search rejects scale %g E=%.15g', scale, e1)
        scale *= .5
    return None, scale, None


def kernel(mf, mo_coeff=None, mo_occ=None, dm=None,
           conv_tol=1e-10, conv_tol_grad=None, max_cycle=50, dump_chk=True,
           callback=None, verbose=logger.NOTE):
    cput0 = (logger.process_clock(), logger.perf_counter())
    log = logger.new_logger(mf, verbose)
    mol = mf._scf.mol

    if conv_tol_grad is None:
        conv_tol_grad = numpy.sqrt(conv_tol)
        log.info('Set conv_tol_grad to %g', conv_tol_grad)

    h1e = mf._scf.get_hcore(mol)
    s1e = mf._scf.get_ovlp(mol)
    x_orth = mf._scf.check_linear_dependency(s1e, log)

    if mo_coeff is not None and mo_occ is not None:
        dm = mf.make_rdm1(mo_coeff, mo_occ)
        vhf = mf._scf.get_veff(mol, dm)
        fock = mf.get_fock(h1e, s1e, vhf, dm, level_shift_factor=0)
        mo_energy = _mo_energy_from_fock(mo_coeff, fock)
    else:
        if dm is None:
            logger.debug(mf, 'Initial guess density matrix is not given. '
                         'Generating initial guess from %s', mf.init_guess)
            dm = mf.get_init_guess(mol, mf.init_guess)
        vhf = mf._scf.get_veff(mol, dm)
        fock = mf.get_fock(h1e, s1e, vhf, dm, level_shift_factor=0)
        mo_energy, mo_coeff = mf.eig(fock, s1e, x=x_orth)
        mo_occ = mf.get_occ(mo_energy, mo_coeff)
        dm_last = dm
        dm = mf.make_rdm1(mo_coeff, mo_occ)
        vhf = mf._scf.get_veff(mol, dm, dm_last=dm_last, vhf_last=vhf)
        fock = mf.get_fock(h1e, s1e, vhf, dm, level_shift_factor=0)
        mo_energy = _mo_energy_from_fock(mo_coeff, fock)

    mf.mo_coeff, mf.mo_occ = mo_coeff, mo_occ
    e_tot = mf._scf.energy_tot(dm, h1e, vhf)
    g_orb, h_diag = mf.gen_g_hdiag(mo_coeff, mo_occ, fock, h1e)
    norm_gorb = _norm_gorb(g_orb)
    log.info('Initial guess E= %.15g  |g|= %g', e_tot, norm_gorb)

    if mf.max_cycle <= 0:
        return False, e_tot, mo_energy, mo_coeff, mo_occ

    if dump_chk and mf.chkfile:
        chkfile.save_mol(mol, mf.chkfile)

    if mol is mf.mol and not getattr(mf, 'with_df', None):
        mf._eri = mf._scf._eri

    scf_conv = g_orb.size == 0 or norm_gorb < conv_tol_grad
    history = []
    cput1 = log.timer('initializing GDM SCF', *cput0)

    for cycle in range(max_cycle):
        if scf_conv:
            break

        last_hf_e = e_tot
        h_diag = _regularize_hdiag(h_diag, mf.gdm_hdiag_floor)
        step = _lbfgs_direction(g_orb, h_diag, history)
        if (not numpy.all(numpy.isfinite(step))
                or _dot(g_orb, step) >= -mf.gdm_min_curvature):
            history = []
            step = -g_orb / h_diag

        step, norm_step = _scale_step(step, mf.max_stepsize)
        if norm_step < mf.gdm_step_tol:
            log.warn('GDM step size below threshold %g', mf.gdm_step_tol)
            break

        accepted_step, _, out = _line_search(
            mf, mo_coeff, mo_occ, dm, vhf, h1e, s1e, e_tot,
            g_orb, step, log)
        if out is None:
            history = []
            step = -g_orb / h_diag
            step, norm_step = _scale_step(step, mf.max_stepsize)
            accepted_step, _, out = _line_search(
                mf, mo_coeff, mo_occ, dm, vhf, h1e, s1e, e_tot,
                g_orb, step, log)
        if out is None:
            log.warn('GDM line search failed to find a downhill step')
            break

        mo0 = mo_coeff
        e_tot, mo_energy, mo_coeff, dm, vhf, fock = out
        if mf.gdm_pcanonicalization:
            mo_energy, mo_coeff = mf._scf.canonicalize(mo_coeff, mo_occ, fock)
        u = _orbital_transform(mo0, mo_coeff, s1e)
        g1, h_diag1 = mf.gen_g_hdiag(mo_coeff, mo_occ, fock, h1e)
        norm_gorb = _norm_gorb(g1)
        _add_history(history, accepted_step, g_orb, g1, mo_occ, u,
                     mf.gdm_space, mf.gdm_min_curvature)

        log.info('cycle= %d E= %.15g  delta_E= %g  |g|= %g  |step|= %g',
                 cycle+1, e_tot, e_tot-last_hf_e, norm_gorb,
                 numpy.linalg.norm(accepted_step))
        cput1 = log.timer('cycle= %d'%(cycle+1), *cput1)

        g_orb = g1
        h_diag = h_diag1
        mf.cycles = cycle + 1

        if callable(mf.check_convergence):
            scf_conv = mf.check_convergence(locals())
        elif abs(e_tot-last_hf_e) < conv_tol and norm_gorb < conv_tol_grad:
            scf_conv = True

        if dump_chk:
            mf.dump_chk(locals())

        if callable(callback):
            callback(locals())

    if callable(callback):
        callback(locals())

    fock = mf.get_fock(h1e, s1e, vhf, dm, level_shift_factor=0)
    mo_energy, mo_coeff1 = mf._scf.canonicalize(mo_coeff, mo_occ, fock)
    if mf.canonicalization:
        log.info('Canonicalize SCF orbitals')
        mo_coeff = mo_coeff1
        if dump_chk:
            mf.dump_chk(locals())

    log.info('GDM macro X = %d  E=%.15g  |g|= %g',
             getattr(mf, 'cycles', 0), e_tot, norm_gorb)
    return scf_conv, e_tot, mo_energy, mo_coeff, mo_occ


class _GDM_SCF:
    '''
    Attributes for GDM solver:
        gdm_space : int
            L-BFGS history size. Default is 10.
        max_stepsize : float
            Maximum norm of one orbital-rotation step. Default is 0.05.
        canonicalization : bool
            Whether to canonicalize final orbitals. Default is True.
    '''

    __name_mixin__ = 'GDM'

    gdm_space = getattr(__config__, 'soscf_gdm_GDM_space', 10)
    max_stepsize = getattr(__config__, 'soscf_gdm_GDM_max_stepsize', .05)
    canonicalization = getattr(__config__, 'soscf_gdm_GDM_canonicalization', True)
    gdm_line_search = getattr(__config__, 'soscf_gdm_GDM_line_search', True)
    gdm_line_search_max_cycle = getattr(
        __config__, 'soscf_gdm_GDM_line_search_max_cycle', 12)
    gdm_armijo = getattr(__config__, 'soscf_gdm_GDM_armijo', 1e-4)
    gdm_hdiag_floor = getattr(__config__, 'soscf_gdm_GDM_hdiag_floor', 1e-4)
    gdm_min_curvature = getattr(__config__, 'soscf_gdm_GDM_min_curvature', 1e-10)
    gdm_step_tol = getattr(__config__, 'soscf_gdm_GDM_step_tol', 1e-12)
    gdm_pcanonicalization = getattr(
        __config__, 'soscf_gdm_GDM_pcanonicalization', True)

    _keys = {
        'gdm_space', 'max_stepsize', 'canonicalization', 'gdm_line_search',
        'gdm_line_search_max_cycle', 'gdm_armijo', 'gdm_hdiag_floor',
        'gdm_min_curvature', 'gdm_step_tol', 'gdm_pcanonicalization',
    }

    def __init__(self, mf):
        self.__dict__.update(mf.__dict__)
        self._scf = mf

    def undo_gdm(self):
        '''Remove the GDM mixin.'''
        obj = lib.view(self, lib.drop_class(self.__class__, _GDM_SCF))
        del obj._scf
        if hasattr(self._scf, 'with_df'):
            obj.with_df = self._scf.with_df
        return obj

    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
        log.info('\n')
        super().dump_flags(verbose)
        log.info('******** %s GDM solver flags ********', self._scf.__class__)
        log.info('SCF tol = %g', self.conv_tol)
        log.info('conv_tol_grad = %s', self.conv_tol_grad)
        log.info('max. SCF cycles = %d', self.max_cycle)
        log.info('direct_scf = %s', self._scf.direct_scf)
        if self._scf.direct_scf:
            log.info('direct_scf_tol = %g', self._scf.direct_scf_tol)
        if self.chkfile:
            log.info('chkfile to save SCF result = %s', self.chkfile)
        log.info('gdm_space = %d', self.gdm_space)
        log.info('max_stepsize = %g', self.max_stepsize)
        log.info('gdm_line_search = %s', self.gdm_line_search)
        log.info('gdm_line_search_max_cycle = %d',
                 self.gdm_line_search_max_cycle)
        log.info('gdm_armijo = %g', self.gdm_armijo)
        log.info('gdm_hdiag_floor = %g', self.gdm_hdiag_floor)
        log.info('gdm_pcanonicalization = %s', self.gdm_pcanonicalization)
        log.info('canonicalization = %s', self.canonicalization)
        log.info('max_memory %d MB (current use %d MB)',
                 self.max_memory, lib.current_memory()[0])
        return self

    def build(self, mol=None):
        if mol is None:
            mol = self.mol
        if self.verbose >= logger.WARN:
            self.check_sanity()
        self._scf.build(mol)
        self._opt = {None: None}
        self._eri = None
        return self

    def reset(self, mol=None):
        if mol is not None:
            self.mol = mol
        self._scf.reset(mol)
        return self

    def kernel(self, mo_coeff=None, mo_occ=None, dm0=None):
        cput0 = (logger.process_clock(), logger.perf_counter())
        if dm0 is not None:
            if isinstance(dm0, str):
                sys.stderr.write('GDM solver reads density matrix from chkfile %s\n' % dm0)
                dm0 = self.from_chk(dm0)

        elif mo_coeff is not None and mo_occ is None:
            logger.warn(self, 'GDM solver expects mo_coeff with '
                        'mo_occ as initial guess but mo_occ is not found in '
                        'the arguments.\n      The given '
                        'argument is treated as density matrix.')
            dm0 = mo_coeff
            mo_coeff = mo_occ = None

        else:
            if mo_coeff is None:
                mo_coeff = self.mo_coeff
            if mo_occ is None:
                mo_occ = self.mo_occ

        self.build(self.mol)
        self.dump_flags()

        self.converged, self.e_tot, \
                self.mo_energy, self.mo_coeff, self.mo_occ = \
                kernel(self, mo_coeff, mo_occ, dm0, conv_tol=self.conv_tol,
                       conv_tol_grad=self.conv_tol_grad,
                       max_cycle=self.max_cycle,
                       callback=self.callback, verbose=self.verbose)

        logger.timer(self, 'GDM SCF', *cput0)
        self._finalize()
        return self.e_tot

    def from_dm(self, dm):
        '''Transform the initial guess density matrix to orbital coefficients.'''
        mf = self._scf
        mol = mf.mol
        h1e = mf.get_hcore(mol)
        s1e = mf.get_ovlp(mol)
        vhf = mf.get_veff(mol, dm)
        fock = mf.get_fock(h1e, s1e, vhf, dm)
        mo_energy, mo_coeff = mf.eig(fock, s1e)
        mo_occ = mf.get_occ(mo_energy, mo_coeff)
        return mo_coeff, mo_occ

    gen_g_hop = newton_ah.gen_g_hop_rhf
    gen_g_hdiag = gen_g_hdiag_rhf

    def update_rotate_matrix(self, dx, mo_occ, u0=1, mo_coeff=None):
        dr = hf.unpack_uniq_var(dx, mo_occ)
        if isinstance(u0, int) and u0 == 1:
            return expmat(dr)
        else:
            return numpy.dot(u0, expmat(dr))

    def rotate_mo(self, mo_coeff, u, log=None):
        mo = numpy.dot(mo_coeff, u)
        if self._scf.mol.symmetry:
            orbsym = hf_symm.get_orbsym(self._scf.mol, mo_coeff)
            mo = lib.tag_array(mo, orbsym=orbsym)
        return mo

    def density_fit(self, auxbasis=None, with_df=None, only_dfj=False):
        return self.undo_gdm().density_fit(auxbasis, with_df, only_dfj).gdm()

    def to_gpu(self):
        return self.undo_gdm().to_gpu().gdm()


class _GDMROHF(_GDM_SCF):
    gen_g_hop = newton_ah.gen_g_hop_rohf
    gen_g_hdiag = gen_g_hdiag_rohf


class _GDMUHF(_GDM_SCF):
    gen_g_hop = newton_ah.gen_g_hop_uhf
    gen_g_hdiag = gen_g_hdiag_uhf

    def update_rotate_matrix(self, dx, mo_occ, u0=1, mo_coeff=None):
        occidxa = mo_occ[0] > 0
        occidxb = mo_occ[1] > 0
        viridxa = ~occidxa
        viridxb = ~occidxb

        nmo = len(occidxa)
        dr = numpy.zeros((2,nmo,nmo), dtype=dx.dtype)
        uniq = numpy.array((viridxa[:,None] & occidxa,
                            viridxb[:,None] & occidxb))
        dr[uniq] = dx
        dr = dr - dr.conj().transpose(0,2,1)

        if (self._scf.mol.symmetry and
                self._scf.mol.groupname in ('SO3', 'Dooh', 'Coov')):
            orbsyma, orbsymb = uhf_symm.get_orbsym(self._scf.mol, mo_coeff)
            if self._scf.mol.groupname == 'SO3':
                newton_ah._force_SO3_degeneracy_(dr[0], orbsyma)
                newton_ah._force_SO3_degeneracy_(dr[1], orbsymb)
            else:
                newton_ah._force_Ex_Ey_degeneracy_(dr[0], orbsyma)
                newton_ah._force_Ex_Ey_degeneracy_(dr[1], orbsymb)

        if isinstance(u0, int) and u0 == 1:
            return numpy.asarray((expmat(dr[0]), expmat(dr[1])))
        else:
            return numpy.asarray((numpy.dot(u0[0], expmat(dr[0])),
                                  numpy.dot(u0[1], expmat(dr[1]))))

    def rotate_mo(self, mo_coeff, u, log=None):
        mo = numpy.asarray((numpy.dot(mo_coeff[0], u[0]),
                            numpy.dot(mo_coeff[1], u[1])))
        if self._scf.mol.symmetry:
            orbsym = uhf_symm.get_orbsym(self._scf.mol, mo_coeff)
            mo = lib.tag_array(mo, orbsym=orbsym)
        return mo

    def spin_square(self, mo_coeff=None, s=None):
        if mo_coeff is None:
            mo_coeff = (self.mo_coeff[0][:,self.mo_occ[0]>0],
                        self.mo_coeff[1][:,self.mo_occ[1]>0])
        if getattr(self, '_scf', None) and self._scf.mol != self.mol:
            s = self._scf.get_ovlp()
        return self._scf.spin_square(mo_coeff, s)

    def kernel(self, mo_coeff=None, mo_occ=None, dm0=None):
        if isinstance(mo_coeff, numpy.ndarray) and mo_coeff.ndim == 2:
            mo_coeff = (mo_coeff, mo_coeff)
        if isinstance(mo_occ, numpy.ndarray) and mo_occ.ndim == 1:
            mo_occ = (numpy.asarray(mo_occ > 0, dtype=numpy.double),
                      numpy.asarray(mo_occ == 2, dtype=numpy.double))
        return _GDM_SCF.kernel(self, mo_coeff, mo_occ, dm0)


class _GDMGHF(_GDM_SCF):
    gen_g_hop = newton_ah.gen_g_hop_ghf
    gen_g_hdiag = gen_g_hdiag_ghf

    def rotate_mo(self, mo_coeff, u, log=None):
        mo = numpy.dot(mo_coeff, u)
        if self._scf.mol.symmetry:
            orbsym = ghf_symm.get_orbsym(self._scf.mol, mo_coeff)
            mo = lib.tag_array(mo, orbsym=orbsym)
        return mo


class _GDMRHF(_GDM_SCF):
    gen_g_hop = newton_ah.gen_g_hop_rhf
    gen_g_hdiag = gen_g_hdiag_rhf


def gdm(mf):
    '''Geometric direct minimization SCF solver.

    Examples:

    >>> mol = gto.M(atom='H 0 0 0; H 0 0 1.1', basis='cc-pvdz')
    >>> mf = scf.RHF(mol).gdm()
    >>> mf.kernel()
    -1.0811707843774987
    '''
    if isinstance(mf, _GDM_SCF):
        return mf

    assert isinstance(mf, hf.SCF)

    if mf.istype('ROHF'):
        cls = _GDMROHF
    elif mf.istype('UHF'):
        cls = _GDMUHF
    elif mf.istype('GHF'):
        cls = _GDMGHF
    elif mf.istype('DHF') or mf.istype('RDHF'):
        raise NotImplementedError('GDM is not implemented for DHF/RDHF')
    else:
        cls = _GDMRHF
    return lib.set_class(cls(mf), (cls, mf.__class__))


def remove_gdm(mf):
    '''Remove the GDM decorator.'''
    return mf.undo_gdm()
