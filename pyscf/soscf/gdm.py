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

r'''Geometric direct minimization SCF solver.

The implementation follows the Grassmann-manifold GDM algorithm of
Van Voorhis and Head-Gordon [1], the restricted open-shell extension of
Dunietz, Van Voorhis, and Head-Gordon [2], and the flag-manifold parallel
transport formulation of Burton [3].

References:

[1] Mol. Phys. 100, 1713 (2002), doi:10.1080/00268970110103642.
[2] J. Theor. Comput. Chem. 1, 255 (2002).
[3] J. Chem. Theory Comput. 21, 9444 (2025),
    doi:10.1021/acs.jctc.5c00898.
'''

import sys
import warnings

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


def _directional_derivative(g, step):
    # PySCF stores one triangle of the anti-Hermitian rotation generator.
    return 2 * _dot(g, step)


def _norm_gorb(g):
    norm_gorb = numpy.linalg.norm(g)
    if g.size > 0 and not hf.TIGHT_GRAD_CONV_TOL:
        norm_gorb /= numpy.sqrt(g.size)
    return norm_gorb


def _as_step_control_mode(mode):
    if mode is True:
        return 'armijo'
    if not mode:
        return 'none'
    mode = str(mode).lower()
    if mode in ('1', 'true', 'yes'):
        return 'armijo'
    if mode in ('wolfe', 'line_search', 'linesearch'):
        return 'strong_wolfe'
    if mode in ('qls', 'gdm_qls', 'quadratic'):
        warnings.warn('The experimental GDM quadratic/QLS controller has '
                      'been removed; using Armijo backtracking instead.',
                      DeprecationWarning, stacklevel=3)
        return 'armijo'
    return mode


def _step_control_mode(mf):
    legacy = getattr(mf, 'gdm_line_search', None)
    if legacy is not None:
        warnings.warn('gdm_line_search is deprecated; use gdm_step_control.',
                      DeprecationWarning, stacklevel=3)
        mode = _as_step_control_mode(legacy)
    else:
        mode = _as_step_control_mode(mf.gdm_step_control)
    if mode not in ('strong_wolfe', 'dogleg', 'armijo', 'none'):
        raise ValueError('Unsupported GDM step controller %s' % mode)
    return mode


def _hdiag_shift(mf, delta_e):
    shift = mf.gdm_hdiag_shift
    if isinstance(shift, str):
        if shift.lower() == 'auto':
            return delta_e
        return float(shift)
    return float(shift)


def _ewc_scale(h_diag, floor):
    alpha = numpy.sqrt(numpy.abs(numpy.asarray(h_diag).real.ravel()))
    alpha[alpha < floor] = floor
    return alpha


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


def _fock_mo(mo_coeff, fock):
    if _is_uhf_mo(mo_coeff):
        return numpy.asarray((
            mo_coeff[0].conj().T.dot(fock[0]).dot(mo_coeff[0]),
            mo_coeff[1].conj().T.dot(fock[1]).dot(mo_coeff[1])))

    fock1 = mo_coeff.conj().T.dot(fock).dot(mo_coeff)
    if getattr(fock, 'focka', None) is not None:
        focka = mo_coeff.conj().T.dot(fock.focka).dot(mo_coeff)
        fockb = mo_coeff.conj().T.dot(fock.fockb).dot(mo_coeff)
        fock1 = lib.tag_array(fock1, focka=focka, fockb=fockb)
    return fock1


def _mo_energy_from_fock_mo(fock_mo):
    if isinstance(fock_mo, numpy.ndarray) and fock_mo.ndim == 3:
        return numpy.asarray((fock_mo[0].diagonal().real,
                              fock_mo[1].diagonal().real))
    mo_energy = fock_mo.diagonal().real
    if getattr(fock_mo, 'focka', None) is not None:
        mo_energy = lib.tag_array(
            mo_energy, mo_ea=fock_mo.focka.diagonal().real,
            mo_eb=fock_mo.fockb.diagonal().real)
    return mo_energy


def _sym_forbid_rhf(mf, mo_coeff, occidx, viridx):
    if mf.mol.symmetry:
        orbsym = hf_symm.get_orbsym(mf.mol, mo_coeff)
        return orbsym[viridx,None] != orbsym[occidx]
    return None


def gen_g_hdiag_rhf(mf, mo_coeff, mo_occ, fock_ao, h1e=None,
                    fock_mo=None):
    occidx = mo_occ > 0
    viridx = ~occidx
    if fock_mo is None:
        fock_mo = _fock_mo(mo_coeff, fock_ao)
    g = fock_mo[numpy.ix_(viridx, occidx)].ravel() * 2

    if g.size == 0:
        return g, g

    e = fock_mo.diagonal().real
    h_diag = (e[viridx,None] - e[occidx]) * 2

    sym_forbid = _sym_forbid_rhf(mf, mo_coeff, occidx, viridx)
    if sym_forbid is not None:
        g = g.copy().reshape(h_diag.shape)
        h_diag = h_diag.copy()
        g[sym_forbid] = 0
        h_diag[sym_forbid] = 0
        g = g.ravel()

    return numpy.asarray(g).ravel(), h_diag.ravel()


def gen_g_hdiag_uhf(mf, mo_coeff, mo_occ, fock_ao, h1e=None,
                    fock_mo=None):
    occidxa = mo_occ[0] > 0
    occidxb = mo_occ[1] > 0
    viridxa = ~occidxa
    viridxb = ~occidxb
    if fock_mo is None:
        fock_mo = _fock_mo(mo_coeff, fock_ao)
    ga = fock_mo[0][numpy.ix_(viridxa, occidxa)].ravel()
    gb = fock_mo[1][numpy.ix_(viridxb, occidxb)].ravel()
    g = numpy.hstack((ga, gb))

    if g.size == 0:
        return g, g

    ea = fock_mo[0].diagonal().real
    eb = fock_mo[1].diagonal().real
    h_diaga = ea[viridxa,None] - ea[occidxa]
    h_diagb = eb[viridxb,None] - eb[occidxb]

    if mf.mol.symmetry:
        orbsyma, orbsymb = uhf_symm.get_orbsym(mf.mol, mo_coeff)
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


def gen_g_hdiag_rohf(mf, mo_coeff, mo_occ, fock_ao, h1e=None,
                     fock_mo=None):
    occidxa = mo_occ > 0
    occidxb = mo_occ == 2
    viridxa = ~occidxa
    viridxb = ~occidxb
    uniq_var_a = viridxa[:,None] & occidxa
    uniq_var_b = viridxb[:,None] & occidxb
    uniq_ab = uniq_var_a | uniq_var_b

    if fock_mo is None:
        fock_mo = _fock_mo(mo_coeff, fock_ao)
    if getattr(fock_mo, 'focka', None) is not None:
        focka = fock_mo.focka
        fockb = fock_mo.fockb
    elif getattr(fock_mo, 'ndim', None) == 3:
        focka, fockb = fock_mo
    else:
        focka = fockb = fock_mo

    g1 = numpy.zeros_like(focka)
    g1[uniq_var_a] = focka[uniq_var_a]
    g1[uniq_var_b] += fockb[uniq_var_b]
    g = g1[uniq_ab]
    if g.size == 0:
        return g, g

    ea = focka.diagonal().real
    eb = fockb.diagonal().real
    h_diaga = ea[viridxa,None] - ea[occidxa]
    h_diagb = eb[viridxb,None] - eb[occidxb]

    h_diag = numpy.zeros((mo_occ.size, mo_occ.size))
    h_diag[uniq_var_a] = h_diaga.ravel()
    h_diag[uniq_var_b] += h_diagb.ravel()

    if mf.mol.symmetry:
        orbsym = hf_symm.get_orbsym(mf.mol, mo_coeff)
        sym_forbid = orbsym[:,None] != orbsym
        g1 = numpy.zeros_like(h_diag)
        g1[uniq_ab] = g
        g1[sym_forbid] = 0
        h_diag[sym_forbid] = 0
        g = g1[uniq_ab]

    return numpy.asarray(g).ravel(), h_diag[uniq_ab]


def gen_g_hdiag_ghf(mf, mo_coeff, mo_occ, fock_ao, h1e=None,
                    fock_mo=None):
    occidx = mo_occ > 0
    viridx = ~occidx
    if fock_mo is None:
        fock_mo = _fock_mo(mo_coeff, fock_ao)
    g = fock_mo[numpy.ix_(viridx, occidx)].ravel()

    if g.size == 0:
        return g, g

    e = fock_mo.diagonal().real
    h_diag = e[viridx,None] - e[occidx]

    if mf.mol.symmetry:
        orbsym = ghf_symm.get_orbsym(mf.mol, mo_coeff)
        sym_forbid = orbsym[viridx,None] != orbsym[occidx]
        g = g.copy().reshape(h_diag.shape)
        h_diag = h_diag.copy()
        g[sym_forbid] = 0
        h_diag[sym_forbid] = 0
        g = g.ravel()

    return numpy.asarray(g).ravel(), h_diag.ravel()


def _hdiag_shift_weights_rhf(mo_occ):
    nocc = numpy.count_nonzero(mo_occ > 0)
    return numpy.full((mo_occ.size - nocc) * nocc, 2.)


def _hdiag_shift_weights_uhf(mo_occ):
    nocca = numpy.count_nonzero(mo_occ[0] > 0)
    noccb = numpy.count_nonzero(mo_occ[1] > 0)
    nmo = len(mo_occ[0])
    return numpy.ones((nmo - nocca) * nocca + (nmo - noccb) * noccb)


def _hdiag_shift_weights_rohf(mo_occ):
    coreidx = mo_occ == 2
    openidx = mo_occ == 1
    viridx = mo_occ == 0
    weights = numpy.zeros((mo_occ.size, mo_occ.size))
    weights[openidx[:,None] & coreidx] = 1.
    weights[viridx[:,None] & coreidx] = 2.
    weights[viridx[:,None] & openidx] = 1.
    return weights[weights != 0]


def _hdiag_shift_weights_ghf(mo_occ):
    nocc = numpy.count_nonzero(mo_occ > 0)
    return numpy.ones((mo_occ.size - nocc) * nocc)


def _lbfgs_direction(g_ewc, history):
    q = g_ewc.copy()
    alphas = []
    for s, y, rho in reversed(history):
        alpha = rho * _dot(s, q)
        q -= alpha * y
        alphas.append(alpha)

    if history:
        s, y, _ = history[-1]
        yy = _dot(y, y)
        if yy > 0:
            z = q * max(1e-3, min(1e3, _dot(s, y) / yy))
        else:
            z = q
    else:
        z = q.copy()

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


def _force_rotation_degeneracy(mf, kappa, mo_coeff):
    mol = mf.mol
    if (not newton_ah.WITH_EX_EY_DEGENERACY or not mol.symmetry
            or mol.groupname not in ('SO3', 'Dooh', 'Coov')):
        return
    if kappa.ndim == 3:
        orbsyma, orbsymb = uhf_symm.get_orbsym(mol, mo_coeff)
        orbsyms = (orbsyma, orbsymb)
        kappas = kappa
    else:
        if mf.istype('GHF'):
            orbsyms = (ghf_symm.get_orbsym(mol, mo_coeff),)
        else:
            orbsyms = (hf_symm.get_orbsym(mol, mo_coeff),)
        kappas = (kappa,)
    for matrix, orbsym in zip(kappas, orbsyms):
        if mol.groupname == 'SO3':
            newton_ah._force_SO3_degeneracy_(matrix, orbsym)
        else:
            newton_ah._force_Ex_Ey_degeneracy_(matrix, orbsym)


def _grassmann_rotation_path(block, occidx):
    viridx = ~occidx
    nmo = occidx.size
    dtype = numpy.result_type(block.dtype, numpy.float64)
    if block.size:
        uv, singular_values, vo = scipy.linalg.svd(
            block, full_matrices=False, lapack_driver='gesdd')
        uo = vo.conj().T
    else:
        singular_values = numpy.zeros(0)
        uv = numpy.zeros((numpy.count_nonzero(viridx), 0), dtype=dtype)
        uo = numpy.zeros((numpy.count_nonzero(occidx), 0), dtype=dtype)

    def rotation(scale):
        transform = numpy.eye(nmo, dtype=dtype)
        if singular_values.size:
            cosine = numpy.cos(scale*singular_values) - 1
            sine = numpy.sin(scale*singular_values)
            transform[numpy.ix_(occidx, occidx)] += (
                uo*cosine).dot(uo.conj().T)
            transform[numpy.ix_(viridx, viridx)] += (
                uv*cosine).dot(uv.conj().T)
            transform[numpy.ix_(viridx, occidx)] = (
                uv*sine).dot(uo.conj().T)
            transform[numpy.ix_(occidx, viridx)] = -(
                uo*sine).dot(uv.conj().T)
        return transform

    def rotate(mo_coeff, scale):
        mo = numpy.array(mo_coeff, dtype=numpy.result_type(
            mo_coeff.dtype, dtype), copy=True)
        if singular_values.size:
            cosine = numpy.cos(scale*singular_values) - 1
            sine = numpy.sin(scale*singular_values)
            mo_occ = mo_coeff[:,occidx]
            mo_vir = mo_coeff[:,viridx]
            occ_active = mo_occ.dot(uo)
            vir_active = mo_vir.dot(uv)
            mo[:,occidx] += (occ_active*cosine
                             + vir_active*sine).dot(uo.conj().T)
            mo[:,viridx] += (vir_active*cosine
                             - occ_active*sine).dot(uv.conj().T)
        return mo

    rotation.rotate = rotate
    return rotation


def _rotation_path(mf, step, mo_occ, mo_coeff):
    if _is_uhf_mo(mo_coeff):
        kappa = _unpack_uhf(step, mo_occ)
        _force_rotation_degeneracy(mf, kappa, mo_coeff)
        paths = []
        for spin in range(2):
            occidx = numpy.asarray(mo_occ[spin]) > 0
            viridx = ~occidx
            block = kappa[spin][numpy.ix_(viridx, occidx)]
            paths.append(_grassmann_rotation_path(block, occidx))

        def rotation(scale):
            return numpy.asarray((paths[0](scale), paths[1](scale)))

        def rotate(mo_coeff, scale):
            return numpy.asarray((paths[0].rotate(mo_coeff[0], scale),
                                  paths[1].rotate(mo_coeff[1], scale)))

        rotation.rotate = rotate
        return rotation, _pack_uhf(kappa, mo_occ)

    kappa = hf.unpack_uniq_var(step, mo_occ)
    _force_rotation_degeneracy(mf, kappa, mo_coeff)
    effective_step = hf.pack_uniq_var(kappa, mo_occ)
    if mf.istype('ROHF'):
        return lambda scale: expmat(kappa*scale), effective_step

    occidx = mo_occ > 0
    viridx = ~occidx
    block = kappa[numpy.ix_(viridx, occidx)]
    return _grassmann_rotation_path(block, occidx), effective_step


def _split_rohf(dx, mo_occ):
    x = numpy.zeros((mo_occ.size, mo_occ.size), dtype=dx.dtype)
    mask = hf.uniq_var_indices(mo_occ)
    x[mask] = dx
    coreidx = mo_occ == 2
    openidx = mo_occ == 1
    viridx = mo_occ == 0
    return (x[numpy.ix_(openidx, coreidx)],
            x[numpy.ix_(viridx, coreidx)],
            x[numpy.ix_(viridx, openidx)])


def _join_rohf(blocks, mo_occ):
    coreidx = mo_occ == 2
    openidx = mo_occ == 1
    viridx = mo_occ == 0
    x = numpy.zeros((mo_occ.size, mo_occ.size), dtype=blocks[0].dtype)
    x[numpy.ix_(openidx, coreidx)] = blocks[0]
    x[numpy.ix_(viridx, coreidx)] = blocks[1]
    x[numpy.ix_(viridx, openidx)] = blocks[2]
    return x[hf.uniq_var_indices(mo_occ)]


def _transport_horizontal(dx, mo_occ, u, is_rohf=False):
    if isinstance(u, numpy.ndarray) and u.ndim == 3:
        out = []
        offset = 0
        for spin in range(2):
            occidx = numpy.asarray(mo_occ[spin]) > 0
            viridx = ~occidx
            size = numpy.count_nonzero(viridx) * numpy.count_nonzero(occidx)
            block = dx[offset:offset+size].reshape(
                numpy.count_nonzero(viridx), numpy.count_nonzero(occidx))
            uo = u[spin][numpy.ix_(occidx, occidx)]
            uv = u[spin][numpy.ix_(viridx, viridx)]
            out.append(uv.conj().T.dot(block).dot(uo).ravel())
            offset += size
        return numpy.hstack(out)

    if is_rohf:
        coreidx = mo_occ == 2
        openidx = mo_occ == 1
        viridx = mo_occ == 0
        uc = u[numpy.ix_(coreidx, coreidx)]
        uo = u[numpy.ix_(openidx, openidx)]
        uv = u[numpy.ix_(viridx, viridx)]
        oc, vc, vo = _split_rohf(dx, mo_occ)
        return _join_rohf((uo.conj().T.dot(oc).dot(uc),
                           uv.conj().T.dot(vc).dot(uc),
                           uv.conj().T.dot(vo).dot(uo)), mo_occ)

    occidx = mo_occ > 0
    viridx = ~occidx
    block = dx.reshape(numpy.count_nonzero(viridx),
                       numpy.count_nonzero(occidx))
    uo = u[numpy.ix_(occidx, occidx)]
    uv = u[numpy.ix_(viridx, viridx)]
    return uv.conj().T.dot(block).dot(uo).ravel()


def _rohf_commutator_tangent(kappa, tangent):
    k_oc, k_vc, k_vo = kappa
    v_oc, v_vc, v_vo = tangent
    return (-k_vo.conj().T.dot(v_vc) + v_vo.conj().T.dot(k_vc),
            k_vo.dot(v_oc) - v_vo.dot(k_oc),
            -k_vc.dot(v_oc.conj().T) + v_vc.dot(k_oc.conj().T))


def _transport_rohf_vertical(dx, step, mo_occ, tol=1e-8, max_cycle=12):
    if dx.size == 0 or not numpy.any(step):
        return dx.copy()
    kappa = _split_rohf(step, mo_occ)
    term = _split_rohf(dx, mo_occ)
    result = [block.copy() for block in term]
    coefficient = 1.
    for order in range(1, max_cycle + 1):
        term = _rohf_commutator_tangent(kappa, term)
        coefficient *= -.5 / order
        for result_block, term_block in zip(result, term):
            result_block += coefficient * term_block
        max_term = max((abs(block).max() for block in term if block.size),
                       default=0.)
        if max_term < tol:
            break
    return _join_rohf(result, mo_occ)


def _transport_history_horizontal(history, mo_occ, u, is_rohf=False):
    for i, (step, y) in enumerate(history):
        history[i] = (_transport_horizontal(step, mo_occ, u, is_rohf),
                      _transport_horizontal(y, mo_occ, u, is_rohf))


def _transport_history_rohf(history, step, mo_occ, tol, max_cycle):
    for i, (s, y) in enumerate(history):
        history[i] = (_transport_rohf_vertical(
            s, step, mo_occ, tol, max_cycle),
                      _transport_rohf_vertical(
            y, step, mo_occ, tol, max_cycle))


def _build_ewc_history(history, alpha, min_curvature):
    history_ewc = []
    for step, y0 in history:
        s = step * alpha
        y = y0 / alpha
        sy = _dot(s, y)
        scale = max(1., numpy.linalg.norm(s) * numpy.linalg.norm(y))
        if sy > min_curvature * scale:
            history_ewc.append((s, y, 1. / sy))
    return history_ewc


def _add_history(history, step, y, max_space, min_curvature):
    sy = _dot(step, y)
    if sy <= min_curvature * max(
            1., numpy.linalg.norm(step) * numpy.linalg.norm(y)):
        return False
    history.append((step.copy(), y.copy()))
    if len(history) > max_space:
        del history[0]
    return True


def _step_norm(step, norm_type):
    if step.size == 0:
        return 0.
    if norm_type == 'inf':
        return abs(step).max()
    if norm_type in ('2', 2):
        return numpy.linalg.norm(step)
    raise ValueError('Unsupported GDM step norm %s' % norm_type)


def _scale_step(step, max_stepsize, norm_type='inf'):
    norm_step = _step_norm(step, norm_type)
    capped = norm_step > max_stepsize
    if capped:
        step = step * (max_stepsize / norm_step)
        norm_step = max_stepsize
    return step, norm_step, capped


def _subspace_eigh(mo_coeff, fock_mo, subspaces, orbsym=None):
    nmo = mo_coeff.shape[1]
    u = numpy.eye(nmo, dtype=numpy.result_type(mo_coeff.dtype,
                                               fock_mo.dtype))
    for subspace in subspaces:
        labels = (numpy.unique(orbsym[subspace]) if orbsym is not None
                  else (None,))
        for label in labels:
            idx = subspace.copy()
            if label is not None:
                idx &= orbsym == label
            idx = numpy.flatnonzero(idx)
            if idx.size:
                _, rotation = scipy.linalg.eigh(
                    fock_mo[numpy.ix_(idx, idx)])
                u[numpy.ix_(idx, idx)] = rotation
    return mo_coeff.dot(u), u


def _rotate_fock_mo(fock_mo, u):
    if fock_mo.ndim == 3:
        return numpy.asarray((u[0].conj().T.dot(fock_mo[0]).dot(u[0]),
                              u[1].conj().T.dot(fock_mo[1]).dot(u[1])))
    fock1 = u.conj().T.dot(fock_mo).dot(u)
    if getattr(fock_mo, 'focka', None) is not None:
        focka = u.conj().T.dot(fock_mo.focka).dot(u)
        fockb = u.conj().T.dot(fock_mo.fockb).dot(u)
        fock1 = lib.tag_array(fock1, focka=focka, fockb=fockb)
    return fock1


def _pseudocanonicalize(mf, mo_coeff, mo_occ, fock, fock_mo=None):
    if fock_mo is None:
        fock_mo = _fock_mo(mo_coeff, fock)
    if not mf.gdm_pcanonicalization:
        if _is_uhf_mo(mo_coeff):
            nmo = mo_coeff[0].shape[1]
            u = numpy.asarray((numpy.eye(nmo, dtype=mo_coeff[0].dtype),
                               numpy.eye(nmo, dtype=mo_coeff[1].dtype)))
        else:
            u = numpy.eye(mo_coeff.shape[1], dtype=mo_coeff.dtype)
        return _mo_energy_from_fock_mo(fock_mo), mo_coeff, u, fock_mo

    if _is_uhf_mo(mo_coeff):
        orbsym = (uhf_symm.get_orbsym(mf.mol, mo_coeff)
                  if mf.mol.symmetry else (None, None))
        mo = []
        rotations = []
        for spin in range(2):
            occidx = numpy.asarray(mo_occ[spin]) > 0
            mo1, u1 = _subspace_eigh(
                mo_coeff[spin], fock_mo[spin], (occidx, ~occidx),
                orbsym[spin])
            mo.append(mo1)
            rotations.append(u1)
        mo = numpy.asarray(mo)
        if mf.mol.symmetry:
            mo = lib.tag_array(mo, orbsym=numpy.asarray(orbsym))
        rotations = numpy.asarray(rotations)
        fock_mo = _rotate_fock_mo(fock_mo, rotations)
        return (_mo_energy_from_fock_mo(fock_mo), mo, rotations,
                fock_mo)

    if mf.mol.symmetry:
        if mf.istype('GHF'):
            orbsym = ghf_symm.get_orbsym(mf.mol, mo_coeff)
        else:
            orbsym = hf_symm.get_orbsym(mf.mol, mo_coeff)
    else:
        orbsym = None

    if mf.istype('ROHF'):
        subspaces = (mo_occ == 2, mo_occ == 1, mo_occ == 0)
    else:
        occidx = mo_occ > 0
        subspaces = (occidx, ~occidx)
    mo, u = _subspace_eigh(mo_coeff, fock_mo, subspaces, orbsym)
    if orbsym is not None:
        mo = lib.tag_array(mo, orbsym=orbsym)
    fock_mo = _rotate_fock_mo(fock_mo, u)
    return _mo_energy_from_fock_mo(fock_mo), mo, u, fock_mo


def _accepted(e_tot, de, armijo, scale, e1):
    return numpy.isfinite(e1) and e1 <= e_tot + armijo * scale * de


def _stagnated(norms, ncycle):
    if ncycle <= 0 or len(norms) <= ncycle:
        return False
    return norms[-1] > .8 * norms[-ncycle-1]


def _tag_orbital_symmetry(mf, mo, mo_coeff):
    if not mf.mol.symmetry:
        return mo
    if _is_uhf_mo(mo_coeff):
        orbsym = uhf_symm.get_orbsym(mf.mol, mo_coeff)
    elif mf.istype('GHF'):
        orbsym = ghf_symm.get_orbsym(mf.mol, mo_coeff)
    else:
        orbsym = hf_symm.get_orbsym(mf.mol, mo_coeff)
    return lib.tag_array(mo, orbsym=orbsym)


def _evaluate_step(mf, mo_coeff, mo_occ, dm, vhf, h1e, s1e, step, log,
                   rotation=None, scale=1.):
    mol = mf.mol
    scf_summary = mf.scf_summary
    mf.scf_summary = scf_summary.copy()
    try:
        if rotation is None:
            u = mf.update_rotate_matrix(step, mo_occ, mo_coeff=mo_coeff)
            mo1 = mf.rotate_mo(mo_coeff, u, log)
        elif hasattr(rotation, 'rotate'):
            mo1 = rotation.rotate(mo_coeff, scale)
            mo1 = _tag_orbital_symmetry(mf, mo1, mo_coeff)
        else:
            mo1 = mf.rotate_mo(mo_coeff, rotation(scale), log)
        dm1 = mf.make_rdm1(mo1, mo_occ)
        vhf1 = mf.get_veff(mol, dm1, dm_last=dm, vhf_last=vhf)
        e1 = mf.energy_tot(dm1, h1e, vhf1)
        fock1 = mf.get_fock(h1e, s1e, vhf1, dm1, level_shift_factor=0)
        fock_mo1 = _fock_mo(mo1, fock1)
        g1, h_diag1 = mf.gen_g_hdiag(
            mo1, mo_occ, fock1, h1e, fock_mo1)
        trial_summary = mf.scf_summary
    finally:
        mf.scf_summary = scf_summary
    return (e1, mo1, dm1, vhf1, fock1, g1, h_diag1, fock_mo1,
            trial_summary)


def _armijo_search(mf, mo_coeff, mo_occ, dm, vhf, h1e, s1e, e_tot,
                   g_orb, step, log, max_scale=1.):
    rotation, step = _rotation_path(mf, step, mo_occ, mo_coeff)
    de = _directional_derivative(g_orb, step)
    if not numpy.isfinite(de) or de >= 0:
        return None, 0., None, 0, 'armijo-failed'

    scale = min(1., max_scale)
    for ntrial in range(1, mf.gdm_step_control_max_cycle + 1):
        trial_step = step * scale
        out = _evaluate_step(mf, mo_coeff, mo_occ, dm, vhf, h1e, s1e,
                             trial_step, log, rotation, scale)
        if _accepted(e_tot, de, mf.gdm_wolfe_c1, scale, out[0]):
            return trial_step, scale, out, ntrial, 'armijo'
        log.debug('GDM Armijo search rejects scale %g E=%.15g',
                  scale, out[0])
        scale *= .5
    return None, scale, None, mf.gdm_step_control_max_cycle, 'armijo-failed'


def _zoom_scale(lo, hi):
    alo, flo, dlo = lo[:3]
    ahi, fhi = hi[:2]
    delta = ahi - alo
    denom = 2 * (fhi - flo - dlo * delta)
    if numpy.isfinite(denom) and denom > 0:
        scale = alo - dlo * delta**2 / denom
    else:
        scale = (alo + ahi) * .5
    lower = min(alo, ahi) + .1 * abs(delta)
    upper = max(alo, ahi) - .1 * abs(delta)
    if not numpy.isfinite(scale) or scale <= lower or scale >= upper:
        scale = (alo + ahi) * .5
    return scale


def _strong_wolfe_search(mf, mo_coeff, mo_occ, dm, vhf, h1e, s1e,
                         e_tot, g_orb, step, log, max_scale=1.):
    rotation, step = _rotation_path(mf, step, mo_occ, mo_coeff)
    der0 = _directional_derivative(g_orb, step)
    if not numpy.isfinite(der0) or der0 >= 0:
        return None, 0., None, 0, 'wolfe-failed'

    c1 = mf.gdm_wolfe_c1
    c2 = mf.gdm_wolfe_c2
    max_cycle = mf.gdm_step_control_max_cycle
    ntrial = 0

    def evaluate(scale):
        nonlocal ntrial
        out = _evaluate_step(
            mf, mo_coeff, mo_occ, dm, vhf, h1e, s1e, step*scale, log,
            rotation, scale)
        ntrial += 1
        derivative = _directional_derivative(out[5], step)
        return scale, out[0], derivative, out

    def zoom(lo, hi):
        nonlocal ntrial
        while ntrial < max_cycle:
            trial = evaluate(_zoom_scale(lo, hi))
            scale, energy, derivative, out = trial
            if (not _accepted(e_tot, der0, c1, scale, energy)
                    or energy >= lo[1]):
                hi = trial
            else:
                if abs(derivative) <= -c2 * der0:
                    return step*scale, scale, out, ntrial, 'strong-wolfe'
                if derivative * (hi[0] - lo[0]) >= 0:
                    hi = lo
                lo = trial
            if abs(hi[0] - lo[0]) < numpy.finfo(float).eps:
                break
        return None, 0., None, ntrial, 'wolfe-failed'

    previous = (0., e_tot, der0, None)
    scale = min(1., max_scale)
    iteration = 0
    while ntrial < max_cycle:
        trial = evaluate(scale)
        energy, derivative, out = trial[1:]
        if (not _accepted(e_tot, der0, c1, scale, energy)
                or iteration > 0 and energy >= previous[1]):
            return zoom(previous, trial)
        if abs(derivative) <= -c2 * der0:
            return step*scale, scale, out, ntrial, 'strong-wolfe'
        if derivative >= 0:
            return zoom(trial, previous)
        if scale >= max_scale:
            return step*scale, scale, out, ntrial, 'wolfe-boundary'
        previous = trial
        scale = min(2*scale, max_scale)
        iteration += 1
    return None, 0., None, ntrial, 'wolfe-failed'


def _dogleg_step(p_cauchy, p_newton, trust_radius):
    norm_newton = numpy.linalg.norm(p_newton)
    if norm_newton <= trust_radius:
        return p_newton.copy(), 'newton'
    norm_cauchy = numpy.linalg.norm(p_cauchy)
    if norm_cauchy >= trust_radius:
        if norm_cauchy < numpy.finfo(float).tiny:
            return numpy.zeros_like(p_cauchy), 'cauchy'
        return p_cauchy * (trust_radius / norm_cauchy), 'cauchy'

    direction = p_newton - p_cauchy
    a = _dot(direction, direction)
    b = 2 * _dot(p_cauchy, direction)
    c = _dot(p_cauchy, p_cauchy) - trust_radius**2
    discriminant = max(0., b*b - 4*a*c)
    tau = (-b + numpy.sqrt(discriminant)) / (2*a)
    tau = max(0., min(1., tau))
    return p_cauchy + tau*direction, 'dogleg'


def _dogleg_search(mf, mo_coeff, mo_occ, dm, vhf, h1e, s1e,
                   e_tot, g_orb, h_diag, p_cauchy, p_newton,
                   trust_radius, log):
    noise = 32 * numpy.finfo(float).eps * (1 + abs(e_tot))
    norm_g = _norm_gorb(g_orb)
    ntrial = 0
    for _ in range(mf.gdm_trust_max_cycle):
        step, leg = _dogleg_step(p_cauchy, p_newton, trust_radius)
        step, _, _ = _scale_step(step, mf.max_stepsize, 'inf')
        rotation, step = _rotation_path(mf, step, mo_occ, mo_coeff)
        predicted = -2 * (_dot(g_orb, step)
                          + .5 * _dot(step, h_diag*step))
        if not numpy.isfinite(predicted) or predicted <= 0:
            trust_radius = max(.25*trust_radius, mf.gdm_trust_min)
            ntrial += 1
            continue

        out = _evaluate_step(mf, mo_coeff, mo_occ, dm, vhf, h1e, s1e,
                             step, log, rotation)
        ntrial += 1
        actual = e_tot - out[0]
        if predicted > 4*noise:
            rho = actual / predicted
            if rho < .25:
                trust_radius = max(.25*trust_radius, mf.gdm_trust_min)
            elif rho > .75:
                trust_radius = min(2*numpy.linalg.norm(step),
                                   mf.gdm_trust_max)
            if rho > mf.gdm_trust_eta:
                return (step, 1., out, ntrial,
                        '%s rho=%.2f' % (leg, rho), trust_radius)
        else:
            no_energy_rise = out[0] <= e_tot + noise
            if no_energy_rise and _norm_gorb(out[5]) <= norm_g:
                trust_radius = min(2*numpy.linalg.norm(step),
                                   mf.gdm_trust_max)
                return (step, 1., out, ntrial, leg+' grad-step',
                        trust_radius)
            trust_radius = max(.25*trust_radius, mf.gdm_trust_min)

        if trust_radius <= mf.gdm_trust_min:
            break
    return None, 0., None, ntrial, 'dogleg-failed', mf.gdm_trust_init


def _select_step(mf, mode, mo_coeff, mo_occ, dm, vhf, h1e, s1e,
                 e_tot, g_orb, step, log, max_scale=1.):
    if mode == 'strong_wolfe':
        return _strong_wolfe_search(
            mf, mo_coeff, mo_occ, dm, vhf, h1e, s1e, e_tot,
            g_orb, step, log, max_scale)
    if mode == 'armijo':
        return _armijo_search(
            mf, mo_coeff, mo_occ, dm, vhf, h1e, s1e, e_tot,
            g_orb, step, log, max_scale)
    if mode == 'none':
        rotation, step = _rotation_path(mf, step, mo_occ, mo_coeff)
        out = _evaluate_step(mf, mo_coeff, mo_occ, dm, vhf, h1e, s1e,
                             step, log, rotation)
        return step, 1., out, 1, 'none'
    raise ValueError('Unsupported GDM step controller %s' % mode)


def kernel(mf, mo_coeff=None, mo_occ=None, dm=None,
           conv_tol=1e-10, conv_tol_grad=None, max_cycle=50, dump_chk=True,
           callback=None, verbose=logger.NOTE):
    cput0 = (logger.process_clock(), logger.perf_counter())
    log = logger.new_logger(mf, verbose)
    mol = mf.mol

    if conv_tol_grad is None:
        conv_tol_grad = numpy.sqrt(conv_tol)
        log.info('Set conv_tol_grad to %g', conv_tol_grad)

    h1e = mf.get_hcore(mol)
    s1e = mf.get_ovlp(mol)
    x_orth = mf.check_linear_dependency(s1e, log)

    if mo_coeff is not None and mo_occ is not None:
        dm = mf.make_rdm1(mo_coeff, mo_occ)
        vhf = mf.get_veff(mol, dm)
        fock = mf.get_fock(h1e, s1e, vhf, dm, level_shift_factor=0)
        fock_mo = _fock_mo(mo_coeff, fock)
        mo_energy, mo_coeff, _, fock_mo = _pseudocanonicalize(
            mf, mo_coeff, mo_occ, fock, fock_mo)
    else:
        if dm is None:
            logger.debug(mf, 'Initial guess density matrix is not given. '
                         'Generating initial guess from %s', mf.init_guess)
            dm = mf.get_init_guess(mol, mf.init_guess)
        vhf = mf.get_veff(mol, dm)
        fock = mf.get_fock(h1e, s1e, vhf, dm, level_shift_factor=0)
        mo_energy, mo_coeff = mf.eig(fock, s1e, x=x_orth)
        mo_occ = mf.get_occ(mo_energy, mo_coeff)
        dm_last = dm
        dm = mf.make_rdm1(mo_coeff, mo_occ)
        vhf = mf.get_veff(mol, dm, dm_last=dm_last, vhf_last=vhf)
        fock = mf.get_fock(h1e, s1e, vhf, dm, level_shift_factor=0)
        fock_mo = _fock_mo(mo_coeff, fock)
        mo_energy, mo_coeff, _, fock_mo = _pseudocanonicalize(
            mf, mo_coeff, mo_occ, fock, fock_mo)

    mf.mo_coeff, mf.mo_occ = mo_coeff, mo_occ
    e_tot = mf.energy_tot(dm, h1e, vhf)
    g_orb, h_diag = mf.gen_g_hdiag(
        mo_coeff, mo_occ, fock, h1e, fock_mo)
    norm_gorb = _norm_gorb(g_orb)
    log.info('Initial guess E= %.15g  |g|= %g', e_tot, norm_gorb)

    if mf.max_cycle <= 0:
        return False, e_tot, mo_energy, mo_coeff, mo_occ

    if dump_chk and mf.chkfile:
        chkfile.save_mol(mol, mf.chkfile)

    scf_conv = g_orb.size == 0 or norm_gorb < conv_tol_grad
    history = []
    grad_norms = [norm_gorb]
    max_stepsize = mf.max_stepsize
    last_delta_e = 0.
    step_control = _step_control_mode(mf)
    trust_radius = mf.gdm_trust_init
    cput1 = log.timer('initializing GDM SCF', *cput0)

    for cycle in range(max_cycle):
        if scf_conv:
            break

        last_hf_e = e_tot
        shift = _hdiag_shift(mf, last_delta_e)
        shift_weights = mf.gen_hdiag_shift_weights(mo_occ)
        shifted_hdiag = h_diag + shift_weights * shift
        alpha_floor = max(mf.gdm_ewc_floor,
                          numpy.sqrt(mf.gdm_hdiag_floor))
        alpha = _ewc_scale(shifted_hdiag, alpha_floor)
        h_diag_precond = alpha**2
        g_ewc = g_orb / alpha
        history_ewc = _build_ewc_history(history, alpha,
                                         mf.gdm_min_curvature)
        step_ewc = _lbfgs_direction(g_ewc, history_ewc)
        step = step_ewc / alpha
        p_cauchy = -g_orb / h_diag_precond
        if (not numpy.all(numpy.isfinite(step))
                or _dot(g_orb, step) >= -mf.gdm_min_curvature):
            history = []
            step = p_cauchy.copy()

        if step_control == 'dogleg':
            (accepted_step, ls_scale, out, ls_trials, control_note,
             trust_radius) = _dogleg_search(
                mf, mo_coeff, mo_occ, dm, vhf, h1e, s1e, e_tot,
                g_orb, h_diag_precond, p_cauchy, step,
                trust_radius, log)
            capped = (accepted_step is not None and
                      _step_norm(accepted_step, 'inf')
                      >= mf.max_stepsize*(1-1e-12))
        else:
            step, norm_step, capped = _scale_step(
                step, max_stepsize, mf.gdm_step_norm)
            if norm_step < mf.gdm_step_tol:
                log.warn('GDM step size below threshold %g',
                         mf.gdm_step_tol)
                break
            if capped:
                max_scale = 1.
            else:
                max_scale = max_stepsize / max(
                    norm_step, numpy.finfo(float).tiny)
            (accepted_step, ls_scale, out, ls_trials,
             control_note) = _select_step(
                mf, step_control, mo_coeff, mo_occ, dm, vhf, h1e,
                s1e, e_tot, g_orb, step, log, max_scale)

        if out is None:
            history = []
            trust_radius = mf.gdm_trust_init
            step, norm_step, capped = _scale_step(
                p_cauchy, max_stepsize, mf.gdm_step_norm)
            (accepted_step, ls_scale, out, fallback_trials,
             fallback_note) = _armijo_search(
                mf, mo_coeff, mo_occ, dm, vhf, h1e, s1e, e_tot,
                g_orb, step, log)
            ls_trials += fallback_trials
            control_note += '+' + fallback_note
        if out is None:
            log.warn('GDM step controller failed to find a downhill step')
            break

        (e_tot, mo_coeff, dm, vhf, fock, g1, h_diag1, fock_mo,
         mf.scf_summary) = out
        is_rohf = mf.istype('ROHF')
        if is_rohf:
            _transport_history_rohf(
                history, accepted_step, mo_occ, mf.gdm_transport_tol,
                mf.gdm_transport_max_cycle)
            g0 = _transport_rohf_vertical(
                g_orb, accepted_step, mo_occ, mf.gdm_transport_tol,
                mf.gdm_transport_max_cycle)
            accepted_step = _transport_rohf_vertical(
                accepted_step, accepted_step, mo_occ,
                mf.gdm_transport_tol, mf.gdm_transport_max_cycle)
        else:
            g0 = g_orb

        mo_energy, mo_coeff, u, fock_mo = _pseudocanonicalize(
            mf, mo_coeff, mo_occ, fock, fock_mo)
        _transport_history_horizontal(history, mo_occ, u, is_rohf)
        accepted_step = _transport_horizontal(
            accepted_step, mo_occ, u, is_rohf)
        g0 = _transport_horizontal(g0, mo_occ, u, is_rohf)
        if mf.gdm_pcanonicalization:
            g1, h_diag1 = mf.gen_g_hdiag(
                mo_coeff, mo_occ, fock, h1e, fock_mo)
        norm_gorb = _norm_gorb(g1)
        added = _add_history(
            history, accepted_step, g1-g0, mf.gdm_space,
            mf.gdm_min_curvature)
        if not added:
            log.debug('GDM skips L-BFGS pair with non-positive curvature')

        if (mf.gdm_adaptive_stepsize and capped and ls_scale > .99
                and max_stepsize < mf.gdm_max_stepsize):
            max_stepsize = min(mf.gdm_max_stepsize, max_stepsize * 1.5)

        log.info('cycle= %d E= %.15g  delta_E= %g  |g|= %g  '
                 '|step|= %g  |step|max= %g  control=%s %.3g/%d  hist= %d',
                 cycle+1, e_tot, e_tot-last_hf_e, norm_gorb,
                 numpy.linalg.norm(accepted_step),
                 _step_norm(accepted_step, 'inf'), control_note, ls_scale,
                 ls_trials, len(history))
        cput1 = log.timer('cycle= %d'%(cycle+1), *cput1)

        g_orb = g1
        h_diag = h_diag1
        last_delta_e = e_tot - last_hf_e
        mf.cycles = cycle + 1
        grad_norms.append(norm_gorb)
        if history and _stagnated(grad_norms, mf.gdm_stagnation_cycles):
            log.debug('GDM resets L-BFGS history after %d stagnant cycles',
                      mf.gdm_stagnation_cycles)
            history = []

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
    if mf.canonicalization:
        log.info('Canonicalize SCF orbitals')
        mo_energy, mo_coeff = mf.canonicalize(mo_coeff, mo_occ, fock)
        if dump_chk:
            mf.dump_chk(locals())
    else:
        mo_energy = _mo_energy_from_fock(mo_coeff, fock)

    log.info('GDM macro X = %d  E=%.15g  |g|= %g',
             getattr(mf, 'cycles', 0), e_tot, norm_gorb)
    return scf_conv, e_tot, mo_energy, mo_coeff, mo_occ


def _run_diis_start(mf, dm0, log):
    scf_obj = mf.undo_gdm()

    scf_obj.max_cycle = mf.gdm_diis_max_cycle
    scf_obj.conv_tol_grad = mf.gdm_diis_switch_tol
    scf_obj.conv_tol = max(mf.conv_tol, mf.gdm_diis_switch_tol**2)
    if hasattr(scf_obj, 'conv_check'):
        scf_obj.conv_check = False

    log.info('Run DIIS pre-iterations before GDM: max_cycle=%d '
             'switch |g|=%g', mf.gdm_diis_max_cycle,
             mf.gdm_diis_switch_tol)
    scf_obj.kernel(dm0=dm0)

    if scf_obj.mo_coeff is None or scf_obj.mo_occ is None:
        raise RuntimeError('DIIS pre-iterations did not produce orbitals')
    return scf_obj.mo_coeff, scf_obj.mo_occ


class _GDM_SCF:
    '''Geometric direct minimization mixin.

    Selected attributes:

        gdm_space : int
            L-BFGS history size. Default is 20.
        gdm_step_control : str
            Step controller. ``'strong_wolfe'`` (default), ``'dogleg'``,
            ``'armijo'``, and ``'none'`` are supported.
        max_stepsize : float
            Maximum orbital-rotation component. Default is 0.5.
        gdm_hdiag_shift : float or str
            Bacskay energy shift. ``'auto'`` uses the previous-cycle energy
            change. Default is ``'auto'``.
        gdm_pcanonicalization : bool
            Update energy-weighted coordinates in pseudocanonical orbitals on
            every accepted cycle. Default is True.
        canonicalization : bool
            Whether to canonicalize final orbitals. Default is True.
    '''

    __name_mixin__ = 'GDM'

    gdm_space = getattr(__config__, 'soscf_gdm_GDM_space', 20)
    max_stepsize = getattr(__config__, 'soscf_gdm_GDM_max_stepsize', .5)
    canonicalization = getattr(__config__, 'soscf_gdm_GDM_canonicalization', True)
    gdm_step_control = getattr(
        __config__, 'soscf_gdm_GDM_step_control', 'strong_wolfe')
    gdm_line_search = getattr(
        __config__, 'soscf_gdm_GDM_line_search', None)
    gdm_step_control_max_cycle = getattr(
        __config__, 'soscf_gdm_GDM_step_control_max_cycle', 12)
    gdm_wolfe_c1 = getattr(__config__, 'soscf_gdm_GDM_wolfe_c1', 1e-4)
    gdm_wolfe_c2 = getattr(__config__, 'soscf_gdm_GDM_wolfe_c2', .9)
    gdm_trust_init = getattr(
        __config__, 'soscf_gdm_GDM_trust_init', .5)
    gdm_trust_max = getattr(__config__, 'soscf_gdm_GDM_trust_max', 2.)
    gdm_trust_min = getattr(__config__, 'soscf_gdm_GDM_trust_min', 1e-8)
    gdm_trust_eta = getattr(__config__, 'soscf_gdm_GDM_trust_eta', .1)
    gdm_trust_max_cycle = getattr(
        __config__, 'soscf_gdm_GDM_trust_max_cycle', 15)
    gdm_hdiag_floor = getattr(__config__, 'soscf_gdm_GDM_hdiag_floor', 1e-4)
    gdm_ewc_floor = getattr(__config__, 'soscf_gdm_GDM_ewc_floor', .1)
    gdm_hdiag_shift = getattr(__config__, 'soscf_gdm_GDM_hdiag_shift', 'auto')
    gdm_min_curvature = getattr(__config__, 'soscf_gdm_GDM_min_curvature', 1e-10)
    gdm_step_tol = getattr(__config__, 'soscf_gdm_GDM_step_tol', 1e-12)
    gdm_step_norm = getattr(__config__, 'soscf_gdm_GDM_step_norm', 'inf')
    gdm_adaptive_stepsize = getattr(
        __config__, 'soscf_gdm_GDM_adaptive_stepsize', True)
    gdm_max_stepsize = getattr(__config__, 'soscf_gdm_GDM_max_stepsize_cap', .5)
    gdm_stagnation_cycles = getattr(
        __config__, 'soscf_gdm_GDM_stagnation_cycles', 10)
    gdm_diis_start = getattr(__config__, 'soscf_gdm_GDM_diis_start', False)
    gdm_diis_max_cycle = getattr(
        __config__, 'soscf_gdm_GDM_diis_max_cycle', 50)
    gdm_diis_switch_tol = getattr(
        __config__, 'soscf_gdm_GDM_diis_switch_tol', 1e-2)
    gdm_pcanonicalization = getattr(
        __config__, 'soscf_gdm_GDM_pcanonicalization', True)
    gdm_transport_tol = getattr(
        __config__, 'soscf_gdm_GDM_transport_tol', 1e-8)
    gdm_transport_max_cycle = getattr(
        __config__, 'soscf_gdm_GDM_transport_max_cycle', 12)

    _keys = {
        'gdm_space', 'max_stepsize', 'canonicalization',
        'gdm_step_control', 'gdm_line_search', 'gdm_hdiag_floor',
        'gdm_step_control_max_cycle', 'gdm_wolfe_c1', 'gdm_wolfe_c2',
        'gdm_trust_init', 'gdm_trust_max', 'gdm_trust_min',
        'gdm_trust_eta', 'gdm_trust_max_cycle',
        'gdm_ewc_floor', 'gdm_hdiag_shift', 'gdm_min_curvature',
        'gdm_step_tol', 'gdm_step_norm', 'gdm_adaptive_stepsize',
        'gdm_max_stepsize', 'gdm_stagnation_cycles', 'gdm_diis_start',
        'gdm_diis_max_cycle', 'gdm_diis_switch_tol',
        'gdm_pcanonicalization', 'gdm_transport_tol',
        'gdm_transport_max_cycle',
    }

    def __init__(self, mf):
        self.__dict__.update(mf.__dict__)
        self._scf = mf

    def undo_gdm(self):
        '''Remove the GDM mixin.'''
        obj = lib.view(self, lib.drop_class(self.__class__, _GDM_SCF))
        del obj._scf
        return obj

    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
        step_control = _step_control_mode(self)
        log.info('\n')
        super().dump_flags(verbose)
        log.info('******** %s GDM solver flags ********', self._scf.__class__)
        log.info('SCF tol = %g', self.conv_tol)
        log.info('conv_tol_grad = %s', self.conv_tol_grad)
        log.info('max. SCF cycles = %d', self.max_cycle)
        log.info('direct_scf = %s', self.direct_scf)
        if self.direct_scf:
            log.info('direct_scf_tol = %g', self.direct_scf_tol)
        if self.chkfile:
            log.info('chkfile to save SCF result = %s', self.chkfile)
        log.info('gdm_space = %d', self.gdm_space)
        log.info('max_stepsize = %g', self.max_stepsize)
        log.info('gdm_step_norm = %s', self.gdm_step_norm)
        log.info('gdm_step_control = %s', step_control)
        log.info('gdm_step_control_max_cycle = %d',
                 self.gdm_step_control_max_cycle)
        if step_control == 'strong_wolfe':
            log.info('gdm_wolfe_c1 = %g', self.gdm_wolfe_c1)
            log.info('gdm_wolfe_c2 = %g', self.gdm_wolfe_c2)
        elif step_control == 'dogleg':
            log.info('gdm_trust_init = %g', self.gdm_trust_init)
            log.info('gdm_trust_max = %g', self.gdm_trust_max)
            log.info('gdm_trust_min = %g', self.gdm_trust_min)
            log.info('gdm_trust_eta = %g', self.gdm_trust_eta)
        log.info('gdm_hdiag_floor = %g', self.gdm_hdiag_floor)
        log.info('gdm_ewc_floor = %g', self.gdm_ewc_floor)
        log.info('gdm_hdiag_shift = %s', self.gdm_hdiag_shift)
        log.info('gdm_stagnation_cycles = %d', self.gdm_stagnation_cycles)
        log.info('gdm_pcanonicalization = %s', self.gdm_pcanonicalization)
        log.info('gdm_transport_tol = %g', self.gdm_transport_tol)
        log.info('gdm_transport_max_cycle = %d',
                 self.gdm_transport_max_cycle)
        if self.gdm_diis_start:
            log.info('gdm_diis_max_cycle = %d', self.gdm_diis_max_cycle)
            log.info('gdm_diis_switch_tol = %g', self.gdm_diis_switch_tol)
        log.info('canonicalization = %s', self.canonicalization)
        log.info('max_memory %d MB (current use %d MB)',
                 self.max_memory, lib.current_memory()[0])
        return self

    def build(self, mol=None):
        return super().build(mol)

    def reset(self, mol=None):
        return super().reset(mol)

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

        if self.gdm_diis_start and (mo_coeff is None or mo_occ is None):
            log = logger.new_logger(self, self.verbose)
            mo_coeff, mo_occ = _run_diis_start(self, dm0, log)
            dm0 = None

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
        mol = self.mol
        h1e = self.get_hcore(mol)
        s1e = self.get_ovlp(mol)
        vhf = self.get_veff(mol, dm)
        fock = self.get_fock(h1e, s1e, vhf, dm)
        mo_energy, mo_coeff = self.eig(fock, s1e)
        mo_occ = self.get_occ(mo_energy, mo_coeff)
        return mo_coeff, mo_occ

    gen_g_hop = newton_ah.gen_g_hop_rhf
    gen_g_hdiag = gen_g_hdiag_rhf
    gen_hdiag_shift_weights = staticmethod(_hdiag_shift_weights_rhf)

    def update_rotate_matrix(self, dx, mo_occ, u0=1, mo_coeff=None):
        rotation, _ = _rotation_path(self, dx, mo_occ, mo_coeff)
        u = rotation(1.)
        if isinstance(u0, int) and u0 == 1:
            return u
        return numpy.dot(u0, u)

    def rotate_mo(self, mo_coeff, u, log=None):
        mo = numpy.dot(mo_coeff, u)
        if self.mol.symmetry:
            orbsym = hf_symm.get_orbsym(self.mol, mo_coeff)
            mo = lib.tag_array(mo, orbsym=orbsym)
        return mo

    def density_fit(self, auxbasis=None, with_df=None, only_dfj=False):
        obj = self.undo_gdm().density_fit(auxbasis, with_df, only_dfj).gdm()
        return self._copy_gdm_settings(obj)

    def to_gpu(self):
        obj = self.undo_gdm().to_gpu().gdm()
        return self._copy_gdm_settings(obj)

    def _copy_gdm_settings(self, obj):
        for key in self._keys:
            setattr(obj, key, getattr(self, key))
        return obj


class _GDMROHF(_GDM_SCF):
    gen_g_hop = newton_ah.gen_g_hop_rohf
    gen_g_hdiag = gen_g_hdiag_rohf
    gen_hdiag_shift_weights = staticmethod(_hdiag_shift_weights_rohf)


class _GDMUHF(_GDM_SCF):
    gen_g_hop = newton_ah.gen_g_hop_uhf
    gen_g_hdiag = gen_g_hdiag_uhf
    gen_hdiag_shift_weights = staticmethod(_hdiag_shift_weights_uhf)

    def update_rotate_matrix(self, dx, mo_occ, u0=1, mo_coeff=None):
        rotation, _ = _rotation_path(self, dx, mo_occ, mo_coeff)
        u = rotation(1.)
        if isinstance(u0, int) and u0 == 1:
            return u
        return numpy.asarray((numpy.dot(u0[0], u[0]),
                              numpy.dot(u0[1], u[1])))

    def rotate_mo(self, mo_coeff, u, log=None):
        mo = numpy.asarray((numpy.dot(mo_coeff[0], u[0]),
                            numpy.dot(mo_coeff[1], u[1])))
        if self.mol.symmetry:
            orbsym = uhf_symm.get_orbsym(self.mol, mo_coeff)
            mo = lib.tag_array(mo, orbsym=orbsym)
        return mo

    def spin_square(self, mo_coeff=None, s=None):
        if mo_coeff is None:
            mo_coeff = (self.mo_coeff[0][:,self.mo_occ[0]>0],
                        self.mo_coeff[1][:,self.mo_occ[1]>0])
        return super().spin_square(mo_coeff, s)

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
    gen_hdiag_shift_weights = staticmethod(_hdiag_shift_weights_ghf)

    def rotate_mo(self, mo_coeff, u, log=None):
        mo = numpy.dot(mo_coeff, u)
        if self.mol.symmetry:
            orbsym = ghf_symm.get_orbsym(self.mol, mo_coeff)
            mo = lib.tag_array(mo, orbsym=orbsym)
        return mo


class _GDMRHF(_GDM_SCF):
    gen_g_hop = newton_ah.gen_g_hop_rhf
    gen_g_hdiag = gen_g_hdiag_rhf
    gen_hdiag_shift_weights = staticmethod(_hdiag_shift_weights_rhf)


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


def diis_gdm(mf):
    '''Hybrid DIIS then GDM SCF solver.'''
    mf = gdm(mf)
    mf.gdm_diis_start = True
    return mf


def remove_gdm(mf):
    '''Remove the GDM decorator.'''
    return mf.undo_gdm()
