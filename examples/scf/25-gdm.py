#!/usr/bin/env python
#

from pyscf import gto
from pyscf import scf

'''
Geometric direct minimization SCF by decorating the mean-field object with
.gdm().

GDM optimizes orbital rotations directly.  It supports RHF, UHF, ROHF, GHF,
and the corresponding RKS, UKS, ROKS DFT objects.
'''

mol = gto.M(
    verbose = 0,
    atom = '''
    O 0 0 0
    H 0 -0.757 0.587
    H 0  0.757 0.587
    ''',
    basis = 'ccpvdz',
)

mf = scf.RHF(mol).gdm()
energy = mf.kernel()
print('RHF/GDM E = %.12f, ref = -76.026765672992' % energy)

mf = scf.RKS(mol)
mf.xc = 'pbe,pbe'
mf = mf.gdm()
energy = mf.kernel()
print('RKS/GDM E = %.12f' % energy)

mol = gto.M(
    verbose = 0,
    atom = 'O 0 0 0; H 0 0 0.97',
    basis = 'ccpvdz',
    spin = 1,
)

mf = scf.UKS(mol)
mf.xc = 'pbe,pbe'
mf = scf.gdm(mf)
energy = mf.kernel()
print('UKS/GDM E = %.12f' % energy)
