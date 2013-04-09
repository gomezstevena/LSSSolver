#!/usr/bin/env python
import numpy as np
import LSSSolver as LSS
from LSSSolver.lss import VCycleKrylov
from LSSSolver.krylov import conjGrad
from LSSSolver import isoturb
import os, time



if __name__ == '__main__':
    k = 16
    nu = 0.001
    dt = np.pi / 16
    Nt = 32
    M = 4*k*k*(k+1) * 6
    tol = 1e-6

    tlvls = int( np.log2(Nt) - 2 )

    lvls = 3*[ (False,True) ]  + tlvls*[ (True,False) ]


    tol = 1e-6
    DIR = '/home/gomezs/IsoTurbData/'


    ftraj = os.path.join( DIR, 'traj_2048.dat')
    shape = (Nt, M)

    A = isoturb.parallel.IF3DParallel(k, nu, dt, ftraj, shape=shape, top=True)

    rhs = A.perturbation.copy()

    vcycle = VCycleKrylov(A, lvls, skip=5, pre_iters=5, post_iters=10, tol=tol)


    nr = A.normf(rhs)
    if LSS.parallel.MASTER:
        print 'Initial res = {:.3e}'.format( nr )

    
    call = LSS.callbacks.LogCallback(A, rhs)

    w, err = conjGrad( A, rhs, dot=A.dot, callback=call, tol=tol, M=vcycle, maxiter=20 )

    

