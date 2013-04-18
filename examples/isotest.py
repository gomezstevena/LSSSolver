#!/usr/bin/env python
import numpy as np
import LSSSolver as LSS
from LSSSolver.lss import VCycleKrylov
from LSSSolver.krylov import conjGrad
from LSSSolver import isoturb
import os, sys, datetime



if __name__ == '__main__':
    k = 16
    nu = 0.001
    dt = np.pi / 16
    Nt = int(sys.argv[1]) if len(sys.argv)>1 else 32
    M = 4*k*k*(k+1) * 6
    tol = 1e-6

    Ntlvls = int( np.log2( Nt / LSS.parallel.SIZE) ) - 1

    lvls = 3*[ (False,True) ]  + Ntlvls*[ (True,False) ]
    lvls.reverse()


    tol = 1e-6
    DIR = '/home/gomezs/IsoTurbData/'
    DIR = '/master/home/gomezs/lss/IsoTurbData/'


    ftraj = os.path.join( DIR, 'traj_2048.dat')
    shape = (Nt, M)

    A = isoturb.parallel.IF3DParallel(k, nu, dt, ftraj, shape=shape, top=True)

    rhs = A.perturbation.copy()

    vcycle = VCycleKrylov(A, lvls, skip=15, pre_iters=5, post_iters=15, tol=tol)


    nrhs = A.normf(rhs)
    if LSS.parallel.MASTER:
        print 'Initial res = {:.3e}'.format( nrhs )
        print 'Number of time levels =', Ntlvls

    
    call = LSS.callbacks.LogCallback(A, rhs)

    w, err = conjGrad( A, rhs, dot=A.dot, callback=call, tol=tol, M=vcycle, maxiter=20 )
    
    

    res = rhs - A*w
    nr = A.normf(res)
    relres = nr/nrhs

    if LSS.parallel.MASTER:
        print 'Final  res = {:.3e}\tr: {:.3e}\tlog10(r/r0) = {:.2f}'.format( nr, relres, np.log10(relres) )
        i, r, dt = zip(*call.log)
        t = np.cumsum(dt)
        DT = datetime.timedelta(0, t[-1])
        print 'Total Time (log) =', DT
