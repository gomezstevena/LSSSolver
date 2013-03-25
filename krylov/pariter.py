#!/usr/bin/env python
import numpy, math
from numpy import *
from scipy.sparse import linalg as splinalg


def conjGrad( A, b, x0=None, tol=1e-5, maxiter=None, dot=dot, callback=None, skip=1, M=None ):
    """Implementation of conjugate gradient, mimics api of scipy.sparse.linalg.cg with addition of user supplied inner product.
    --- Also uses Polak-Ribiere formula for beta to improve performance for non fixed preconditioners"""
    n = len(b)

    if M is None:
        M = splinalg.LinearOperator( (n,n), matvec=lambda x: x, dtype=float )


    if maxiter is None:
        maxiter = n*10

    x = zeros_like(b) if x0 is None else x0.copy()
    r = b.copy() if x0 is None else b - A*x
    z = M*r
    p = z.copy()

    rr0 = dot(r, r)


    for k in xrange(maxiter):

        Ap = A*p
        pAp = dot(p, Ap)
        zr = dot(z, r)


        alpha = zr / pAp
        x += alpha*p
        Ap *= -alpha; r += Ap; # r = r - alpha*Ap

        rr = dot(r,r)
        if rr < tol*tol or rr/rr0 < tol*tol:
            err = (0, k+1)
            break

        if callback is not None and (k+1)%skip==0:
            callback(x, r)

        z = M*r
        beta = dot(z, Ap)/zr # at this point Ap = r_{k+1} - r_k

        
        
        p*=beta; p+=z #p = z + beta*p

    else:
        err = (-1, maxiter)

    return x, err

def minRes(  A, b, x0=None, tol=1e-5, maxiter=None, dot=dot, callback=None, skip=1, M=None  ):
    n = len(b)

    sqrt = math.sqrt # local lookup

    if M is None:
        M = splinalg.LinearOperator( (n,n), matvec=lambda x: x, dtype=float )


    if maxiter is None:
        maxiter = n*10

    x = zeros_like(b) if x0 is None else x0.copy()
    vi = b.copy() if x0 is None else b - A*x
    vim = zeros_like(vi)

    betai = sqrt( dot(vi,vi) )

    r0 = betai

    eta = betai
    gammai = gammaim = 1; sigmai = sigmaim = 0;

    wi = zeros_like(vi); wim = zeros_like(vi)

    for k in xrange(maxiter):

        #Lancczos
        vi /= betai;

        Avi = A*vi;
        alphai = dot(vi, Avi)
        vip = Avi -alphai*vi - betai*vim
        betaip = sqrt( dot(vip,vip) )

        # QR Factor
        delta = gammai*alphai - gammaim*sigmai*betai
        p1 = sqrt( delta*delta + betaip*betaip )
        p2 = sigmai*alphai + gammaim*gammai*betai
        p3 = sigmaim*betai
        # Givens Rotation
        gammaip = delta/p1; sigmaip = betaip/p1
        #Update soln
        wip = (vi - p3*wim - p2*wi )/p1
        x += gammaip*eta*wip
        eta *= -sigmaip

        # roll over
        betai = betaip;
        sigmaim = sigmai; sigmai = sigmaip;
        gammaim = gammai; gammai = gammaip;
        wim = wi; wi = wip;
        vim = vi; vi = vip;

        if abs(eta) < tol or abs(eta)/r0 < tol:
            return x, (0, k+1)

        if callback is not None and (k+1)%skip==0:
            callback(x, b-A*x )

    else:
        return x, (1, maxiter)



def _main():
    k, nu = 16, 0.001
    dt = pi / 16.
    M = 4*k*k*(k+1) * 2 * 3
    Nt = int(sys.argv[1]) if len(sys.argv) > 1 else 32

    shp = (Nt, M)
    ftraj = os.path.join( ROOT, 'traj_2048.dat')

    ns = IF3DParallel(k, nu, dt, ftraj, shp, top=True)
    rhs = ns.perturbation.copy()

    tol = 1e-5
    levels = [ (False,True), (False,True), (False,True), (True,False) ]

    vcycle = VCycleKrylov(ns, levels, skip = 20, pre_iters = 20, 
                                      post_iters = 40, tol = tol, method = minRes)

    call = SaveLogCallback( ns, rhs, 
                            fname = os.path.join(ROOT, 'log.npy'),
                            outname = os.path.join(ROOT, 'restart_out.dat') )

    if MASTER:
        print 'init res: {nr:.6e}, rel_res: {rr:.3e}'.format(nr=call.nb, rr=1.0)

    w, err = conjGrad( ns, rhs, dot=ns.dot, callback=call, M=vcycle, tol=tol, maxiter=150)
    
    call(w)

    #w_full = ns.comm.collect(w)
    
    if MASTER:
        print('Got Results')
        print 'CG error =', err

        #save( os.path.join(ROOT, 'MG_results.npy'), w_full)



if __name__ == '__main__':
    from numpy import sqrt
    from scipy.linalg import norm
    from scipy.sparse import linalg as splinalg
    import time, sys, os
    from parallel import IF3DParallel, MASTER
    from parutil import SaveLogCallback

    ROOT = "/home/gomezs/IsoTurbData/"
    #ROOT = "/master/home/gomezs/isoturb/"
    _main()
