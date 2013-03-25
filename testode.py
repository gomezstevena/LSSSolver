import time, os 
import LSSSolver as LSS

from numpy import *
from sys import stdout
from matplotlib.pyplot import *

if __name__ == '__main__':


    sys = LSS.ode.LorenzSystem()

    dt = 0.002
    N = 2**13
    M = 3
    shape = (N, M)
    lvls = 5
    tol = 1e-7

    if LSS.parallel.MASTER:
        print 'N = {}, tol = {:.3e}'.format(N, tol)


    ## Create Initial Trajectory or load from file
    if os.path.isfile('lorenz_traj.npy'):
        u = load('lorenz_traj.npy')[:N+1]
    elif LSS.parallel.MASTER:
        ## Wind-up
        u0 = r_[1., 0., 1.]
        u = LSS.ode.trapIntegrate( sys, u0, dt, N//2 )

        ## Real Trajectory
        u0 = u[-1]
        u = LSS.ode.trapIntegrate( sys, u0, dt, N )
        save('lorenz_traj.npy', u)
    else:
        u = None

    # Create Least Squares System
    ls = LSS.ode.ODEPar(sys, dt, u, shape=shape, top=True )
    rhs = ls.mapTraj( sys.dfdrho ).ravel()

    #Create VCycle and callback function
    vcycle = LSS.lss.VCycleKrylov(ls, lvls, pre_iters=20, post_iters=80, tol=tol, method=LSS.krylov.minRes)
    call = LSS.callbacks.LogCallback(ls, rhs, fname='ode_log.npy', skip=5)

    # Run CG with multigrid preconditioning
    w, err = LSS.krylov.conjGrad(ls, rhs, callback=call, dot=ls.dot, M=vcycle, tol=tol)
    
    # Final residual
    res = rhs - ls*w
    nr = ls.normf(res)

    v = ls.BT(w)

    w_full = ls.comm.collect(w)
    v_full = ls.comm.collect(v)

    if LSS.parallel.MASTER:
        print 'CG Info err:{}, n_iters:{}'.format(*err)
        print 'Final res:', nr, 'Rel Res:', nr/call.nb

        eps = 0.25
        vx,vy,vz = v_full.T
        x,y,z = u.T
        print 'VZ mean =', mean(vz)


        subplot(1,2,1)
        plot(x,z, x + eps*vx, z + eps*vz)

        subplot(1,2,2)

        i, res, dt = zip(*call.log)
        t = cumsum(dt)
        semilogy( t, res )
        xlabel(r'$t$ (seconds)', fontsize=18)
        ylabel(r'$\|\|$res$\|\|_2$', fontsize=18)

        show()
