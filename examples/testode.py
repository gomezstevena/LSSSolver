import LSSSolver as LSS
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    # Define Lorenz ODE
    lorenz = LSS.ode.LorenzSystem(sigma = 10., rho = 28., beta = 8./3.)

    dt = 0.02   # Time step size
    N = 2**11   # Number of time steps
    tol = 1e-6  # Relative Tolerance for solver


    if LSS.parallel.MASTER: # Only output if Master node (i.e. Node 0)
        print 'N = {}, dt = {}, tol = {:.3e}'.format(N, dt, tol)


    # Create Initial Trajectory or load from file
    T_final = (N+1)*dt
    T_windup = (N+1)*dt/10.0
    traj = LSS.Trajectory( lorenz, time=(T_windup, T_final), dt=dt, filename='lorenz_data')


    # Create Least Squares System and rhs
    # iters are (pre, post) iterations for VCycle
    lsq = LSS.ShadowODE( traj, iters=(30,60), tol=tol )
    rhs = lsq.mapTraj( lorenz.dfdrho )

    # Optional: callback logs iterations and outputs to stdout every skip iterations
    call = LSS.callbacks.LogCallback(lsq, rhs, fname='ode_log.npy', skip=5 ) 

    # Solve System with rhs, uses Conjugate Gradient w/ Multigrid Preconditioner
    w, err = lsq.solve(rhs, callback=call, tol=tol)
    
    # Final residual
    res = rhs - lsq*w
    nr = lsq.normf(res)

    v = lsq.BT(w)

    w_full = lsq.comm.collect(w)
    v_full = lsq.comm.collect(v)

    ##-------------------------------------------------------------##
    ## Post processing done only on master node                    ##
    ##-------------------------------------------------------------##
    if LSS.parallel.MASTER:
        print 'CG Info err:{}, n_iters:{}'.format(*err)
        print 'Final res:', nr, 'Rel Res:', nr/call.nb

        vx,vy,vz = v_full.T
        x,y,z = traj.u.T
        ddrho = np.mean(v_full, axis=0)
        print 'Average Z sensitivity mean =', ddrho[2]

        # Plot Cumulative sensivity
        plt.subplot(1,2,1)

        plt.plot( traj.times, vz)#np.cumsum(vz)/np.r_[1:N+2] )
        plt.xlabel(r'$t$')
        plt.ylabel(r'Instantaneous Mean Z Sensivity')
        plt.plot([0, (N+1)*dt], [ddrho[2], ddrho[2]], '--r' )
        plt.xlim( 0, (N+1)*dt )

        # Plot Residual over time
        plt.subplot(1,2,2)

        i, res, dt = zip(*call.log)
        t = np.cumsum(dt)
        print 'Total Time:', t[-1]
        plt.semilogy( t, res )
        plt.xlabel(r'Wall Time (seconds)')
        plt.ylabel(r'$\|rhs - A w\|_2$')
        plt.grid('on')
        plt.tight_layout()
        plt.show()
