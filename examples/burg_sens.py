#!/usr/bin/env python

import numpy as np
import LSSSolver as LSS
from LSSSolver import isoturb
from scipy.sparse import bsr_matrix
from sys import stdout
import time

from matplotlib import pyplot as plt
from matplotlib import animation


class BurgersLSS(LSS.parallel.MGridParallel):
    def iterHook(self, res, lvl, pre=True):
        return
        nr = self.normf(res)
        if self.comm.start_node:
            prefix = 'pre ' if pre else 'post'
            print '{:s}:\tlvl = {:d}\tres = {:.3e}\t{}'.format(prefix, lvl, nr, self.shape)

    def normf(self, u):
        'Norm normalized by spatial scale'
        return super(BurgersLSS, self).normf(u)*np.sqrt(self.ns.dx)

    def _prepare(self):
        self.t = 0
        LSS.ode._constructMatrices(self)

    def matBTvec(self, w):
        w = w.reshape( (-1, self.m) )
        out = self._BT.dot( w.ravel() )
        self.comm.fixOverlap(out, add=True)
        return out

    def schur(self, w):
        w = w.reshape( (-1, self.m) )
        out = np.zeros_like(w)
        out[1:-1] = np.reshape(self._B.dot( self.matBTvec(w[1:-1]) ) + self.eps * self.matEETvec(w[1:-1]), (-1,self.m) )
        self.comm.fixOverlap(out)
        return out.ravel()

    __mul__ = schur



def animate_data(data, u):
    N, M = data.shape


    x = np.linspace(0, 2*np.pi, M)

    fig = plt.figure()
    h, hu  = plt.plot(x, np.zeros_like(x), '-b', x, np.zeros_like(x), '--g', lw=2)
    plt.xlim(0, 2*np.pi)
    plt.ylim(data.min(), data.max() )

    def anim(i):
        di = data[i,:]
        h.set_ydata(di)
        hu.set_ydata(u[i,:])
        plt.title('T = {:d} / {:d}'.format(i,N) )
        return h,hu

    return animation.FuncAnimation(fig, anim, frames=N, interval=1)

def main():
    k = 16
    nu = 0.001
    dt = np.pi / 16
    Nt = 2048

    tol = 1e-6
    DIR = 'LSSSolver/isoturb/'

    lvls = [ (False,True), (False,True), (False,True), \
             (True,False), (True,False), (True,False), (True,False), (True,False), (True,False) ]

    lvls.reverse()

    traj = np.load('btraj_2048.npy')[:Nt+1]
    shp  = (traj.shape[0]-1, traj.shape[1])

    sys = isoturb.Burgers(3*k, nu)

    A = BurgersLSS(sys, dt, traj, shape=shp, top=True)



    rhs = A.mapTraj( sys.dfdnu ).ravel()

    nr = A.normf(rhs)
    if LSS.parallel.MASTER:
        print 'Nt =', Nt, '\tInitial Res =', nr


    vcycle = LSS.lss.VCycleKrylov(A, lvls, skip=40, pre_iters=40, post_iters=80, tol=tol)
    #from IPython import embed
    #embed()
    #return

    call = LSS.callbacks.LogCallback(A, rhs )

    t1 = time.time()
    w, err = LSS.krylov.conjGrad(A, rhs, dot=A.dot, callback=call, M=vcycle, tol=tol, maxiter=20)
    t1 = time.time() - t1

    res = rhs - A*w
    r1 = A.normf( res ) / A.normf(rhs)


    v = A.BT(w)
    v_full = A.comm.collect( v )
    w_full = A.comm.collect( w )


    if LSS.parallel.MASTER:
        np.savez( 'burgers_sens_{:d}.npz'.format(Nt), v=v_full, w=w_full, dt=dt, nu=nu )

        print 'Total time: {}, res: {}'.format(t1, r1)

        #View 10% perturbation in viscosity (nu)
        anim = animate_data(v_full*(nu/10) + traj, traj)
        plt.show()


if __name__ == '__main__':
    main()