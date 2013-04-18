#!/usr/bin/env python

import numpy as np
import LSSSolver as LSS
from LSSSolver import isoturb
import time, itertools
import cPickle as pickle

"""
from matplotlib import pyplot as plt
from matplotlib import animation
"""

class BurgersLSS(LSS.parallel.MGridParallel):
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

    __mul__ = __call__ = schur

    def restrictTime(self, w):
        w = w.reshape((-1,self.m))
        n = len(w)

        if n == self.n+2:
            #residual
            nc = self.n//2
            out = np.zeros( (nc+2, self.m) )


            #print nc, n, w.shape, out.shape
            for i in xrange(nc):
                offs = 0.0
                if i == 0 and self.comm.start_node:
                    offs += 0.5
                if i == nc-1 and self.comm.end_node:
                    offs += 0.5

                im = 2*i
                out[i+1] = (w[im] + 3*w[im+1] + 3*w[im+2] + w[im+3])/(8.0+offs)
            self.ns_coarse.comm.fixOverlap(out)
            return out
        elif n == self.n+1:
            #trajectory
            #return super(IF3DParallel, self).restrictTime(w)
            w = self.comm.pad(w)
            weight = 2.0
            nc = self.n//2
            out = np.zeros( (nc+1, self.m) )
            for i in xrange(nc+1):
                out[i] = (w[2*i] + weight*w[2*i+1] + w[2*i+2])/(weight+2.0)

            if self.comm.start_node:
                out[ 0] = w[ 1]

            if self.comm.end_node:
                out[-1] = w[-2]

            return out

        else:
            raise ValueError('Unexpected Trajectory Length')

    def interpolateTime(self, wc):
        wc = wc.reshape((-1, self.m))
        n = len(wc)

        if n == self.n//2 + 2:
            nc = self.n//2
            out = np.zeros( (self.n+2, self.m) )

            for i in xrange(1, nc+1):
                out[2*i-1] = 0.25 * (3*wc[i] + wc[i-1] )
                out[ 2*i ] = 0.25 * (3*wc[i] + wc[i+1] )

            if self.comm.start_node:
                out[ 1] = 0.5 * wc[ 1]
            if self.comm.end_node:
                out[-2] = 0.5 * wc[-2]

            self.comm.fixOverlap(out)
            return out
        else:
            raise ValueError('Unexpected Trajectory Length')



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

def main(lvls):
    k = 16
    nu = 0.001
    dt = np.pi / 16
    Nt = 2048

    tol = 1e-6

    

    traj = np.load(DIR+'btraj_2048.npy')[:Nt+1]
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
    w, err = LSS.krylov.conjGrad(A, rhs, dot=A.dot, callback=call, M=vcycle, tol=tol, maxiter=15)
    t1 = time.time() - t1


    res = rhs - A*w
    r1 = A.normf(res) / A.normf(rhs)

    i,r,dt = zip(*call.log)
    t = np.cumsum(dt)

    return call.log

    call2 = LSS.callbacks.LogCallback(A, rhs, skip=200 )
    t2 = time.time()
    w2, err = LSS.krylov.minRes(A, rhs, dot=A.dot, callback=call2, tol=tol, maxiter=1000)
    t2 = time.time() - t2


    v = A.BT(w)
    v_full = A.comm.collect( v )
    w_full = A.comm.collect( w )


    if LSS.parallel.MASTER:
        np.savez( 'burgers_sens_{:d}.npz'.format(Nt), v=v_full, w=w_full, dt=dt, nu=nu )

        print 'Total time: {}, res: {}'.format(t1, r1)

        #View 10% perturbation in viscosity (nu)
        #anim = animate_data(v_full*nu + traj, traj)
        #anim.save('Burg_Sens.mp4', fps=48)

        

        i2,r2,dt2 = zip(*call2.log)
        t2 = np.cumsum(dt2)

        plt.figure();
        plt.semilogy(t, r, t2, r2)
        plt.xlim(0, t.max())
        plt.legend(['MGrid', 'MINRES'], loc='best')


        plt.show()

def anagrams(*args):
    data = []
    for i, arg in enumerate(args):
        data += [i]*arg

    return np.unique(itertools.permutations(data))

def gen_paths(x, y):
    #print x, y
    paths = []
    if x > 0:
        sub = gen_paths(x-1, y)
        for s in sub:
            paths.append( [(1,0)] + s )
    if y > 0:
        sub = gen_paths(x, y-1)
        for s in sub:
            paths.append( [(0,1)] + s )

    if x> 0 and y > 0:
        sub = gen_paths(x-1, y-1)
        for s in sub:
            paths.append( [(1,1)] + s )


    if x==0 and y==0:
        paths.append( [] )

    return paths

if __name__ == '__main__':
    DIR = '/master/home/gomezs/lss/'

    """
    results = []
    all_data = gen_paths(5,3)
    all_data.reverse()
    size = len(all_data)
    if LSS.parallel.MASTER:
        print 'Trying {} unique level sets'.format( size )



    for i, dat in enumerate(all_data):
        lvls = dat

        if LSS.parallel.MASTER:
            print '{}/{}: Trying {}'.format(i+1, size, lvls)
        log = main(lvls)
        results.append(  log  )


    if LSS.parallel.MASTER:
        with open(DIR+'logs_all.pkl', 'w') as fid:
            pickle.dump(results, fid)
    """
    lvls = 5*[(1,0)] + 3*[(0,1)]
    main(lvls)