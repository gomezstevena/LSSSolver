#!/usr/bin/env python
from numpy import *
from scipy.linalg import *
from scipy.sparse import linalg as splinalg
from scipy.sparse import bsr_matrix
from ..lss import multigrid
from ..parallel import MGridParallel, MASTER



class Ode (object):
    def __init__(self, dim):
        self.dim=dim

    def f( self, u, t):
        return zeros(u.shape)

    def dfdu( self, u, t):
        return zeros( u.shape + (self.dim,) )
    
    def __call__(self, u, t):
        return self.f(u,t)

    def checkDimensions( self, N=100 ):
        u = squeeze( zeros((N,self.dim)) )
        front = (N,) if N>1 else ()
        dudt = self(u, 0)
        assert dudt.shape == front + (self.dim,)

        J = self.dfdu(u, 0)
        assert J.shape == front + (self.dim, self.dim)


class LorenzSystem (Ode):
    def __init__(self, sigma = 10.0, rho = 28.0, beta = 8./3., z0 = 0):
        self.sigma = sigma
        self.rho = rho
        self.beta = beta
        self.z0 = z0
        self.dim = 3

    def f(self, u, t = 0):
        x,y,z = u.T
        dudt = c_[ self.sigma*(y-x), x*(self.rho+self.z0-z), x*y - self.beta*(z-self.z0) ]
        return  squeeze( dudt )
    def dfdu(self, u, t=0 ):
        x,y,z = u.T
        N = len(u) if u.size!=len(u) else 1
        J = zeros( (N,3,3) )
        J[:,0,0] = -self.sigma
        J[:,0,1] = self.sigma
        J[:,1,0] = (self.rho + self.z0) - z
        J[:,1,2] = -x
        J[:,2,0] = y
        J[:,2,1] = x
        J[:,2,2] = -self.beta

        return squeeze(J)
    def dfdz0(self, u ):
        x,y,z = u
        return r_[ 0, x, self.beta ]
    def dfdrho(self, u ):
        x,y,z = u
        return r_[ 0,  x, 0 ]

    #def perturbation(self, u):
    #    return self.dfdrho(u)


def _constructMatrices(obj):
        u_mid = 0.5*(obj.traj[1:] + obj.traj[:-1])
        obj.Jac = obj.ns.dfdu( u_mid, obj.t )
        A = obj.Jac

        I = eye(obj.m)[newaxis,:,:]
        obj.F = -I/obj.dt - A/2.
        obj.G =  I/obj.dt - A/2.
        N = obj.n
        m = obj.m
        obj._B =  bsr_matrix( (obj.F, r_[:N], r_[:N+1]),
                                    blocksize=(m,m), shape=(N*m, (N+1)*m) ) \
                + bsr_matrix( (obj.G, r_[1:N+1], r_[:N+1]),
                                    blocksize=(m,m), shape=(N*m, (N+1)*m) )
        obj._BT = obj._B.T.tobsr()
        obj._S = obj._B * obj._B.T

class ODELSS (multigrid.MGrid):
    def __init__(self, sys, dt, traj):
        super(OdeLSS, self).__init__( sys, dt, traj, shape=None)
        self.t = r_[:self.n]*self.dt + self.dt/2.0
        self._constructMatrices()

    def coarsen(self, d_level = True):
        print 'called coarsen', d_level
        #self.coarse_level = (d_level, False)
        nt = self.n // 2
        traj_coarse = self.restrict(self.traj).reshape( (nt+1,self.m) )
        self.ns_coarse = type(self)( self.ns, 2*self.dt, traj_coarse, (nt,self.m) )
        return self.ns_coarse

    def _prepare(self):
        self.coarse_level = (True, False)
        _constructMatrices(self)

    def checkB(self, v):

        w = self._B * v.ravel()
        v = v.reshape((-1, self.m))
        dvdt = (v[1:] - v[:-1] )/dt
        Av = zeros( (self.n+1, self.m) )
        for i in xrange(self.n+1):
            Av[i] = dot( self.Jac[i], v[i] )

        Avmid = (Av[1:] + Av[:-1])/2

        bv = dvdt - Avmid

        err = w - bv.ravel()
        print self.normf(err)/self.normf( v.ravel() )

    def matBTvec(self, w):
        return self._BT * w.ravel()

    def schur(self, w):
        return self._B*self.matBTvec(w) + self.eps * self.matEETvec(w)

    def restrict(self, w):
        return self.restrictTime(w).ravel()

    def interpolate(self, wc):
        wc = wc.reshape( (-1,self.m) )
        return self.interpolateTime(wc).ravel()

    __mul__ = schur


class ODEPar (MGridParallel, ODELSS):
    def __init__(self, sys, dt, traj, shape=None, top=False):
        MGridParallel.__init__(self, sys, dt, traj, shape, top)
        self.t = r_[(self.comm.rank-1):(self.comm.rank-1+self.comm.chunk)]*self.dt - self.dt/2
        self._constructMatrices()
        self._base = ODELSS

