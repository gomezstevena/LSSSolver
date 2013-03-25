from numpy import *
from scipy import linalg
from scipy.sparse import linalg as splinalg
from .. import krylov
from . import lss

class MultigridCycle(object):
    def __init__(self, system):
        self.ns = system

    def smoothing(self, w, rhs, pre=True):
        raise NotImplementedError('Smoothing not defined')

    def runCycle(self, rhs):
        raise NotImplementedError('runCycle not defined')

    @property 
    def dt(self):
        return self.ns.dt

    __call__ = __mul__ = runCycle

class VCycle(MultigridCycle):
    def __init__(self, system, levels, pre_iters=3, post_iters=6, 
                                        relax_range=(1.0, 0.75), depth=0 ):
        super(VCycle, self).__init__(system)
        self.pre_iters = pre_iters
        self.post_iters = post_iters or self.pre_iters # default to same pre/post iterations
        self.depth = depth
        self.relax = tuple(relax_range)

        self.shape = (self.ns.n, self.ns.m)

        #print self.depth, self.pre_iters, self.post_iters

        if levels:
            dlevel = levels.pop(0)
            self.ns.coarsen(dlevel)
            self.coarse_cycle = type(self)( self.ns.ns_coarse, levels = levels, 
                                    pre_iters = self.pre_iters, post_iters = self.post_iters,
                                    relax_range = self.relax, depth=self.depth+1 )
        else:
            self.coarse_cycle = None


    def smoothing(self, w, rhs, pre=True ):
        iters = self.pre_iters if pre else self.post_iters
        relaxes = linspace(self.relax[0], self.relax[1], iters)
        res = rhs.copy() if pre else rhs - self.ns * w
        for omega in relaxes:
            w += omega * self.ns.solveSchurLower(res)
            res = rhs - self.ns * w
            self.ns.iterHook(res, self.depth, pre )
        return res

    def runCycle(self, rhs):

        rhs = rhs.ravel()

        w = zeros_like(rhs)

        res = self.smoothing( w, rhs, True)

        if self.coarse_cycle:
            res_coarse = self.ns.restrict(res)
            dw_coarse = self.coarse_cycle(res_coarse)
            dw = self.ns.interpolate( dw_coarse )
            w += dw

            res = self.smoothing( w, rhs, False)

        return w

    __call__ = __mul__ = runCycle


class VCycleKrylov (VCycle):
    def __init__(self, system, levels, skip=10, tol=1e-5, method=krylov.conjGrad, **kwargs ):
        self.skip = skip
        self.tol = tol
        self.method = method
        super(VCycleKrylov, self).__init__(system, levels, **kwargs)


    def smoothing(self, w, rhs, pre=True):
        iters = self.pre_iters if pre else self.post_iters
        call = lambda x,r: self.ns.iterHook(r, self.depth, pre)

        new_w, err = self.method( self.ns, rhs, x0=w, maxiter=iters, dot=self.ns.dot, 
                                callback=call, skip=self.skip, tol=self.tol )
        
        w[:] = new_w
        res = rhs - self.ns * w
        return res




class MGrid(lss.LSS):

    def __init__(self, ns, dt, traj, shape=None):
        super(MGrid, self).__init__(ns, dt, traj)
        self.ns_coarse = None
        self.coarse_level = None


    def coarsen(self, d_level=True):
        raise NotImplementedError("Must implement coarsening in subclass")

    def restrict(self, w):
        """Restrict values in w to coarser grid"""
        ## Coarsen in Time
        w_tc = w.reshape([-1, self.m])
        if self.coarse_level[0]: #coarsen in time
            w_tc = self.restrictTime(w_tc) 

        ## Coarsen in Space
        w_c = w_tc
        nt = w_tc.shape[0]
        if self.coarse_level[1]:
            w_c = zeros( (nt, self.cshape[1]) )
            for i in xrange( nt ):
                w_c[i] = self.restrictSpace(w_tc[i])

        return w_c.ravel()

    def restrictTime(self, u):
        """Restrict solution from fine to coarse time-level"""
        u = u.reshape( (-1, self.m) )
        if len(u) == self.n + 1:
            return u[::2]
        else:
            #return self.R_mat.dot(u)
            return 0.5*(u[1::2] + u[:-1:2])

    def interpolate(self, wc):
        """Interpolate values from coarser grid to finer grid"""
        wc = wc.reshape( (-1, self.cshape[1]) )
        nc = len(wc)
        ## Refine in space but not time
        if self.coarse_level[1]:
            w_f = zeros( (nc, self.shape[1]) ) 
            for i in xrange( nc ):
                w_f[i] = self.ns_coarse.interpolateSpace( wc[i] )
        else:
            w_f = wc

        #refine in time
        if self.coarse_level[0]: 
            w_f = self.interpolateTime(w_f)

        return w_f.ravel()

    def interpolateTime(self, wc):
        """Interpolate values from coarse to fine time"""
        m = wc.shape[1]
        if len(wc) == self.n//2 + 1:
            res = zeros( (self.n+1, m) )
            res[::2] = wc
            res[1:-1:2] = 0.5*(wc[1:] + wc[:-1])
        else:
            return hstack( (wc, wc) ).reshape((-1,m))

    def iterHook(self, res, lvl, pre=True):
        pass




