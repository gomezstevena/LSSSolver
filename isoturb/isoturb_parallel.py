from numpy import *
from . import IF3DTrap
from .isoturb_multigrid import IF3DTrap, IF3DMGrid
from ..parallel import MGridParallel

class IF3DParallel(MGridParallel, IF3DMGrid):
    def __init__(self, k, nu, dt, traj, shape, top=False):
        ns = IF3DTrap(k, nu)
        MGridParallel.__init__(self, ns, dt, traj, shape=shape, top=top)
        self._base = IF3DMGrid


    def solveSchurLower(self, res):
        res = res.reshape((-1, self.m) )
        out = zeros_like(res)
        out[1:-1] = IF3DMGrid.solveSchurLower( self, res[1:-1] ).reshape( (-1,self.m) )
        self.comm.fixOverlap(out)
        return out.ravel()

    def iterHook(self, res, lvl, pre=True):
        #print 'calling iterhook'
        pre_text = ' pre' if pre else 'post'
        if self.comm.start_node:
            print ' {}: {}\t{:.3e}\t{:d}\t{:d}'.format(pre_text, lvl, res, self.comm.N, self.m)

    """
    def restrictTime(self, w):
        w = w.reshape((-1,self.m))
        n = len(w)

        if n == self.n+2:
            #residual
            nc = self.n//2
            out = zeros( (nc+2, self.m) )


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
            out = zeros( (nc+1, self.m) )
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
            out = zeros( (self.n+2, self.m) )

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
    """
