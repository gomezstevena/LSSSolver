from numpy import *
from ..isoturb.trapezoidal import IF3DTrap

class IF3DParallel(MGridParallel, IF3DMGrid):
    def __init__(self, k, nu, dt, traj, shape, top=False):
        ns = IF3DTrap(k, nu)
        MGridParallel.__init__(self, ns, dt, traj, shape=shape, top=top)


    def solveSchurLower(self, res):
        res = res.reshape((-1, self.m) )
        out = zeros_like(res)
        out[1:-1] = IF3DMGrid.solveSchurLower( self, res[1:-1] ).reshape( (-1,self.m) )
        self.comm.fixOverlap(out)
        return out.ravel()

    def iterHook(self, res, lvl, pre=True):
        nr = self.normf(res)
        pre_text = ' pre' if pre else 'post'
        if self.comm.start_node:
            print ' {}: {}\t{:.3e}\t{:d}\t{:d}'.format(pre_text, lvl, nr, self.comm.N, self.m)
