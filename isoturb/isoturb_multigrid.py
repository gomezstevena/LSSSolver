from ..lss.multigrid import MGrid
from .trapezoidal import IF3DTrap
import functools
from numpy import zeros

def unraved(func):
    '''Decorator to unravel arguments and apply funciton to each component'''
    @functools.wraps(func)
    def wrapped( self, x ):
        uvw = self.ns.unravel(x)
        uvw = ( func(self,i) for i in uvw )
        return self.ns.ravel( *uvw )

    return wrapped

class IF3DMGrid(MGrid):

    def __init__(self, k, nu, dt, traj, shape=None):
        ns = IF3DTrap(k, nu)
        super(IF3DMGrid, self).__init__(ns, dt, traj)

    def coarsen(self, d_level=(False, True) ):
        """Returns new ns object with half the resolution in space and/or time"""
        self.coarse_level = d_level
        nt, nk = self.n, self.ns.n
        ndt = self.dt
        nu = self.ns.nu

        if d_level[0]: # coarsen in time
            nt //= 2
            ndt *= 2

        if d_level[1]:
            nk //= 2
            nu  *= 1

        nm = 6*4*nk*nk*(nk+1)
        self.cshape = ( nt, nm  )

        traj_coarse = self.restrict(self.traj).reshape([nt+1, nm])
        self.ns_coarse = type(self)(nk, nu, ndt, traj_coarse, shape=self.cshape)
        return self.ns_coarse

    @unraved
    def restrictSpec(self, u):
        k = self.ns.n # wavenumber
        kn = k//2
        assert u.shape == (2*k, 2*k, k+1)

        uc = zeros( (k, k, kn + 1), dtype=complex )
        
        uc[:kn,:kn,:kn] = u[:kn,:kn,:kn]
        uc[-kn+1:, :kn, :kn] = u[-kn+1:, :kn, :kn]
        uc[:kn, -kn+1:, :kn] = u[:kn, -kn+1:, :kn]
        uc[-kn+1:,-kn+1:,:kn] = u[-kn+1,-kn+1:,:kn]
        return uc

    @unraved
    def interpolateSpec(self, uc):
        k = self.ns.n
        assert uc.shape == (2*k, 2*k, k + 1)
        uf = zeros( (4*k, 4*k, 2*k+1), dtype=complex )
        uf[:k,:k,:k] = uc[:k,:k,:k]
        uf[-k+1:,:k,:k] = uc[-k+1:,:k,:k]
        uf[:k,-k+1:,:k] = uc[:k,-k+1:,:k]
        uf[-k+1:,-k+1:,:k] = uc[-k+1:,-k+1:,:k]
        return uf

    restrictSpace = restrictSpec
    interpolateSpace = interpolateSpec

    def iterHook(self, res, lvl, pre=True):
        nr = self.normf(res)
        pre_text = ' pre' if pre else 'post'
        print ' {pt}: {lvl}\t{r:.3e}\t{nt:d}\t{m:d}'.format(pt=pre_text,lvl=lvl,r=nr,nt=self.n, m=self.m)




##----------------------------------------------------------------------------##
##-----------------------------Multigrid Testing------------------------------##
##----------------------------------------------------------------------------##
def _main():
    from os import path

    n, nu = 16, 0.001
    dt = pi / 16.
    ns = IF3DMGrid(n, nu, dt, load('traj.npy')[:17] )

    levels = [ (False, True)]*3

    vcycle = VCycle(ns, levels=levels)

    rhs = ns.perturbation.copy()

    if False:#path.isfile('MG_restart_in.npy'):
        w = load('MG_restart_in.npy').ravel()
        assert w.shape == rhs.shape

        if path.isfile('MG_rec.npy'):
            rec = load('MG_rec.npy').tolist()
        else:
            rec = []
    else:
        w = zeros_like(rhs)
        rec = []


    res = rhs - ns*w
    n_res = ns.normf(res)
    n_rhs = ns.normf(rhs)
    print 'init res: {nr:.6g}, rel_res: {rr:.3e}'.format(nr=n_res, rr=n_res/n_rhs)

    rec.append( (0, n_res, 0.0) )

    for i in xrange(1):

        Tstart = time.time()

        w += vcycle(res)

        res = rhs - ns*w

        n_res = ns.normf(res)

        rel_res = n_res/n_rhs

        print 'i:{i:d}, res: {nr:.6g}, rel_res: {rel_res:.3e}'.format(i=i, nr=n_res, rel_res=rel_res)

        rec.append( (rec[-1][0]+1, n_res, time.time()-Tstart ) )

        print '- REC:', rec[-1]


    v = ns.matBTvec(w).reshape( (ns.n+1, ns.m) )
    eta = ns.matETvec(w)

    savez('lss_MG_run.npz', w=w, v=v, eta=eta, rec=rec)


if __name__ == '__main__':
    _main()