import krylov, ode
from lss import VCycleKrylov

class ShadowODE (object):
    "Friendly wrapper class for computing shadow trajectories for an ODE"
    def __init__(self, traj, outer=krylov.conjGrad, inner = krylov.minRes, 
                                          levels=5, iters=(40,80), tol = 1e-6 ):
        self._traj = traj
        self.outer = outer
        self.inner = inner
        self.shape = (traj.N, traj.dim)
        self.lss_eqns = ode.ODEPar(traj.system, traj.dt, traj.u, 
                                                   shape = self.shape, top=True)
        self.vcycle = VCycleKrylov( self.lss_eqns, levels, 
                                    pre_iters = iters[0], post_iters = iters[1],
                                                        tol=tol, method = inner)
    def solve(self, rhs, maxiter=None, tol=1e-6, callback=None):
        rhs = rhs.ravel()
        ls = self.lss_eqns
        w, err = self.outer( ls, rhs, tol=tol, callback=callback, 
                               maxiter = maxiter, dot = ls.dot, M = self.vcycle)
        
        
        self._last_lagrange_multiplier = w
        self._last_shadow_trajectory = self.lss_eqns.BT(w)
        
        if err[0] != 0:
            raise RuntimeWarning('Solution did not converge within tolerance in {} iterations'.format(err[1]) )
        
        return self._last_shadow_trajectory



    @property 
    def traj(self):
        return self._traj

    @property 
    def ode(self):
        return self.traj.system

    def __getattr__(self, thing): #Ugly hack for now
        return getattr(self.lss_eqns, thing)

    def __mul__(self, w):
        return ( self.lss_eqns*w.ravel() ).reshape( (-1, self.traj.dim) )