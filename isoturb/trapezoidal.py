import os, time
from numpy import *
from scipy.integrate import ode
from scipy import sparse
from scipy.linalg import norm
import scipy.sparse.linalg as splinalg


import isoturb


class IF3DTrap(isoturb.IF3DJacob):
    '''
    IF3DJacob with trapezoidal time integration
    '''
    def __init__(self, n, nu):
        super(isoturb.IF3DJacob, self).__init__(n, nu)

    def integrate(self, u0_h, v0_h, w0_h, dt, debug=False, tol=1e-6):
        uvw0 = self.ravel(u0_h, v0_h, w0_h)
        uvw1 = uvw0.copy()
        ddt0, ddt1 = self.ddt(0, uvw0), self.ddt(0, uvw1)
        res = (uvw1 - uvw0) - dt/2 * (ddt0 + ddt1)

        if debug:
            print '  ires = {:.3e}'.format( norm(res) )

        def matvec(uvwp):
            u1_h, v1_h, w1_h = self.unravel(uvw1)
            up_h, vp_h, wp_h = self.unravel(uvwp)
            dupdt_h, dvpdt_h, dwpdt_h = \
                self.navierTan(u1_h, v1_h, w1_h, up_h, vp_h, wp_h)
            return uvwp - dt/2 * self.ravel(dupdt_h, dvpdt_h, dwpdt_h)

        for iiter in range(12):   # newton ralphson
            shape = (res.size, res.size)
            A = splinalg.LinearOperator(shape, matvec=matvec, dtype=float)
            duvw, info = splinalg.bicgstab(A, res, tol=tol)
            uvw1 -= duvw
            res = (uvw1 - uvw0) - dt/2 * (ddt0 + self.ddt(0, uvw1))

            nr = norm(res)
            #if debug:
            #    print nr
            if nr < tol and iiter > 3:
                if debug:
                    print '{:d}: res = {:.3e}'.format(iiter, nr)
                break


        assert norm(res) < tol
        return self.unravel(uvw1)
           
'''
if __name__ == '__main__':
    from scipy.linalg import norm
    from matplotlib.pyplot import *

    n, nu = 16, 0.001
    ns = IF3DTrap(n, nu)
    dt = pi / 16.

    NT = 2048

    print 'dt =', dt

    if os.path.exists('init.npz'):
        print 'loading'
        npz = load('init.npz')
        u_h, v_h, w_h = npz['u_h'], npz['v_h'], npz['w_h']
        npz.close()
    else:
        u = cos(ns.x) * sin(ns.y)
        v = -sin(ns.x) * cos(ns.y) - cos(ns.y) * sin(ns.z)
        w = sin(ns.y) * cos(ns.z)
        u_h, v_h, w_h = ns.f2c(u), ns.f2c(v), ns.f2c(w)
        ns.pressure(u_h, v_h, w_h)

        for i in range(128):
            u_h, v_h, w_h = super(IF3DTrap,ns).integrate(u_h, v_h, w_h, dt, debug=True)
            E = norm(u_h)**2 + norm(v_h)**2 + norm(w_h)**2 \
              - abs(u_h[0,0,0])**2 - abs(v_h[0,0,0])**2 - abs(w_h[0,0,0])**2
            print (i+1)*dt, E, float(i)/128*100.0 #, E * exp(.5 * pi * nu * i)

        savez('init.npz', u_h=u_h, v_h=v_h, w_h=w_h )

    u0 = ns.ravel(u_h, v_h, w_h)
    m = len(u0)
    U = memmap( DIR+'traj_{}.dat'.format(NT), dtype=float64, mode='w+', shape=(NT+1,m) )
    U[0,:] = u0

    for i in range(NT):
        u_h, v_h, w_h = ns.integrate(u_h, v_h, w_h, dt)
        U[i+1,:] = ns.ravel(u_h, v_h, w_h)
        print 'step: {:d}'.format(i+1)

    savez('restart_out.npz', u_h=u_h, v_h=v_h, w_h=w_h)

   ''' 
