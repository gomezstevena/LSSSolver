import os, time
from numpy import *
from scipy.linalg import *
from scipy.integrate import ode


"""Set to True to monkey patch default rfftn and irfftn to use FFTW via ANFFT wrapper"""
_USE_ANFFT_ = True
if _USE_ANFFT_:
    try:
        import anfft
        rfftn  = lambda x:  anfft.rfftn(x, measure=True) # measure tries to determine fastest
        irfftn = lambda x: anfft.irfftn(x, measure=True) # method for hardware
    except ImportError:
        print 'ANFFT is not installed, defaulting to numpy fft'
        _USE_ANFFT_ = False


class IsoBox(object):
    '''
    An isotropic box in spectral representation. Maximum wavenumber is n
    The number of Fourier modes is 2n x 2n x 2n

    c2f: transform fourier modes (complex array of 2n x 2n x (n+1))
         to a real grid (real array of 3n x 3n x 3n)
    f2c: adjoint of c2f, also a dealiased psuedo inverse of c2f, i.e.,
         f2c(c2f(c)) == c
         c2f(f2c(f)) == dealiase(f)
    '''
    def __init__(self, n):
        'n is the max wavenumber in all 3 spatial dimensions'
        self.n = n
        self.n32 = int(ceil(n * 1.5))

        n32 = self.n32
        self.x, self.y, self.z = mgrid[:n32*2,:n32*2,:n32*2] * pi / n32

        self.kx = hstack([r_[:n+1], r_[-n+1:0]])[:,newaxis,newaxis]
        self.ky = hstack([r_[:n+1], r_[-n+1:0]])[newaxis,:,newaxis]
        self.kz = r_[:n+1][newaxis,newaxis,:]

        self.derivX = 1j * hstack([r_[:n], 0, r_[-n+1:0]])[:,newaxis,newaxis]
        self.derivY = 1j * hstack([r_[:n], 0, r_[-n+1:0]])[newaxis,:,newaxis]
        self.derivZ = 1j * hstack([r_[:n], 0])[newaxis,newaxis,:]
        self.laplace = - hstack([r_[:n+1], r_[-n+1:0]])[:,newaxis,newaxis]**2 \
                       - hstack([r_[:n+1], r_[-n+1:0]])[newaxis,:,newaxis]**2 \
                       - r_[:n+1][newaxis,newaxis,:]**2

        self.laplace4div = array(self.laplace, float).copy()
        self.laplace4div[0,0,0] = inf

    def extend(self, x_h):
        '3/2 rule, padded with 0s'
        n = self.n
        assert x_h.shape == (n * 2, n * 2, n + 1)
        xe_h = zeros([self.n32 * 2, self.n32 * 2, self.n32 + 1], complex)
        xe_h[:n,:n,:n] = x_h[:n,:n,:n]
        xe_h[-n+1:,:n,:n] = x_h[-n+1:,:n,:n]
        xe_h[:n,-n+1:,:n] = x_h[:n,-n+1:,:n]
        xe_h[-n+1:,-n+1:,:n] = x_h[-n+1:,-n+1:,:n]
        return xe_h

    def truncate(self, xe_h):
        '3/2 rule, truncate higher wavenumbers than n'
        n = self.n
        assert xe_h.shape == (self.n32 * 2, self.n32 * 2, self.n32 + 1)
        x_h = zeros([n * 2, n * 2, n + 1], complex)
        x_h[:n,:n,:n] = xe_h[:n,:n,:n]
        x_h[-n+1:,:n,:n] = xe_h[-n+1:,:n,:n]
        x_h[:n,-n+1:,:n] = xe_h[:n,-n+1:,:n]
        x_h[-n+1:,-n+1:,:n] = xe_h[-n+1:,-n+1:,:n]
        return x_h

    def c2f(self, x_h):
        'see class doc'
        assert x_h.shape == (self.n*2, self.n*2, self.n+1)
        x_h[:,:,[0,-1]] *= sqrt(2)
        x = irfftn( self.extend(x_h))/sqrt(self.n)
        x_h[:,:,[0,-1]] /= sqrt(2)
        return x
    
    def f2c(self, x):
        'see class doc'
        assert x.shape == (self.n32*2, self.n32*2, self.n32*2)
        #x = pyfftw.n_byte_align(x, pyfftw.simd_alignment)
        x_h = self.truncate( rfftn(x)/sqrt(self.n) )
        x_h[:,:,[0,-1]] /= sqrt(2)
        return x_h

class IF3D(IsoBox):
    '''
    Homogeneous incompressible flow in a 3D istrotropic box, forced
    on the 3 lowest wavenumbers (1,0,0), (0,1,0), (0,0,1) with fixed POWER.
    '''
    def __init__(self, n, nu):
        IsoBox.__init__(self, n)
        self.nu = float(nu)

        self.ode = ode(self.ddt)
        self.ode.set_integrator('dopri5', nsteps=10000, rtol=1e-5, atol=1e-9)

    def forcing(self, u_h, v_h, w_h):
        POWER = 20 * self.nu * (self.n*2)**6
        i = [0,0,1], [0,1,0], [1,0,0]
        energy = norm(u_h[i])**2 + norm(v_h[i])**2 + norm(w_h[i])**2

        
        c = POWER / energy

        fx_h, fy_h, fz_h = u_h * 0, v_h * 0, w_h * 0
        fx_h[i] += c * u_h[i]
        fy_h[i] += c * v_h[i]
        fz_h[i] += c * w_h[i]
        return fx_h, fy_h, fz_h

    def navierStokes(self, u_h, v_h, w_h):
        dudt_h, dvdt_h, dwdt_h = self.forcing(u_h, v_h, w_h)
        # viscosity
        dudt_h += self.nu * self.laplace * u_h
        dvdt_h += self.nu * self.laplace * v_h
        dwdt_h += self.nu * self.laplace * w_h
        # advection
        u, v, w = self.c2f(u_h), self.c2f(v_h), self.c2f(w_h)
        uu_h, vv_h, ww_h = self.f2c(u**2), self.f2c(v**2), self.f2c(w**2)
        uv_h, uw_h, vw_h = self.f2c(u*v), self.f2c(u*w), self.f2c(v*w)
        dudt_h -= self.derivX * uu_h + self.derivY * uv_h + self.derivZ * uw_h
        dvdt_h -= self.derivX * uv_h + self.derivY * vv_h + self.derivZ * vw_h
        dwdt_h -= self.derivX * uw_h + self.derivY * vw_h + self.derivZ * ww_h
        # pressure
        self.p = self.pressure(dudt_h, dvdt_h, dwdt_h)
        return dudt_h, dvdt_h, dwdt_h

    def pressure(self, u_h, v_h, w_h):
        div = self.derivX * u_h + self.derivY * v_h + self.derivZ * w_h
        p = div / self.laplace4div
        u_h -= self.derivX * p
        v_h -= self.derivY * p
        w_h -= self.derivZ * p
        return p

    def ravel_old(self, u_h, v_h, w_h):
        'Package 3 complex 3D arrays into a single 1D real array'
        return hstack([ravel(u_h.real), ravel(u_h.imag),
                       ravel(v_h.real), ravel(v_h.imag),
                       ravel(w_h.real), ravel(w_h.imag)])

    def unravel_old(self, uvw1d):
        'Unpackage 3 complex 3D arrays from a single 1D real array'
        ur_h, ui_h, vr_h, vi_h, wr_h, wi_h \
            = uvw1d.reshape([6, self.n*2, self.n*2, self.n+1])
        return ur_h + 1j * ui_h, vr_h + 1j * vi_h, wr_h + 1j * wi_h
    
    def ravel(self, u_h, v_h, w_h):
        'Package 3 complex 3D arrays into a single 1D real array'
        return r_[u_h.ravel(), v_h.ravel(), w_h.ravel()].view(float)

    def unravel(self, uvw1d):
        'Unpackage 3 complex 3D arrays from a single 1D real array'
        uvw = uvw1d.view(dtype=complex)
        uvw = uvw.reshape( (3, self.n*2, self.n*2, self.n+1) )
        return uvw.copy()

    def perturbation(self, uvw1d):
        uvw = self.unravel(uvw1d)
        fuvw = self.forcing(*uvw)
        return self.ravel(*fuvw)

    def ddt(self, t, uvw1d):
        'Wrapper of navierStokes'
        u_h, v_h, w_h = self.unravel(uvw1d)
        dudt_h, dvdt_h, dwdt_h = self.navierStokes(u_h, v_h, w_h)
        return self.ravel(dudt_h, dvdt_h, dwdt_h)

    def integrate(self, u0_h, v0_h, w0_h, t):
        'self.ode was set up in __init__, uses dopri scheme (matlab ode45)'
        self.ode.set_initial_value(self.ravel(u0_h, v0_h, w0_h), 0)
        self.ode.integrate(t)
        return self.unravel(self.ode.y)

    def energySpectrum(self, u_h, v_h, w_h):
        'return values (k, e) to be plotted as loglog(k, e)'
        e_h = ravel(abs(u_h)**2 + abs(v_h)**2 + abs(w_h)**2) / (2*self.n)**6
        k = ravel(sqrt(self.kx**2 + self.ky**2 + self.kz**2))
        ik = k.argsort()[::-1]
        e_h, k = e_h[ik], k[ik]
        spectrum = cumsum(e_h)
        return k[::-1], spectrum[::-1]

    def Qcriterion(self, u_h, v_h, w_h):
        'Computes the Q criterion, sqr rotation rate minus sqr shear rate'
        dX, dY, dZ = self.derivX, self.derivY, self.derivZ
        ux_h, uy_h, uz_h = dX * u_h, dY * u_h, dZ * u_h
        vx_h, vy_h, vz_h = dX * v_h, dY * v_h, dZ * v_h
        wx_h, wy_h, wz_h = dX * w_h, dY * w_h, dZ * w_h

        S1, S2, S3 = ux_h, vy_h, wz_h
        S4, S5, S6 = (uy_h + vx_h) / 2, (uz_h + wx_h) / 2, (vz_h + wy_h) / 2
        R1, R2, R3 = (uy_h - vx_h) / 2, (uz_h - wx_h) / 2, (vz_h - wy_h) / 2

        S = [self.c2f(S1), self.c2f(S2), self.c2f(S3),
             self.c2f(S4), self.c2f(S5), self.c2f(S6)]
        R = [self.c2f(R1), self.c2f(R2), self.c2f(R3)]

        return (sum(r**2 for r in R) - sum(s**2 for s in S)) / 2.
        
    def tecplot(self, fname, u_h, v_h, w_h):
        u, v, w = self.c2f(u_h), self.c2f(v_h), self.c2f(w_h)
        Q = self.Qcriterion(u_h, v_h, w_h)

        varStr = '"X", "Y", "Z", "U-X", "U-Y", "U-Z", "Q"'
        fmtStr = ', '.join(['%f'] * 7) + '\n'
        zoneStr = 'ZONE T="ISO", I={0}, J={0}, K={0}, F=POINT\n'

        f = open(fname, 'wt')
        f.write('TITLE = "DataLine"\n')
        f.write('VARIABLES = {0}\n'.format(varStr))
        f.write(zoneStr.format(self.n32 * 2))

        variables = (self.x, self.y, self.z, u, v, w, Q)
        for k in range(self.n32 * 2):
            for j in range(self.n32 * 2):
                for i in range(self.n32 * 2):
                    f.write(fmtStr % tuple(var[i,j,k] for var in variables))
        f.close()


class IF3DJacob(IF3D):
    '''
    IF3D class with Jacobians (Tangent and Adjoint) computed and verified.
    '''
    def __init__(self, n, nu):
        IF3D.__init__(self, n, nu)

    def forcingTan(self, u_h, v_h, w_h, up_h, vp_h, wp_h):
        i = [0,0,1], [0,1,0], [1,0,0]
        POWER = 20 * self.nu * (self.n*2)**6
        energy = norm(u_h[i])**2 + norm(v_h[i])**2 + norm(w_h[i])**2
        energyP = 2 * real(u_h[i] * conj(up_h[i]) + \
                           v_h[i] * conj(vp_h[i]) + \
                           w_h[i] * conj(wp_h[i])).sum()
        
        c = POWER / energy
        cp = POWER / energy**2 * -energyP

        fpx_h, fpy_h, fpz_h = up_h * 0, vp_h * 0, wp_h * 0
        fpx_h[i] += c * up_h[i] + cp * u_h[i]
        fpy_h[i] += c * vp_h[i] + cp * v_h[i]
        fpz_h[i] += c * wp_h[i] + cp * w_h[i]
        return fpx_h, fpy_h, fpz_h

    def navierTan(self, u_h, v_h, w_h, up_h, vp_h, wp_h):
        dupdt_h, dvpdt_h, dwpdt_h \
            = self.forcingTan(u_h, v_h, w_h, up_h, vp_h, wp_h)

        # viscosity
        dupdt_h += self.nu * self.laplace * up_h
        dvpdt_h += self.nu * self.laplace * vp_h
        dwpdt_h += self.nu * self.laplace * wp_h

        # advection
        u, v, w = self.c2f(u_h), self.c2f(v_h), self.c2f(w_h)
        up, vp, wp = self.c2f(up_h), self.c2f(vp_h), self.c2f(wp_h)

        uup, vvp, wwp = 2 * u * up, 2 * v * vp, 2 * w * wp
        uvp, uwp, vwp = u * vp + v * up, u * wp + w * up, v * wp + w * vp
        uup_h, vvp_h, wwp_h = self.f2c(uup), self.f2c(vvp), self.f2c(wwp)
        uvp_h, uwp_h, vwp_h = self.f2c(uvp), self.f2c(uwp), self.f2c(vwp)

        dX, dY, dZ = self.derivX, self.derivY, self.derivZ
        dupdt_h -= dX * uup_h + dY * uvp_h + dZ * uwp_h
        dvpdt_h -= dX * uvp_h + dY * vvp_h + dZ * vwp_h
        dwpdt_h -= dX * uwp_h + dY * vwp_h + dZ * wwp_h

        # pressure projection
        self.pp = self.pressure(dupdt_h, dvpdt_h, dwpdt_h)
        return dupdt_h, dvpdt_h, dwpdt_h

    def forcingAdj(self, u_h, v_h, w_h, fax_h, fay_h, faz_h):
        i = [0,0,1], [0,1,0], [1,0,0]
        POWER = 20 * self.nu * (self.n*2)**6
        energy = norm(u_h[i])**2 + norm(v_h[i])**2 + norm(w_h[i])**2

        c = POWER / energy

        # fax_h, fay_h, faz_h are NOT complex derivative,
        # but derivatives wrt real and imaginary parts, i.e.,
        # objp = sum(fax_h.real * fpx_h.real) + sum(fax_h.imag * fpx_h.imag)
        #      + ...
        ca = sum(u_h[i] * conj(fax_h[i]) \
               + v_h[i] * conj(fay_h[i]) \
               + w_h[i] * conj(faz_h[i])).real
        energyA = POWER / energy**2 * -ca
        

        ua_h, va_h, wa_h = u_h * 0, v_h * 0, w_h * 0
        ua_h[i] += c * fax_h[i]
        va_h[i] += c * fay_h[i]
        wa_h[i] += c * faz_h[i]

        ua_h[i] += 2 * energyA * u_h[i]
        va_h[i] += 2 * energyA * v_h[i]
        wa_h[i] += 2 * energyA * w_h[i]
        return ua_h, va_h, wa_h

    def Tan(self, uvw1d, duvw1d):
        uvw = self.unravel(uvw1d)
        duvw = self.unravel(duvw1d)
        Adu = self.navierTan(uvw[0], uvw[1], uvw[2], duvw[0], duvw[1], duvw[2] )
        return self.ravel(*Adu)

    def Adj(self, uvw1d, duvw1d):
        uvw = self.unravel(uvw1d)
        duvw = self.unravel(duvw1d)
        ATdu = self.navierAdj(uvw[0], uvw[1], uvw[2], duvw[0], duvw[1], duvw[2] )
        return self.ravel(*ATdu)


    def navierAdj(self, u_h, v_h, w_h, dudta_h, dvdta_h, dwdta_h):
        # assumes both (up_h, vp_h, wp_h) and (dudta_h, dvdta_h, dwdta_h)
        # are div-free, so pressure projection can be performed at the end
        ua_h, va_h, wa_h = \
            self.forcingAdj(u_h, v_h, w_h, dudta_h, dvdta_h, dwdta_h)

        # viscosity
        ua_h += self.nu * self.laplace * dudta_h
        va_h += self.nu * self.laplace * dvdta_h
        wa_h += self.nu * self.laplace * dwdta_h

        # advection
        u, v, w = self.c2f(u_h), self.c2f(v_h), self.c2f(w_h)

        dX, dY, dZ = self.derivX, self.derivY, self.derivZ
        # Because dX is pure imaginary, dX is the adjoint of -dX
        uua = self.c2f(dX * dudta_h)
        vva = self.c2f(dY * dvdta_h)
        wwa = self.c2f(dZ * dwdta_h)
        uva = self.c2f(dX * dvdta_h + dY * dudta_h)
        uwa = self.c2f(dX * dwdta_h + dZ * dudta_h)
        vwa = self.c2f(dY * dwdta_h + dZ * dvdta_h)

        ua = 2 * u * uua + v * uva + w * uwa
        va = 2 * v * vva + u * uva + w * vwa
        wa = 2 * w * wwa + u * uwa + v * vwa

        ua_h += self.f2c(ua)
        va_h += self.f2c(va)
        wa_h += self.f2c(wa)

        # pressure
        self.pa = self.pressure(ua_h, va_h, wa_h)
        return ua_h, va_h, wa_h

    # -------------- audit subroutines --------------- #
    def randomField(self):
        n32 = self.n32
        u = randn(n32*2, n32*2, n32*2)
        v = randn(n32*2, n32*2, n32*2)
        w = randn(n32*2, n32*2, n32*2)
        u_h, v_h, w_h = self.f2c(u), self.f2c(v), self.f2c(w)
        self.pressure(u_h, v_h, w_h)
        return u_h, v_h, w_h

    def testTan(self):
        EP = 1E-7
        up_h, vp_h, wp_h = self.randomField()
        u0_h, v0_h, w0_h = self.randomField()
        u1_h, v1_h, w1_h = u0_h + EP * up_h, v0_h + EP * vp_h, w0_h + EP * wp_h

        du0dt_h, dv0dt_h, dw0dt_h = self.navierStokes(u0_h, v0_h, w0_h)
        du1dt_h, dv1dt_h, dw1dt_h = self.navierStokes(u1_h, v1_h, w1_h)

        dupdt_h, dvpdt_h, dwpdt_h = \
            self.navierTan(u0_h, v0_h, w0_h, up_h, vp_h, wp_h)

        Dupdt_h = (du1dt_h - du0dt_h) / EP
        Dvpdt_h = (dv1dt_h - dv0dt_h) / EP
        Dwpdt_h = (dw1dt_h - dw0dt_h) / EP

        print norm([dupdt_h, dvpdt_h, dwpdt_h]), \
              norm([dupdt_h - Dupdt_h, dvpdt_h - Dvpdt_h, dwpdt_h - Dwpdt_h])

    def testAdj(self):
        u_h, v_h, w_h = self.randomField()

        up_h, vp_h, wp_h = self.randomField()
        fpx_h, fpy_h, fpz_h = self.navierTan(u_h, v_h, w_h, up_h, vp_h, wp_h)

        fax_h, fay_h, faz_h = self.randomField()
        ua_h, va_h, wa_h = self.navierAdj(u_h, v_h, w_h, fax_h, fay_h, faz_h)

        objp0 = sum(up_h * conj(ua_h) \
                  + vp_h * conj(va_h) \
                  + wp_h * conj(wa_h)).real
        objp1 = sum(fpx_h * conj(fax_h) \
                  + fpy_h * conj(fay_h) \
                  + fpz_h * conj(faz_h)).real
        print objp0, objp1 - objp0


def energySpectrumTan(ns, uvw_h, uvwp_h):
    'tangent of ns.energySpectrum'
    u_h, v_h, w_h = ns.unravel(uvw_h)
    up_h, vp_h, wp_h = ns.unravel(uvwp_h)
    ep_h = 2 * ravel(u_h * conj(up_h) + \
                     v_h * conj(vp_h) + \
                     w_h * conj(wp_h)).real / (2*ns.n)**6
    k = ravel(sqrt(ns.kx**2 + ns.ky**2 + ns.kz**2))
    ik = k.argsort()[::-1]
    ep_h, k = ep_h[ik], k[ik]
    spectrumP = cumsum(ep_h)
    return k[::-1], spectrumP[::-1]


if __name__ == '__main__':
    n, nu = 16, 0.001
    ns = IF3DJacob(n, nu)
    dt = pi / 8.

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
            u_h, v_h, w_h = ns.integrate(u_h, v_h, w_h, dt)
            E = norm(u_h)**2 + norm(v_h)**2 + norm(w_h)**2 \
              - abs(u_h[0,0,0])**2 - abs(v_h[0,0,0])**2 - abs(w_h[0,0,0])**2
            print (i+1)*dt, E #, E * exp(.5 * pi * nu * i)

    for i in range(100):
        # u0_h, v0_h, w0_h = u_h, v_h, w_h
        # t0 = time.time()
        u_h, v_h, w_h = ns.integrate(u_h, v_h, w_h, dt)
        # print time.time() - t0
        # stop

        # dudt_h, dvdt_h, dwdt_h = ns.navierStokes(u_h, v_h, w_h)
        # dudt0_h, dvdt0_h, dwdt0_h = ns.navierStokes(u0_h, v0_h, w0_h)
        # print norm((u_h - u0_h) / dt), \
        #       norm((u_h - u0_h) / dt - (dudt_h + dudt0_h) / 2)

        k, s = ns.energySpectrum(u_h, v_h, w_h)
        clf(); loglog(k, s, 'ok'); axis([1, 30, 1E-6, 1])
        loglog([1, 30], power([1, 30], -5./3), '--'); grid()
        print i+1
        savefig('fig/iso%06d.png' % (i+1))
        if i % 10 == 0:
            ns.tecplot('tec/iso%06d.dat' % (i+1), u_h, v_h, w_h)

    savez('soln.npz', u_h=u_h, v_h=v_h, w_h=w_h)
