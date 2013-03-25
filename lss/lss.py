import os, sys, time
from numpy import *
from scipy.integrate import ode
from scipy import sparse
import scipy.sparse.linalg as splinalg


class LSS(object):
    def __init__(self, ns, dt, traj):
        self.ns = ns
        self.dt = dt
        self.traj = traj

        self.SchurOperator = \
            splinalg.LinearOperator( (self.size, self.size),
                                    matvec = self.schur, dtype=float)

        self.SchurLowerOperator = splinalg.LinearOperator( (self.size,self.size), 
                                            matvec=self.matSchurLowervec, dtype=float)

        # perturbation
        if hasattr(self.ns, 'perturbation'):
            f = array([ self.ns.perturbation(t) for t in traj ])
            self.perturbation = ravel( 0.5 * (f[1:] + f[:-1]) )

        self.ddt_traj = (self.traj[1:] - self.traj[:-1]) / self.dt

        self.eps = 1E-5

    @property
    def n(self):
        'number of time steps (length of trajectory - 1)'
        return self.traj.shape[0] - 1

    @property
    def m(self):
        '#floats per time step'
        return self.traj.shape[1]

    @property 
    def k(self):
        return self.ns.n

    @property 
    def shape(self):
        return (self.n, self.m)

    @property
    def bshape(self):
        return (self.m, self.m)

    def normf( self, w ):
        return sqrt( self.dot(w,w) )

    def dot(self, a, b):
        return dot(a, b)

    def normt( self, w):
        w = w.reshape( (self.n, self.m) )
        d = zeros(self.n)
        for i in xrange(self.n):
            d[i] = linalg.norm( w[i] )
        return d

    def tan(self, istep, vi):
        'Wrapper of ns.navierTan'
        return self.ns.Tan( self.traj[istep], vi )

    def adj(self, istep, wi):
        'Wrapper of ns.navierAdj'
        return self.ns.Adj( self.traj[istep], wi)

    # -------------- matrices in operator form ------- #
    def matBvec(self, v):
        'Matrix B, encodes the tangent system in trapezoidal rule'
        v = v.reshape([self.n + 1, self.m])
        w = zeros([self.n, self.m])
        w[0] = -v[0] / self.dt + 0.5 * self.tan(0, v[0])
        for i in range(1, self.n):
            Bv = 0.5 * self.tan(i, v[i])
            w[i-1] += v[i] / self.dt + Bv
            w[i] = -v[i] / self.dt + Bv
        w[-1] += v[-1] / self.dt + 0.5 * self.tan(-1, v[-1])
        return ravel(w)

    def matBTvec(self, w):
        'Matrix B.T, encodes the adjoint system in (adj-)trapezoidal rule'
        w = w.reshape([self.n, self.m])
        v = zeros([self.n + 1, self.m])
        v[0] = -w[0] / self.dt + .5 * self.adj(0, w[0])
        for i in range(1, self.n):
            v[i] = (w[i-1] - w[i]) / self.dt \
                 + .5 * (self.adj(i, w[i]) + self.adj(i, w[i-1]))
        v[-1] = w[-1] / self.dt + .5 * self.adj(-1, w[-1])
        return ravel(v)

    def matEvec(self, eta):
        'Matrix E, encodes the time dilation eta, a proportional change in dt'
        eta = eta.reshape([self.n, 1])
        dw = self.ddt_traj * eta
        return ravel(dw)

    def matETvec(self, w):
        'Matrix E.T, encodes the adjoint of time dilation'
        w = w.reshape([self.n, self.m])
        eta = einsum('nm,nm->n', self.ddt_traj, w)
        return eta

    def matEETvec(self, w):
        return einsum('ni,nj,nj->ni', self.ddt_traj, self.ddt_traj, w.reshape([self.n, self.m]) ).ravel()

    def schur(self, w):
        'Schur complement B * B.T + E * E.T'
        return self.matBvec(self.matBTvec(w)) + self.eps * self.matEvec(self.matETvec(w))

    __mul__ = schur

    def matBBTvecDiag(self, i, wi):
        '''Multiply by the i'th diagonal block of B*B^T'''
        ATw = self.adj(i, wi)

        out = ( 2./(self.dt*self.dt) )* wi
        out +=  -(self.tan(i, wi) + ATw)/(2*self.dt) + self.tan(i, ATw )/4.

        ATpw = self.adj(i+1, wi)
        out +=  (self.tan(i+1, wi) + ATpw)/(2*self.dt) + self.tan(i+1, ATpw)/4.

        return out

    def matEETvecDiag(self, i, wi):
        '''Multiplies by the i'th diagonal block of E*E^T'''
        dudt = self.ddt_traj[i]
        return dot(dudt, wi) * dudt

    def schurDiag(self, i, wi):
        '''Multiplies by the i'th diagonal block of the Schur matrix B*B^T + r * E*E^T'''
        return self.matBBTvecDiag(i, wi) + self.eps * self.matEETvecDiag(i, wi)

    def solveSchurDiag(self, res ):
        '''Solves diagonal of Schur matrix
        Useful for Block Jacobi'''
        res = res.reshape([self.n, self.m])
        out = zeros_like(res)
        for i in xrange(self.n):
            diag_op = splinalg.LinearOperator( (self.m, self.m) ,matvec = lambda x: self.schurDiag(i,x), dtype = float )

            x0 = out[i-1] if i>0 else None

            ans, err = splinalg.minres(diag_op, res[i], x0=x0)
            out[i] = ans

        return out.ravel()

    def matSchurLowervec(self, w):
        w = w.reshape(self.shape)
        out = zeros_like(w)

        for i in xrange(self.n):
            out[i] = self.schurDiag(i, w[i] )
            
            if i > 0:
                AT = self.adj( i, w[i-1] )
                out[i] -= w[i-1]/(self.dt*self.dt)
                out[i] += (self.tan(i, w[i-1] ) + AT)/(2.*self.dt) + self.tan(i, AT)/4.

        return out.ravel()

    def diagBlockOperator(self, i):
        matvec = lambda x: self.schurDiag(i, x)
        A = splinalg.LinearOperator( self.bshape, matvec=matvec, dtype=float )
        return A

    def solveDiagBlock(self, i, rhs, tol=1e-4, x0=None):
        Mi = self.diagBlockOperator(i)
        ans, err = splinalg.minres(Mi, rhs, tol=tol, x0=x0)
        return ans

    def solveSchurLowerBlocks(self, res):
        '''Solves lower triangular portion of the Schur matrix
        Useful for Block Gauss-Siedel
        '''
        res = res.reshape(self.shape)
        out = zeros_like(res)

        out[0] = self.solveDiagBlock(0, res[0])

        for i in xrange(1, self.n):

            AT = self.adj(i, out[i-1])
            L = -out[i-1]/(self.dt*self.dt)
            L += (self.tan(i, out[i-1] ) + AT)/(2.*self.dt) + self.tan(i, AT)/4.

            out[i] = self.solveDiagBlock(i, res[i]-L, x0=out[i-1])

        return out.ravel()

    solveSchurLower = solveSchurLowerBlocks

    # -------------- audit subroutines --------------- #
    def randomVec(self, n):
        vec = [self.ns.randomField() for i in range(n)]
        vec = [self.ns.ravel(u_h, v_h, w_h) for u_h, v_h, w_h in vec]
        return ravel(vec)

    def testBvec(self):
        'test consistency between matBvec and matBTvec'
        v = self.randomVec(self.n + 1)
        w = self.randomVec(self.n)
        obj0 = dot(w, self.matBvec(v))
        obj1 = dot(v, self.matBTvec(w))
        print obj0, obj1 - obj0

    def testEvec(self):
        'test consistency between matEvec and matETvec'
        eta = randn(self.n)
        w = self.randomVec(self.n)
        obj0 = dot(w, self.matEvec(eta))
        obj1 = dot(eta, self.matETvec(w))
        print obj0, obj1 - obj0
           
    @property 
    def size(self):
        return self.n * self.m



if __name__ == '__main__':
    n, nu = 16, 0.001
    dt = pi / 16.
    ns = IF3DLSS(n, nu, dt, load('traj.npy')[:33] )

    class Callback:
        def __init__(self, A, b):
            self.n = 0
            self.A = A
            self.b = b
            self.recorder = []
    
        def __call__(self, x):
            self.n += 1

            if self.n % 5 == 0:
                nr = self.A.normf( self.A * x - self.b )

                print self.n, nr
                self.recorder.append( (self.n, nr, time.time()) )
                sys.stdout.flush()

    callback = Callback(ns, ns.perturbation)
    #callback()

    embed()

    print 'Starting minres ...'
    w, info = splinalg.minres(ns.SchurOperator, ns.perturbation,
                              maxiter=100, tol=1E-6,  callback=callback)
    print 'done.'
    
    v = ns.matBTvec(w).reshape([ns.n + 1, ns.m])
    eta = ns.matETvec(w)
    save('lss.npy', array(v))

    rec = array(callback.recorder)

    savez('lss_minres_run.npz', w=w, v=v, eta=eta, rec=rec)
    
    # ------------- spectrum analysis ------------- #
    spec = []
    for i in range(ns.n + 1):
        k, spectrumP = energySpectrumTan(ns.ns, ns.traj[i], v[i])
        spec.append(spectrumP)

        clf()
        loglog(k, -spectrumP, 'ok'); axis([1, 30, 1E-10, 1])
        loglog([1, 30], power([1, 30], -5./3), '--'); grid()
        savefig('fig/lss%06d.png' % (i+1))
    

