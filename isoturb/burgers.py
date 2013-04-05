import numpy as np

from scipy import sparse


def _make_Dx(N, dx):
    o = np.ones(N)
    A = sparse.spdiags( [o, -o], [-1, 1], N, N, format='dok' )
    A[0,-1] =  1.0
    A[-1,0] = -1.0
    A = A.tocsr()
    A.data /= 2.0*dx
    return A

def _make_D2x(N, dx):
    o = np.ones(N)
    A = sparse.spdiags([o, -2*o, o], [-1, 0, 1], N, N, format='dok')
    A[0,-1] = 1
    A[-1,0] = 1
    A = A.tocsr()
    A.data /= dx*dx
    return A

class Burgers (object):
    """
    Represents 1D viscous Burgers Equations, du/dt + 1/2 d/dx(u^2) = nu d^2/dx^2(u)
    domain goes from [0, xlim), in Nx steps, dx = xlim/Nx, BCs are periodic
    """
    def __init__(self, Nx, nu, xlim = 2*np.pi):
        self.Nx = Nx
        self.xlim = float(xlim)
        self.nu = nu
        self.dx = self.xlim / self.Nx

        self._Dx  =  _make_Dx(self.Nx, self.dx).todense()
        self._visc = self.nu * _make_D2x(self.Nx, self.dx).todense()


    def _ddx(self, u):
        dudx = np.roll(u, 1, -1) - np.roll(u, -1, -1)
        dudx /= 2*self.dx
        return dudx

    def _d2dx2(self, u):
        d2udx2 = np.roll(u,1,-1) + np.roll(u,-1,-1)
        d2udx2 -= 2*u
        d2udx2 /= self.dx*self.dx
        return d2udx2

    def dudt(self, u, t=0):
        return -self._ddx( 0.5*u*u ) + self.nu * self._d2dx2(u)

    __call__ = dudt

    def dfdu(self, u, t=0):
        'Jacobian of system at one point in time'
        if u.size == self.Nx:
            U = np.diag(u)
            return - np.dot(self._Dx, U) + self._visc
        elif u.shape[1] == self.Nx:
            N = len(u)
            A = np.empty( (N, self.Nx, self.Nx) )
            for i in xrange(N):
                A[i] = self.dfdu( u[i] )
            return A

    def Tan(self, u, du):
        return -self._ddx(u*du) + self.nu * self._d2dx2(du)

    def Adj(self, u, du):
        return  u*self._ddx(du) + self.nu * self._d2dx2(du)

    def dfdnu(self, u, t=0):
        return self._d2dx2(u)

    @property 
    def fixed_params(self):
        return {'nu':self.nu, 'xlim':self.xlim}








