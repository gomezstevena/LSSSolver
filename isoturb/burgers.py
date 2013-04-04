import numpy as np

class Burgers (object):
    """
    Represents 1D Burgers Equations, du/dt + 1/2 d/dx(u**2) = nu d^2/dx^2(u)
    domain goes from [0, xlim), in Nx steps, dx = xlim/Nx, BCs are periodic
    """
    def __init__(self, Nx, nu, xlim = 2*np.pi):
        self.Nx = Nx
        self.xlim = float(xlim)
        self.nu = nu
        self.dx = self.xlim / self.Nx


    def _ddx(self, u):
        dudx = np.roll(u, 1, -1) - np.roll(u, -1, -1)
        dudx /= 2*self.dx
        return dudx

    def _d2dx2(self, u):
        d2udx2 = np.roll(u,1,-1) + np.roll(u,-1,-1)
        d2udx2 += 2*u
        d2udx2 /= self.dx*self.dx
        return d2udx2

    def dudt(self, u, t=0):
        return -self._ddx( 0.5*u*u ) + self.nu * self._d2dx2(u)

    __call__ = dudt

    def Tan(self, u, du):
        return -self._ddx(u*du) + self.nu * self._d2dx2(du)

    def Adj(self, u, du):
        return u*self._ddx(du) + self.nu * self._d2dx2(du)

    def dfdnu(self, u, t=0):
        return self._d2dx2(u)

    @property 
    def fixed_params(self):
        return {'nu':self.nu, 'xlim':self.xlim}








