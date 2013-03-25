from numpy import *
import time


class LogCallback(object):
    def __init__(self, A, b, fname = None, par=True, skip=1):
        self.i = 0
        self.A = A
        self.b = b
        self.x = None
        self.nb = A.normf(b)
        self.log = [ (self.i, self.nb, 0.0) ]
        self.time = time.time()
        self.fname = fname
        self.skip = skip
        self.dt = 0.0

        if par:
            from mpi4py import MPI
            self.MASTER = MPI.COMM_WORLD.rank == 0
        else:
            self.MASTER = True

    def __call__(self, x, r = None):
        self.i +=1
        r = self.b - self.A*x if r is None else r
        nr = self.A.normf(r)
        self.x = x
        
        tend = time.time()
        dt = tend - self.time
        self.time = tend
        self.dt += dt
        
        self.log.append( (self.i, nr, dt) )
        if self.MASTER and self.i%self.skip==0:
            print 'i:{:03d}\tr: {:.3e}\tlog10(r/r0): {:.2f}\tdt: {:0.3f}'.format(
                                                self.i, nr, log10(nr/self.nb), self.dt)
            self.dt = 0.0
        self.saveLog()

    def saveLog(self):
        if self.MASTER and self.fname is not None:
            save(self.fname, array(self.log).T )