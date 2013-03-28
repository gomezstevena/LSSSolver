import numpy as np
from numpy import linalg
import numbers, math, os
import cPickle as pickle

class BaseTrajectory (object):
    def __init__(self, system, time, dt, u0 = None, tol=1e-8, t0 = 0.0 ):

        if callable(system) and hasattr(system, 'dim') and hasattr(system, 'dfdu'):
            self.system = system
        else:
            raise TypeError('Require system to be compatible with ode.Ode type')

        self.dt = dt
        self.tol = tol
        self.t0 = t0
        self.u0 = u0

        
        if isinstance(time, numbers.Number):
            self.end_time = float(time)
        elif isinstance(time, tuple) and len(time) == 2:
            self.windup_time, self.end_time = time
        else:
            raise TypeError("Require either numeric or 2-tuple of numeric types for time paramater")


        self.N = int( math.ceil( self.end_time / self.dt ) ) - 1
        self.end_time = (self.N+1)*self.dt

        self.times = self.t0 + np.arange( 0, self.end_time, self.dt )



    def create(self):
        "Create the trajectory using Implicit-Trapezoidal scheme for integration"
        if self.windup:
            Nw = int( math.ceil( self.windup_time/self.dt ) ) - 1
            u0 = np.ones(self.system.dim) if self.u0 is None else self.u0
            u = trapIntegrate(self.system, u0, self.dt, Nw, tol=self.tol, t0 = self.t0-(Nw+1)*self.dt)
            u0 = u[-1]

        self.u  = trapIntegrate(self.system, u0, self.dt, self.N, tol=self.tol, t0=self.t0)

        return self

    @property 
    def windup(self):
        return hasattr(self, 'windup_time')

    @property 
    def dim(self):
        return self.system.dim

    def __getitem__(self, args):
        return self.u[args]

            
class Trajectory (BaseTrajectory):
    "Defines a trajectory that can be created or optionally loaded from file"
    def __init__(self, *args, **kwargs):
        self.filename = kwargs.pop('filename', 'data.traj' )
        if not self.filename.endswith('.traj'):
            self.filename += '.traj'

        create = kwargs.pop('create', False)


        if os.path.isfile(self.filename) and not create:
            with open(self.filename, 'rb') as data_file:
                data = pickle.load(data_file)
            self.dt = data['dt']
            u = data['u']
            self.N = len(u)-1
            self.u = u
            self.end_time = (self.N+1) * self.dt
            self.t0 = data['t0']
            self.tol = kwargs.pop('tol', 1e-8)
            self.u0 = self.u[0]
            self.system = data['system']
            self.times = self.t0 + np.arange( 0, self.end_time, self.dt )

        else:
            super(Trajectory, self).__init__(*args, **kwargs)
            self.create()
            self.save()

    def save(self):
        with open(self.filename, 'wb') as data_file:
            data = {'dt':self.dt, 'u': self.u, 't0':self.t0, 'system':self.system }
            pickle.dump(data, data_file, protocol=pickle.HIGHEST_PROTOCOL)




def trapIntegrate( f, u0, dt, N, tol=1e-8, t0 = 0.0):
    'Helper function to integrate ode using Implicit-Trapezoidal rule'

    u_traj = np.zeros( (N+1,f.dim) )
    u_traj[0] = u0

    u0 = u0.copy()
    f0 = f( u0, t0)


    I = np.eye(f.dim)

    for i in xrange(1, N+1):
        t = t0 + i*dt

        w = u0.copy()
        R = dt*f0
        nr = linalg.norm(R)
        while nr > tol: #Newton-Raphson loop
            dR = I - (dt/2.)*f.dfdu(w,t)
            w -= linalg.solve(dR, R)

            fw = f(w,t)
            R = w - u0 - (dt/2.)*( f0 + fw )
            nr = linalg.norm(R)

        f0 = fw
        u0 = w

        u_traj[i] = u0

    return np.array(u_traj)