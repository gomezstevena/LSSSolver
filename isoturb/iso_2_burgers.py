import numpy as np
import isoturb
from matplotlib import pyplot as plt
from matplotlib import animation



if __name__ == '__main__':
    k = 16
    nu = 0.001
    dt = np.pi / 16.

    NT = 2048

    ns = isoturb.IF3DJacob(k, nu)

    DIR = "/home/gomezs/IsoTurbData/"

    m = 4*k*k*(k+1) * 3 * 2
    traj = np.memmap(DIR + 'traj_{:d}.dat'.format(NT), dtype=np.float64, 
                                                     mode='r', shape=(NT+1, m) )

    """burg_traj = np.zeros( (NT+1, 3*k) )

    
    for i in xrange(NT+1):
        u_h, v_h, w_h = ns.unravel( traj[i] )
        u = ns.c2f(u_h)

        Ub = u[:,0,0]
        burg_traj[i, :] = Ub

        print i, u.shape

    np.save('btraj_{:d}.npy'.format(NT), burg_traj )
    """
    burg_traj = np.load('btraj_{:d}.npy'.format(NT) )

    x = np.linspace(0, 2*np.pi, 3*k)

    fig = plt.figure()
    h, = plt.plot(x, np.zeros(3*k), '-b', lw=2)
    plt.xlim(0, 2*np.pi)
    plt.ylim(-0.1, 0.1)
    
    def animate(i):
        h.set_ydata( burg_traj[i,:] )
        return h,

    anim = animation.FuncAnimation(fig, animate, frames=NT+1, interval=1, blit=True )

    plt.show()
    