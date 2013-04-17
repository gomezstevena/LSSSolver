#!/usr/bin/env python
from numpy import *
from mpi4py import MPI
from numpy.random import rand
from scipy.linalg import norm
from scipy.sparse import linalg as splinalg
#import time, sys, os

from ..lss import MGrid


MASTER = ( MPI.COMM_WORLD.rank == 0 )

class MPIComm(object):
    
    def __init__(self, shape = None, top=False):
        self._COMM = MPI.COMM_WORLD
        self.size = self._COMM.size
        self.rank = self._COMM.rank
        self.even = self.rank%2 == 0
        self._COMM.Barrier()

        #print "before shape =", shape

        shape = self.broadcast( shape )
        if not top:
            shape = shape[0]*self.size, shape[1]
        #print "Creating COMM object rank: {0:d}, shape:{1}".format(self.rank, shape)


        self.shape = self.N, self.M = shape

        if self.N % self.size != 0:
            raise ValueError("Sizes not consistent with data size")

        self.chunk = self.N // self.size

        self.start_node = self.rank==0
        self.end_node = self.rank==self.size-1

    def broadcast(self, data):
        return self._COMM.bcast( data, 0 )

    def splitArray( self, data = None ):
        """ Split data of length N or N+1 across all of the nodes
        """
        if data is not None and self.start_node:
            data = data.reshape( (-1, self.M) )

            n = data.shape[0]

            if n == self.N:
                sdata = array([0])
                self._COMM.Bcast( (sdata,1,MPI.INT), 0 )
                for proc in xrange(1, self.size):
                    start = proc*self.chunk - 1
                    end = start + self.chunk+2
                    if end > self.N:
                        end = self.N
                        local_data = zeros( (self.chunk+2, self.M) )
                        local_data[:-1] = data[start:end]
                    else:
                        local_data = data[start:end]

                    self._COMM.Send( (local_data, local_data.size, MPI.DOUBLE), dest=proc )


                local_data = zeros( (self.chunk+2, self.M) )
                local_data[1:] = data[:self.chunk+1]
                self._COMM.Barrier()
                return local_data.copy()


            elif n == self.N+1:
                sdata = array([1])
                self._COMM.Bcast( (sdata,1,MPI.INT), 0 )

                for proc in xrange(1, self.size):
                    start = proc * self.chunk
                    end = start + self.chunk+1
                    if end > self.N:
                        local_data = zeros( (self.chunk+1,self.M) )
                        local_data[:] = data[start:]
                    else:
                        local_data = data[start:end]

                    self._COMM.Send( (local_data, local_data.size, MPI.DOUBLE), dest=proc )

                local_data = data[:self.chunk+1]
                self._COMM.Barrier()
                return local_data.copy()

            else:
                raise ValueError("Requre data length to be a multiple of N or N+1, N = {0}, data.shape = {1}".format(self.N, data.shape))

        else:
            # First determine size of final array
            sdata = array([0])
            self._COMM.Bcast( (sdata,1,MPI.INT), 0 )
            off = sdata[0]

            local_data = zeros( (self.chunk+2-off, self.M) )
            self._COMM.Recv( (local_data, local_data.size, MPI.DOUBLE), source=0 )
            self._COMM.Barrier()
            return local_data

    def pad(self, data):
        assert data.ndim == 2 and len(data)>2
        N, M = data.shape
        out = zeros( (N+2, M)  )
        out[1:-1] = data
        if N == self.chunk:
            self.fixOverlap(out)
            return out
        elif N == self.chunk+1:
            if not self.end_node:
                self._COMM.Send( (data[-2], M, MPI.DOUBLE), dest=self.rank+1 )
            if not self.start_node:
                self._COMM.Recv( (out[0], M, MPI.DOUBLE), source=self.rank-1 )

            if not self.start_node:
                self._COMM.Send( ( data[1], M, MPI.DOUBLE), dest = self.rank-1 )
            if not self.end_node:
                self._COMM.Recv( ( out[-1], M, MPI.DOUBLE), source=self.rank+1 )
            return out
        else:
            raise ValueError('bad shape')


    def fixOverlap(self, data, add=False): #if this is an offset array can add boundary arrays

        if data.ndim == 1:
            if data.size == self.chunk+2 or data.size==self.chunk+1:
                data = data.reshape( (-1, 1) )
                M = 1
            else:
                data = data.reshape((-1, self.M))
                M = self.M


            N = len(data)
        elif data.ndim == 2:
            N, M = data.shape
        else:
            raise ValueError('Bad dimensions')

        if N == self.chunk+2:

            if self.start_node and not self.end_node:
                self._COMM.Sendrecv( data[-2], dest=self.rank+1, recvbuf=data[-1], source=self.rank+1 )
            elif self.end_node and not self.start_node:
                self._COMM.Sendrecv( data[ 1], dest=self.rank-1, recvbuf=data[ 0], source=self.rank-1 )
            elif not (self.start_node or self.end_node):
                if self.even:
                    self._COMM.Sendrecv( data[-2], dest=self.rank+1, 
                                         recvbuf=data[-1], source=self.rank+1)
                    self._COMM.Sendrecv( data[ 1], dest=self.rank-1, 
                                         recvbuf=data[ 0], source=self.rank-1)
                else:
                    self._COMM.Sendrecv( data[ 1], dest=self.rank-1, 
                                         recvbuf=data[ 0], source=self.rank-1)
                    self._COMM.Sendrecv( data[-2], dest=self.rank+1, 
                                         recvbuf=data[-1], source=self.rank+1)

            if self.start_node:
                data[ 0] = 0.0
            elif self.end_node:
                data[-1] = 0.0
            

        elif N == self.chunk+1:
            """
            if not self.end_node:
                self._COMM.Send( (data[-1], M, MPI.DOUBLE), dest = self.rank+1 )
            if not self.start_node:
                buff = zeros(M) if add else data[0]
                self._COMM.Recv( (buff, M, MPI.DOUBLE), source=self.rank-1 )
                if add:
                    data[0] += buff

            if add:

                if not self.start_node:
                    self._COMM.Send( (data[0], M, MPI.DOUBLE), dest = self.rank-1 )
                if not self.end_node:
                    self._COMM.Recv( (data[-1], M, MPI.DOUBLE), source=self.rank+1 )
            """
            if add:
                buff = data[0].copy()

            if self.start_node and not self.end_node:
                self._COMM.Send( data[-1],  dest =self.rank+1 )
            elif self.end_node and not self.start_node:
                self._COMM.Recv( data[ 0], source=self.rank-1 )
            elif not ( self.start_node or self.end_node ):
                self._COMM.Sendrecv( data[-1], dest=self.rank+1, recvbuf=data[0], source=self.rank-1 )

            
            if add:
                if not self.start_node:
                    data[0] += buff

                if self.end_node and not self.start_node:
                    self._COMM.Send( data[0], dest=self.rank-1 )
                elif self.start_node and not self.end_node:
                    self._COMM.Recv( data[-1], source=self.rank+1 )
                elif not (self.start_node or self.end_node):
                    self._COMM.Sendrecv( data[0], dest=self.rank-1, recvbuf=data[-1], source=self.rank+1 )



        else:
            raise ValueError("Data must have N/D+2 or N/D+1 timesteps, N/D = {0:d}, data_len = {1:d}".format(self.chunk, N) )


    def parMap(self, func, data, mid=False):
        data = data.reshape( (-1, self.M) )

        if mid:
            data = 0.5*( data[1:] + data[:-1] )
        
        f_data = array([ func(d) for d in data ])

        z = zeros( (1,self.M) )
        final = vstack( [z, f_data, z] )
        self.fixOverlap(final)
        return final


    def printInSequence(self, data):
        data = data.reshape((-1, self.M))

        for i in xrange(self.size):
            if self.rank == i:
                for d in data:
                    print '[', ','.join('{0:.3f}'.format(k) for k in d), ']'
                print '-'*50

            self._COMM.Barrier()

    def collect(self, data, dest=0):
        if data.size == (self.chunk+2)*self.M:
            data = data.reshape( (self.chunk+2, self.M) )
            ldata = data[1:-1]

            all_data = zeros( (self.N, self.M) )
            self._COMM.Allgather( ldata, all_data )
            return all_data
        elif data.size == (self.chunk+1)*self.M:
            data = data.reshape((self.chunk+1, self.M))
            i = 0 if self.start_node else 1
            ldata = data[i:]
            all_data = zeros( (self.N+1, self.M) )
            v = [self.chunk]*self.size; v[0]+=1
            all_data = self._COMM.allgather( ldata, all_data)
            all_data = vstack(all_data)
            return all_data

    def readParallel(self, fname, copy=True, dtype=float64, off=0):
        assert off==1 or off==0
        N, M = self.shape

        item_size = array( [0], dtype=dtype).itemsize
        data_size = self.chunk*M*item_size

        start = self.rank*data_size
        data = memmap(fname, dtype=dtype, mode='r', offset=start, shape=(self.chunk+off, M), order='C')

        if off == 1 and copy:
            data = data.copy()
        elif off == 0:
            data = vstack( [ zeros(M), data, zeros(M) ] )
            self.fixOverlap(data)

        return data

    def writeParallel(self, fname, data):
        N, M = self.shape
        data = data.reshape( (-1, M) )

        if len(data) == self.chunk+2:
            data = data[1:-1]
            off = 0
        elif len(data) == self.chunk+1:
            st = 0 if self.start_node else 1
            data = data[st:]
            off = 1
        
        data_size = self.chunk*M*data.itemsize
        start = self.rank*data_size

        if not os.path.isfile(fname) and self.start_node:
            print 'Creating file', fname, '...',
            disk_data = memmap(fname, dtype=data.dtype, mode='w+', shape=self.shape )
            del disk_data
            print 'Done'
        
        self.Barrier()

        disk_data = memmap(fname, dtype=data.dtype, mode='w+', offset=start, shape=(self.chunk+off, M) )
        disk_data[:] = data

        del disk_data

        self.Barrier()


    Barrier = MPI.COMM_WORLD.Barrier

    def sum(self, x):
        data_in = array((x,))
        data_out = array((0.0,))
        self._COMM.Allreduce(data_in, data_out)
        return data_out[0]

    def parIter(self, x):
        for proc in xrange(self.size):
            if self.rank==proc:

                for i in xrange(len(x)-2):
                    yield x[i+1]

            self._COMM.Barrier()



class MGridParallel(MGrid):
    def __init__(self, ns, dt, traj, shape = None, top=False):
        self.comm = MPIComm(shape, top)
        
        if top:
            if isinstance(traj, str):
                traj_local = self.comm.readParallel(traj, off=1)
            elif isinstance(traj, ndarray) or traj is None: 
                traj_local = self.comm.splitArray(traj)
        else:
            traj_local = traj

        MGrid.__init__( self, ns, dt, traj_local, shape)

        if hasattr(self, 'perturbation'):
            self.perturbation = r_[ zeros(self.m), self.perturbation.ravel(), zeros(self.m) ]
            self.comm.fixOverlap(self.perturbation)

        size = (self.n+2)*self.m
        self.SchurOperator = \
            splinalg.LinearOperator( (size,size),
                                    matvec = self.schur, dtype=float)

        self._base = MGrid


    def dot(self, a, b):
        'Space-Time inner product done in parallel'
        a = a.reshape((-1, self.m))
        b = b.reshape((-1, self.m))
        ldot = self._base.dot( self, a[1:-1].ravel() , b[1:-1].ravel() )
        return self.comm.sum(ldot)

    def normt(self, x):
        '2-Norm of solutions at each timestep done in parallel'
        x = x.reshape((-1, self.m))
        d = zeros(len(x))
        d[1:-1] = self._base.normt(self, x[1:-1])
        self.comm.fixOverlap(d)
        return d

    def matBTvec(self, w):
        'Parallel Application of adjoint system across all time'
        w = w.reshape( (-1, self.m) )
        out = self._base.matBTvec( self, w )
        self.comm.fixOverlap(out, add = True)
        return out

    def schur(self, w):
        'Parallel Application of schur complement system across all time'
        w = w.reshape( (-1, self.m) )
        out = zeros_like(w)
        out[1:-1] = self._base.schur(self, w[1:-1] ).reshape( (-1, self.m) )
        self.comm.fixOverlap( out )
        return out.ravel()

    def BT(self, w):
        w = w.reshape((-1, self.m) )
        v = self.matBTvec( w[1:-1] ).reshape( (-1, self.m) )
        return v

    __call__ = __mul__ = schur

    def restrictTime(self, u):
        'Time Restriction done in parallel'
        u = u.reshape((-1, self.m))
        n = len(u)
        
        if n == self.n+2:
            uc = zeros( (2+self.n//2, self.m) )
            uc[1:-1] = self._base.restrictTime(self, u[1:-1])
            self.ns_coarse.comm.fixOverlap(uc)
            return uc
        elif n == self.n+1:
            return self._base.restrictTime(self, u)

    def interpolateTime(self, u):
        'Time Interpolation done in parallel'
        n, m = u.shape

        if n == self.n//2+2:
            uf = hstack((u,u)).reshape((-1,m))[1:-1]
        else:
            print u.shape, self.n//2 + 2
            assert False

        return uf

    def mapTraj(self, func):
        return self.comm.parMap(func, self.traj, mid=True)

