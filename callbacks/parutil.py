from util import LogCallback


class SaveLogCallback (LogCallback):
    def __init__(self, *args, **kwargs):
        self.outname = kwargs.pop('outname', 'restart_out.dat')
        super(SaveLogCallback, self).__init__(*args, **kwargs)


    def saveLog(self):
        super(SaveLogCallback, self).saveLog()
        if self.skip%self.i == 0:
            self.A.comm.writeParallel(self.outname, self.x)
            if self.MASTER:
                print self.i, 'Restart saved to', self.outname