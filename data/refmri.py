import os
from data import multiscaleRefMRIdata


class RefMRI(multiscaleRefMRIdata.RefMRIData):
    def __init__(self, args, name='', train=True, benchmark=False):
        print(name)
        super(RefMRI, self).__init__(
            args, name=name, train=train, benchmark=benchmark
        )

    def _scan(self):
        names_hr = super(RefMRI, self)._scan()

        return names_hr

    def _set_filesystem(self, dir_data):
        super(RefMRI, self)._set_filesystem(dir_data)
        self.dir_hr = os.path.join(self.apath)

