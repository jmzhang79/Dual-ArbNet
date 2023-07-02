import os
from data import multiscaleMRI2data


class MRI2(multiscaleMRI2data.MRI2Data):
    def __init__(self, args, name='', train=True, benchmark=False):
        print(name)
        super(MRI2, self).__init__(
            args, name=name, train=train, benchmark=benchmark
        )

    def _scan(self):
        names_hr = super(MRI2, self)._scan()

        return names_hr

    def _set_filesystem(self, dir_data):
        super(MRI2, self)._set_filesystem(dir_data)
        self.dir_hr = os.path.join(self.apath)

