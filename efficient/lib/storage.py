import tempfile
import h5py
import os

TMPDIR = os.environ.get('TMPDIR','.')

class H5TmpFile(h5py.File):
    def __init__(self, filename=None, *args, **kwargs):
        if filename is None:
            tmpfile = tempfile.NamedTemporaryFile(dir=TMPDIR)
            filename = tmpfile.name
        h5py.File.__init__(self, filename, *args, **kwargs)
    def __del__(self):
        self.close()

class _Xlist(list):
    def __init__(self):
        self.scr_h5 = H5TmpFile()
        self.index = []
    
    def __getitem__(self, n):
        key = self.index[n]
        return self.scr_h5[key].value

    def append(self, x):
        key = str(len(self.index) + 1)
        if key in self.index:
            for i in range(len(self.index)+1):
                if str(i) not in self.index:
                    key = str(i)
                    break
        self.index.append(key)
        self.scr_h5[key] = x
        self.scr_h5.flush()

    def __setitime__(self, n, x):
        key = self.index[n]
        self.scr_h5[key][:] = x
        self.scr_h5.flush()

    def __len__(self):
        return len(self.index)

    def pop(self, index):
        key = self.index.pop(index)
        del(self.scr_h5[key])
