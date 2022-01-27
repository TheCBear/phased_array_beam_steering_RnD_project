import scipy.io
from os.path import dirname, join as pjoin
from pathlib import Path
import numpy as np

class Antenna:
    def __init__(self, rel_path, filename):
        script_dir = str(Path().parent.absolute())
        data_dir = pjoin(script_dir, rel_path)
        mat_fname = pjoin(data_dir, filename)

        self.pattern = scipy.io.loadmat(mat_fname)
        return
    
    def get_theta_cut(self, phi):
        if not (phi in self.pattern['azimuth']):
            return False
        phi_2 = phi + 180
        if phi_2 > 180:
            phi_2 = -180 + (phi_2 - 180)
        phi_index_1 = np.argwhere(self.pattern['azimuth'][0,:] == phi)[0]
        phi_index_2 = np.argwhere(self.pattern['azimuth'][0,:] == phi_2)[0]

        cut = self.pattern['gain'][:, phi_index_1]
        cut = np.append(cut, self.pattern['gain'][::-1, phi_index_2])

        thetas = self.pattern['elevation']
        thetas = np.append(thetas, thetas[::-1]+180)
        thetas = thetas*2*np.pi/360
        return thetas, cut


