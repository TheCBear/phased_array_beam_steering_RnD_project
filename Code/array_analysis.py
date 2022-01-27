import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
from . import plot_utils
speed_of_light = 299792458 # m/s

class RectangularArray:
    def __init__(self, dx=0.02, dy=0.02, f_0=10E9, Nx=2, Ny=2,
            phi_res=300, theta_res=200):
        # Rectangular planar phased array
        # Default values are based on the default layout
        # of a 10 GHz phased array in the MATLAB antenna
        # toolbox.

        self.dx=dx # Element spacing in x direction [m]
        self.dy=dy # Element spacing in y direction [m]
        self.f_0 = f_0 # Center frequency [Hz]
        self.Nx = Nx # Number of elements in x direction
        self.Ny = Ny # Number of elements in y direction
        self.phi_res = phi_res # Simulation resolution in phi direction
        self.theta_res = theta_res # Simulation resolution in theta direction
        self.wave_length = speed_of_light/f_0 # Wave length derived from f_0
        self.k = 2*np.pi/self.wave_length # Wave number
        self.phi_null = 0
        self.theta_null = 0

        self.theta = np.linspace(0,np.pi, 200)
        self.phi = np.linspace(0, 2*np.pi, 300)
        self.THETA, self.PHI = np.meshgrid(self.theta, self.phi)

        self.beta_x = 0
        self.beta_y = 0
        self.W = np.ones((self.Nx, self.Ny)) # Elementwise amplitude weights
        self.phase_shifts = np.zeros((self.Nx, self.Ny)) # Element phase shifts
        self.W_complex = np.ones((self.Nx, self.Ny),complex)
        
        return


    def array_factor(self, theta, phi, 
                        theta_null, phi_null, normalize=True):
        M, N = np.shape(self.W)
        
        u = (np.pi*self.dx/self.wave_length) * \
            (np.sin(theta)*np.cos(phi) - \
            np.sin(theta_null)*np.cos(phi_null)
            )
        
        v = (np.pi*self.dy/self.wave_length) * \
            (np.sin(theta)*np.sin(phi) - \
            np.sin(theta_null)*np.sin(phi_null)
            )
        
        AF = np.zeros(np.shape(u),complex)

        for m in range(int(M/2)):
            for n in range(int(N/2)):
                #AF += self.W(m,n] * np.cos((2*(m+1)-1)*u) * np.cos((2*(n+1)-1)*v)
                AF += self.W[m+int(M/2),n+int(N/2)] * np.cos((2*(m+1)-1)*u) * np.cos((2*(n+1)-1)*v)
        AF = 4*AF
        if normalize:
            AF = AF/np.sum(self.W)
        return AF

    def get_S_xm(self, theta, phi):
        s_xm = 0
        for m_ in range(self.Nx):
            m = m_+1 # m for math, m_ for indexing
            I_m1 = self.W[m_,1]
            s_xm += I_m1*np.exp(1j*(m-1)*(self.k*self.dx*np.sin(theta)*np.cos(phi) + self.beta_x))
        return s_xm
    
    def get_S_yn(self, theta, phi):
        s_yn = 0
        for n_ in range(self.Ny):
            n = n_+1 # n for math, n_ for indexing
            I_1n = self.W[1,n_]
            s_yn += I_1n*np.exp(1j*(n-1)(self.k*self.dy*np.sin(theta)*np.sin(phi) + self.beta_y))
        return s_yn

    def plot_array_factor_3D(self, scale='logarithmic', theta_null=0, phi_null=0, min_dB=None):
        #AF = array_factor(self.W, # previously self.W_complex
        #    self.THETA, self.PHI, 
        #    self.dx, self.dy, 
        #    self.wave_length,theta_null,phi_null)
        AF = self.array_factor(self.THETA, self.PHI, 
            theta_null,phi_null)
        if scale=='logarithmic':
            R_dB =  20*np.log10(np.abs(AF))
            if not (min_dB == None):
                R_dB = np.clip(R_dB, min_dB, None)
            R_dB_min = np.min(R_dB)
            R = R_dB-R_dB_min
            actual_range = (R_dB_min, np.max(R_dB))
        elif scale=='linear':
            R = np.abs(AF)
            actual_range = (0,0)
        fig, ax = plot_utils.spherical_surface_plot(R, self.THETA, self.PHI, actual_range=actual_range)
        return fig, ax
    
    def plot_array_factor_vertical(self,theta_null=None, phi_null=None, scale='dB', polar=True, n_points=720, annotate=True):
        if phi_null == None:
            phi_null = self.phi_null
        if theta_null == None:
            theta_null = self.theta_null
        
        _, theta, phi = plot_utils.theta_plane(theta_null, 0, n_points=n_points)
        r_raw = self.array_factor(theta, phi, theta_null, phi_null)
        
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1, polar=polar)
        #ax = plt.axes(polar=polar)
        if (scale.lower() == 'db') or (scale.lower() == 'logarithmic'):
            r = 20*np.log10(np.abs(r_raw))
        else:
            r = np.abs(r_raw)
        ax.plot(theta, r)
        if scale.lower() == 'db' or scale == 'logarithmic':
            if polar == True: 
                ax.set_ylabel('Array factor [dB]', labelpad=30)
            if polar == False:
                ax.set_ylabel('Array factor [dB]')
        else:
            if polar == False:
                ax.set_ylabel('Absolute array factor')
            if polar == True:
                ax.set_ylabel('Absolute array factor', labelpad=30)

        if polar:
            ax.set_theta_offset(np.pi/2)
        else:
            ax.set_xlabel('Angle')
        
        endpoints = (-np.pi+ theta_null, np.pi + theta_null)
        if annotate:
            annotate_beamwidth(ax, np.abs(r_raw), scale=scale, endpoints=endpoints)
            annotate_peaks(ax, np.abs(r_raw), scale=scale, endpoints=endpoints)
        return fig, ax

    def get_af_main_beam(self, theta_null=None, phi_null=None, n_points=720, height_of_width=0.707):
        if theta_null == None:
            theta_null = self.theta_null
        if phi_null == None:
            phi_null = self.phi_null
        
        _, theta, phi = plot_utils.theta_plane(theta_null, phi_null, n_points=n_points)
        af = np.abs(self.array_factor(theta, phi, theta_null, phi_null))
        peaks, properties = sig.find_peaks(af)
        max_peak_index = np.argmax(af[peaks])
        prominences, _, _ = sig.peak_prominences(af, peaks)
        prominence = prominences[max_peak_index]
        amplitude = af[peaks[max_peak_index]]
        amplitude_b = np.max(af)
        widths, heights, _, _ = sig.peak_widths(af, peaks, rel_height=1-height_of_width)
        width = widths[max_peak_index]*2*np.pi/n_points
        width_height = heights[max_peak_index]

        details = { 'peak_theta': theta[peaks[max_peak_index]],
                    'prominence': prominence,
                    'amplitude': amplitude,
                    'amplitude_b': amplitude_b,
                    'width': width,
                    'width_height': width_height
                }
        return details
    
    def get_side_lobe_level(self, theta_null=None, phi_null=None, n_points=720):
        if theta_null == None:
            theta_null = self.theta_null
        if phi_null == None:
            phi_null = self.phi_null
        
        _, theta, phi = plot_utils.theta_plane(theta_null, phi_null, n_points=n_points)
        af = np.abs(self.array_factor(theta, phi, theta_null, phi_null))
        return side_lobe_level(af)




class ArrayController():
    def __init__(self,array=None):
        if array == None:
            self.array = RectangularArray()
        else:
            self.array = array
        return



class LinearPhaseController(ArrayController):
    def __init__(self, array=None):
        super().__init__(array)
        return
    
    def set_steering_direction(self,theta,phi, N_taylor=0):

        # Choose numpy implementation of cos and sin or use taylor expansion
        if N_taylor == 0:
            cos = np.cos
            sin = np.sin
        else:
            cos = lambda x : cos_taylor(x, N_taylor)
            sin = lambda x : sin_taylor(x, N_taylor)
        
        # Calculate progressive phase shift in x and y directions
        self.array.beta_x = -self.array.k*self.array.dx*sin(theta)*cos(phi)
        self.array.beta_y = -self.array.k*self.array.dy*sin(theta)*sin(phi)

        # Calculate phase shifts for all elements and assign complex weights
        for nx in range(self.array.Nx):
            for ny in range(self.array.Ny):
                phase = self.array.beta_x*nx + self.array.beta_y*ny # Is it correct?
                self.array.phase_shifts[nx, ny] = phase      
                self.array.W_complex[nx, ny] = np.cos(phase) + 1j*np.sin(phase)
        return self.array.W_complex


class LinearArray():
    def __init__(self, wave_length=1, d=0.5, beta=0, N=10, a=np.array([None])) -> None:
        self.wave_length = wave_length
        self.d = d
        self.beta = beta
        self.N = N
        if a.any() == None:
            self.a = np.ones(N)
        else:
            self.a = a
        self.k = 2*np.pi/wave_length
    
    def array_factor1(self, theta):
        # Balanis p. 326
        u = np.pi*self.d*np.cos(theta)/self.wave_length
        M = int(np.math.floor(self.N/2))

        if self.N%2 == 0:
            AF = np.sum([self.a[n]*np.cos((2*(n+1)-1)*u) for n in range(M)])
        else:
            AF = np.sum([self.a[n]*np.cos((2*(n+1)-1)*u) for n in range(M+1)])
        return AF
    
    def array_factor(self,theta):
        # Balanis p. 319
        psi = self.k*self.d*np.cos(theta)+self.beta
        #af = np.sum([self.a[n]*np.exp(1j*(n)*psi) for n in range(self.N)])
        af = np.sum([self.a[n-1]*np.exp(1j*(n-1)*self.beta)*np.exp(1j*(n-1)*self.k*self.d*np.cos(theta)) for n in range(1,self.N+1)])
        return af

    def plot_array_factor(self, projection=None):
        thetas = np.linspace(0, 2*np.pi, 720).flatten()
        af = [self.array_factor(theta) for theta in thetas]
        af = 20*np.log10(np.abs(af))
        ax = plt.axes(projection=projection)
        ax.plot(thetas, af)
        return ax

######################################################
# Helper functions
######################################################


def array_factor(w, theta, phi, dx, dy, wave_length, 
                        theta_null, phi_null):
    M, N = np.shape(w)
    
    u = (np.pi*dx/wave_length) * \
        (np.sin(theta)*np.cos(phi) - \
        np.sin(theta_null)*np.cos(phi_null)
        )
    
    v = (np.pi*dy/wave_length) * \
        (np.sin(theta)*np.sin(phi) - \
        np.sin(theta_null)*np.sin(phi_null)
        )
    
    AF = np.zeros(np.shape(u),complex)

    for m in range(int(M/2)):
        for n in range(int(N/2)):
            AF += w[m,n] * np.cos((2*(m+1)-1)*u) * np.cos((2*(n+1)-1)*v)
    AF = 4*AF
    return AF



def beamwidth(arr, index_of_max=None, scale='linear'):
    #peaks, _ = sig.find_peaks(arr)
    #peak_vals = [arr[peak] for peak in peaks]
    max_index = [np.argmax(arr)]
    width = sig.peak_widths(arr, max_index)
    return width



def annotate_beamwidth(ax, arr, scale='dB', endpoints=(-np.pi, np.pi)):
    # Find peaks 
    peaks, _ = sig.find_peaks(arr)
    # Find the widths of the peaks that were found
    widths, heights, left_ips, right_ips = sig.peak_widths(arr, peaks, rel_height=0.5)
    
    left_limit, right_limit = endpoints
    scan_range = right_limit - left_limit
    
    # Scale the horizontal lines
    left_ips = (left_ips)*scan_range/len(arr)
    right_ips = (right_ips)*scan_range/len(arr)
    
    # Offset the lines
    left_ips += left_limit
    right_ips += left_limit

    #left_ips = left_ips*2*np.pi/len(arr)
    #right_ips = right_ips*2*np.pi/len(arr)

    # Scale the vertical direction in dB if selected
    if scale.lower() == 'db':
        heights = 20*np.log10(heights)

    # Combine the parameters
    results_half = widths, heights, left_ips, right_ips

    # Plot the widths as horizontal lines
    ax.hlines(*results_half[1:], color='C2')


def annotate_peaks(ax, arr, scale='dB', endpoints=(-np.pi, np.pi)):
    peaks, _ = sig.find_peaks(arr)
    prominences = sig.peak_prominences(arr, peaks)[0]
    contour_heights = arr[peaks] - prominences

    peak_heights = arr[peaks]
    if scale.lower() == 'db':
        peak_heights = 20*np.log10(peak_heights)
        contour_heights = 20*np.log10(contour_heights)

    left_limit, right_limit = endpoints
    scan_range = right_limit - left_limit

    peaks_angle = peaks*scan_range/len(arr)
    peaks_angle += left_limit

    ax.plot(peaks_angle, peak_heights, "x")
    ax.vlines(x=peaks_angle, ymin=contour_heights, ymax=peak_heights)

    return


def side_lobe_level(arr, full_circle=True, mask=None):
    peaks, _ = sig.find_peaks(arr)
    max_peak_number = np.argmax(arr[peaks])
    max_peak_index = peaks[max_peak_number]
    max_peak_val = arr[max_peak_index]
    arr_len = len(arr)
    side_lobe_max = 0

    # Generate mask to exclude peaks in the main lobe and back lobes
    if mask == None:
        mask = np.zeros((arr_len, 1))
        _, _, left_ips, right_ips = sig.peak_widths(arr, peaks)
        main_left = left_ips[max_peak_number]
        main_right = right_ips[max_peak_number]

        if full_circle:
            back_lobe_index = (max_peak_index + arr_len/2)%arr_len
        
        for i in range(arr_len):
            # Exclude peaks in main lobe
            if i >= main_left and i <= main_right:
                mask[i] = 1
            # Exclude back lobes
            if full_circle:
                if np.abs(i - back_lobe_index) < (arr_len/4) - 2:
                    mask[i] = 1
            else:
                pass

    # Find maximum peak
    for peak in peaks:
        if mask[peak]:
            pass
        else:
            side_lobe_max = arr[peak] if arr[peak] > side_lobe_max else side_lobe_max
    
    # Calculate side lobe level from main peak and maximum side lobe
    side_lobe_level = side_lobe_max/max_peak_val

    return side_lobe_level


def beam_stats(pattern, theta, symmetry_angle=0):
    n_points = len(pattern)
    peaks, properties = sig.find_peaks(pattern)
    max_peak_index = np.argmax(pattern[peaks])
    prominences, _, _ = sig.peak_prominences(pattern, peaks)
    prominence = prominences[max_peak_index]
    amplitude = pattern[peaks[max_peak_index]]
    amplitude_b = np.max(pattern)
    widths, heights, _, _ = sig.peak_widths(pattern, peaks)
    width = widths[max_peak_index]*2*np.pi/n_points
    width_height = heights[max_peak_index]

    # Find the steering angle from the peak index and evaluation angles
    steer_angle = theta[peaks[max_peak_index]]
    # Account for symmetry in the pattern to consistently find the same maximum
    # and not the mirrored beam.
    if symmetry_angle == 0:
        if steer_angle < 0:
            steer_angle = -steer_angle
        if steer_angle > np.pi:
            steer_angle = steer_angle - 2*(steer_angle - np.pi)
    elif symmetry_angle == 90 or symmetry_angle == np.pi/2:
        if steer_angle < 0:
            # add a complete rotation to bring the angle into positives
            steer_angle += 2*np.pi 
        if steer_angle > np.pi/2 and steer_angle < (3/2)*np.pi:
            # Mirror around 90 degrees
            steer_angle = np.pi - steer_angle
            # The result may be a negative angle if angle > 180 deg.
            # This is not a problem

    # Put parameters into a dictionary
    details = { 'peak_theta': theta[peaks[max_peak_index]],
                'prominence': prominence,
                'amplitude': amplitude,
                'amplitude_b': amplitude_b,
                'width': width,
                'width_height': width_height,
                'steering_angle' : steer_angle
            }
    return details

def print_beam_stats(pattern, theta):
    stats = beam_stats(pattern, theta)
    print(f"Peak angle: {stats['peak_theta']:.3f} " 
            f"({stats['peak_theta']*360/(2*np.pi):.2f} deg.)")
    print(f"Peak max: {stats['amplitude']:.3f}")
    print(f"Steering angle: {stats['steering_angle']:.3f} "
            f"({stats['steering_angle']*360/(2*np.pi):.2f} deg.)")
    print(f"Beam width: {stats['width']:.3f} "
            f"({stats['width']*360/(2*np.pi):.2f} deg.)")
    print(f"\n")
    

def dolph_tschebyscheff_1d_side_lobe_level(N, side_lobe_level):
    R_0 = 10**(side_lobe_level/20)
    z_0 = np.cosh(np.arccosh(R_0)/(N-1))
    print(f"R_0 = {R_0:.5f}, z_0 = {z_0:.5f}")
    if N % 2 == 0:
        a = _dolph_tschebyscheff_a_even(z_0, N)
    else:
        raise ValueError('Dolph-Tschebyscheff not implemented for odd N')
        #a = _dolph_tschebyscheff_a_odd(z_0, N)

    return a

def _dolph_tschebyscheff_a_even(z_0, N):
    # Balanis p. 338 and Barbiere
    M = int(N/2)
    a = np.zeros((N,1))
    a_tmp = np.zeros((M,1))
    for n in range(1,M+1):
        for q in range(n, M+1):
            part1 = pow(-1,(M-q))*pow(z_0,(2*q-1))
            part2 = np.math.factorial(q+M-2)*(2*M-1)
            part3 = np.math.factorial(q-n)*np.math.factorial(q+n-1)*np.math.factorial(M-q)
            a_tmp[n-1,0] += part1*part2/part3
    a = np.concatenate((a_tmp[::-1,:], a_tmp[:,:]))
    return a.T/np.max(a)

def _dolph_tschebyscheff_a_odd(z_0, N):
    raise ValueError('Dolph-Tschebyscheff not implemented for odd N')


# def beamwidth_db(arr, index_of_max=None):
#     if index_of_max == None:
#         index_0 = np.argmax(arr)
#         maximum = np.max(arr)
#     else:
#         index_0  = index_of_max
#         maximum = arr[index_of_max]
    
#     low = index_0 - 1
#     high = index_0 + 1

#     while arr[low] > maximum - 3:
#         low -= 1 # negative indexes behave nicely in this case
#         if low == index_0:
#             return None
#     # Choose the closest to 3 dB
#     low = low if arr[low]-3 < arr[low+1]-3 else low+1

#     while arr[high] > maximum - 3:



### COS and SIN using taylor expansion ###

def sin_taylor(x, N_iter):
    y = x
    sign = 1
    for n in range(1,N_iter+1): # range does not include the last value
        sign = -1*sign
        exponent = 2*n+1
        y += sign*np.power(x,exponent)/np.math.factorial(exponent)
    return y

def cos_taylor(x, N_iter):
    y=1
    sign  = 1
    for n in range(1, N_iter+1):
        sign = -1*sign
        exponent = 2*n
        y += sign*np.power(x,exponent)/np.math.factorial(exponent)
    return y
