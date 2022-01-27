import plot_utils
import numpy as np


wave_length = 1
k = 2*np.pi/wave_length # Wave number
d = 1/2 # Distance between elements

def planar_array_factor(w, theta, phi, dx, dy, wave_length, 
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
    
    AF = np.zeros(np.shape(u))

    for m in range(int(M/2)):
        for n in range(int(N/2)):
            AF += w[m,n] * np.cos((2*(m+1)-1)*u) * np.cos((2*(n+1)-1)*v)
    AF = 4*AF
    return AF


# Inspired by https://stackoverflow.com/questions/36816537/spherical-coordinates-plot-in-matplotlib
# and https://matplotlib.org/2.0.2/examples/mplot3d/surface3d_radial_demo.html

theta, phi = np.linspace(0,2*np.pi, 200), np.linspace(0, np.pi, 300)
THETA, PHI = np.meshgrid(theta, phi)

w = np.ones((4,4))
wave_length = 1
dx = wave_length/2
dy = wave_length/2

R = planar_array_factor(w, THETA, PHI, dx, dy, wave_length, 0, 0)

plot_utils.spherical_surface_plot(R,THETA, PHI)
