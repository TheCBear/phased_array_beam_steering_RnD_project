import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as axes3d
from matplotlib import cm, colors
import numpy as np


def spherical_surface_plot(R, THETA, PHI, 
    fig=None, ax=None, subplot_position=None, actual_range=(0,0)):
    if(fig == None and ax == None):
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1, projection='3d')
    elif(fig != None and ax == None):
        ax = fig.add_subplot(subplot_position[0],
            subplot_position[1],
            subplot_position[2], 
            projection='3d')

    (X,Y,Z) = spherical_to_cartesian(R,THETA,PHI)
    
    axis_size = 1.4

    maxR = np.max(R)
    maxAbsR = np.max(np.abs(R))

    axis_min = -maxAbsR*axis_size
    axis_max = maxAbsR*axis_size

    ax.set_xlim(axis_min,axis_max)
    ax.set_ylim(axis_min,axis_max)
    ax.set_zlim(axis_min,axis_max)

    norm = colors.Normalize(
        vmin=np.min(np.abs(R)),
        vmax=np.max(np.abs(R)), 
        clip=False)
    
    if actual_range == (0,0):
        signed_norm = colors.Normalize(
            vmin=np.min(R),
            vmax=np.max(R), 
            clip=False)
    else:
        rmin, rmax = actual_range
        signed_norm = colors.Normalize(
            vmin=rmin,
            vmax=rmax, 
            clip=False)

    plot = ax.plot_surface(
        X,Y,Z, rstride=1, cstride=1, cmap='jet',
        linewidth=5, edgecolor='black', antialiased=True, alpha=1,
        facecolors=cm.jet(norm(np.abs(R))))
    fig.colorbar(cm.ScalarMappable(norm=signed_norm, cmap=cm.jet), label="dB")

    # Plot axes
    
    ax.plot((0,0), (0,0), (-maxAbsR*axis_size, maxAbsR*axis_size),
            'b', label='z-axis')
    ax.plot((0,0), (-maxAbsR*axis_size, maxAbsR*axis_size), (0,0),
            'g', label='y-axis')
    ax.plot((-maxAbsR*axis_size, maxAbsR*axis_size), (0,0), (0,0),
            'r', label='x-axis')
    ax.legend()
    ax.set_box_aspect((1,1,1))
    plt.show()
    return (fig, ax)

def spherical_to_cartesian(R, THETA, PHI):
    X = np.abs(R) * np.sin(THETA) * np.cos(PHI)
    Y = np.abs(R) * np.sin(THETA) * np.sin(PHI)
    Z = np.abs(R) * np.cos(THETA)

    return (X, Y, Z)

def cartesian_to_spherical(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z/r)
    phi = np.arctan(y/x)
    return r, theta, phi

def rotation_matrix_x(angle):
    R_x = np.matrix([[1,               0,                0],
                     [0,   np.cos(angle),   -np.sin(angle)],
                     [0,   np.sin(angle),   np.cos(angle)]])
    return R_x

def rotation_matrix_y(angle):
    R_y = np.matrix([[np.cos(angle),   0,    np.sin(angle)],
                     [0,               1,                0],
                     [-np.sin(angle),  0,    np.cos(angle)]])
    return R_y

def rotation_matrix_z(angle):
    R_z = np.matrix([[np.cos(angle), -np.sin(angle),  0],
                     [np.sin(angle),  np.cos(angle),  0],
                     [            0,              0,  1]])
    return R_z

def phi_plane(theta_0, phi_0, n_points=100):
    theta = np.linspace(-np.pi, np.pi, n_points)
    #theta = np.linspace(0, 2*np.pi, n_points)
    phi = [0 for _ in range(len(theta))]
    r = [1 for _ in range(len(theta))]

    X, Y, Z =  spherical_to_cartesian(r, theta, phi)
    points = np.zeros((3,len(X)))
    R_z = rotation_matrix_z(phi_0)
    R_x = rotation_matrix_x(theta_0)
    for i in range(len(X)):
        point = np.array([X[i], Y[i], Z[i]]).T
        R_matrix = R_z.dot(R_x)
        point = R_matrix.dot(point)
        points[:,i] = point
    r, theta, phi = cartesian_to_spherical(points[0,:], points[1,:], points[2,:])
    return r, theta, phi

def theta_plane(theta_0, phi_0, n_points=100):
    theta = np.linspace(-np.pi+theta_0, np.pi+theta_0, n_points)
    #theta = np.linspace(0+theta_0, 2*np.pi+theta_0, n_points)
    phi = [phi_0 for _ in range(len(theta))]
    r = [1 for _ in range(len(theta))]
    return r, theta, phi
