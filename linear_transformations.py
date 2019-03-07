# linear_transformations.py
"""
Linear Transformations
Jake Callahan
2018/09/18

Linear transformations are the most basic and essential operators in vector space
theory. In this program we visually explore how linear transformations alter points
in the Cartesian plane. We also empirically explore the computational cost of
applying linear transformations via matrix multiplication.
"""
from matplotlib import pyplot as plt
from random import random
import numpy as np
import time

def stretch(A, a, b):
    """Scale the points in A by a in the x direction and b in the
    y direction.

    Parameters:
        A ((2,n) ndarray): Array containing points in R2 stored as columns.
        a (float): scaling factor in the x direction.
        b (float): scaling factor in the y direction.
    """
    #Create transformation matrix, multiply by A, return
    B = np.array([[a, 0], [0, b]])
    A = B @ A
    return A

def shear(A, a, b):
    """Slant the points in A by a in the x direction and b in the
    y direction.

    Parameters:
        A ((2,n) ndarray): Array containing points in R2 stored as columns.
        a (float): scaling factor in the x direction.
        b (float): scaling factor in the y direction.
    """
    #Create transformation matrix, multiply by A, return
    B = np.array([[1, a], [b, 1]])
    A = B @ A
    return A

def reflect(A, a, b):
    """Reflect the points in A about the line that passes through the origin
    and the point (a,b).

    Parameters:
        A ((2,n) ndarray): Array containing points in R2 stored as columns.
        a (float): x-coordinate of a point on the reflecting line.
        b (float): y-coordinate of the same point on the reflecting line.
    """
    #Create transformation matrix, multiply by A, return
    B = (1 / (a**2 + b**2))*np.array([[a**2 - b**2, 2*a*b], [2*a*b, b**2 - a**2]])
    A = B @ A
    return A

def rotate(A, theta):
    """Rotate the points in A about the origin by theta radians.

    Parameters:
        A ((2,n) ndarray): Array containing points in R2 stored as columns.
        theta (float): The rotation angle in radians.
    """
    #Create transformation matrix, multiply by A, return
    B = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    A = B @ A
    return A

def test_transforms():
    #Create stretch, shear, reflect, and rotated horses
    A = np.load("horse.npy")
    stretch_matrix = stretch(A, 1/2, 6/5)
    shear_matrix = shear(A, 1/2, 0)
    reflect_matrix = reflect(A, 0, 1)
    rotate_matrix = rotate(A, np.pi/2)

    #Plot original horse
    ax1 = plt.subplot(2,3,1)
    ax1.plot(A[0], A[1], 'k,')
    plt.axis([-1,1,-1,1])

    #Plot stretched horse
    ax2 = plt.subplot(2,3,2)
    ax2.plot(stretch_matrix[0], stretch_matrix[1], 'k,')
    plt.axis([-1,1,-1,1])

    #Plot sheared horse
    ax3 = plt.subplot(2,3,3)
    ax3.plot(shear_matrix[0], shear_matrix[1], 'k,')
    plt.axis([-1,1,-1,1])

    #Plot reflected horse
    ax4 = plt.subplot(2,3,4)
    ax4.plot(reflect_matrix[0], reflect_matrix[1], 'k,')
    plt.axis([-1,1,-1,1])

    #Plot rotated horse
    ax5 = plt.subplot(2,3,5)
    ax5.plot(rotate_matrix[0], rotate_matrix[1], 'k,')
    plt.axis([-1,1,-1,1])

    plt.show()

def solar_system(T, x_e, x_m, omega_e, omega_m):
    """Plot the trajectories of the earth and moon over the time interval [0,T]
    assuming the initial position of the earth is (x_e,0) and the initial
    position of the moon is (x_m,0).

    Parameters:
        T (int): The final time.
        x_e (float): The earth's initial x coordinate.
        x_m (float): The moon's initial x coordinate.
        omega_e (float): The earth's angular velocity.
        omega_m (float): The moon's angular velocity.
    """
    #set initial position coordinates
    p_e = np.array([x_e, 0])
    p_m = np.array([x_m, 0])
    p_m -= p_e
    earth_positions = []
    moon_positions = []
    #set domain
    t = np.linspace(0, T, 1000)
    #rotate position vectors
    for i in t:
        earth_coord = rotate(p_e, i*omega_e)
        earth_positions.append(earth_coord)
        moon_coord = rotate(p_m, i*omega_m) + earth_positions[-1]
        moon_positions.append(moon_coord)
    #get position arrays in usable form
    earth_positions = np.transpose(earth_positions)
    moon_positions = np.transpose(moon_positions)
    #plot positions
    plt.plot(earth_positions[0], earth_positions[1], linewidth = 3, label = "Earth")
    plt.plot(moon_positions[0], moon_positions[1], linewidth = 3, label = "Moon")
    plt.legend(loc = "lower right")
    plt.axis("equal")
    plt.show()

def random_vector(n):
    """Generate a random vector of length n as a list."""
    return [random() for i in range(n)]

def random_matrix(n):
    """Generate a random nxn matrix as a list of lists."""
    return [[random() for j in range(n)] for i in range(n)]

def matrix_vector_product(A, x):
    """Compute the matrix-vector product Ax as a list."""
    m, n = len(A), len(x)
    return [sum([A[i][k] * x[k] for k in range(n)]) for i in range(m)]

def matrix_matrix_product(A, B):
    """Compute the matrix-matrix product AB as a list of lists."""
    m, n, p = len(A), len(B), len(B[0])
    return [[sum([A[i][k] * B[k][j] for k in range(n)])
                                    for j in range(p) ]
                                    for i in range(m) ]

def time_multiplication():
    """Use time.time() to time matrix_vector_product() and matrix-matrix-mult()
    with increasingly large inputs. Generate the inputs A, x, and B with
    random_matrix() and random_vector() (so each input will be nxn or nx1).

    Plot the results in a single figure with two subplots: one with matrix-
    vector times, and one with matrix-matrix times.
    """
    #Set initial lists to store time values
    matrix_times = []
    vector_times = []
    #Set domain
    domain = 2**np.arange(1, 10)
    for n in domain:
        #Generate matrices and vectors to use
        A = random_matrix(n)
        B = random_matrix(n)
        x = random_vector(n)
        #Time matrix_matrix multiplication
        matrix_start = time.time()
        matrix_matrix_product(A, B)
        matrix_times.append(time.time() - matrix_start)

        #Time matrix_vector multiplication
        vector_start = time.time()
        matrix_vector_product(A, x)
        vector_times.append(time.time() - vector_start)

    #Plot matrix_matrix multiplication
    ax1 = plt.subplot(121)
    ax1.set_title("Matrix-Matrix Multiplication")
    ax1.set_xlabel("n")
    ax1.set_ylabel("Seconds")
    ax1.plot(domain, matrix_times, 'g.-', linewidth = 2, markersize = 15)

    #Plot matrix-vector multiplication
    ax2 = plt.subplot(122)
    ax2.set_title("Matrix-Vector Multiplication")
    ax2.set_xlabel("n")
    ax2.set_ylabel("Seconds")
    ax2.plot(domain, vector_times, 'g.-', linewidth = 2, markersize = 15)

    plt.show()

def time_products():
    """Time matrix_vector_product(), matrix_matrix_product(), and np.dot().

    Plot results in a single figure with two subplots: one with all
    four sets of execution times on a regular linear scale, and one with all
    four sets of exections times on a log-log scale.
    """
    #Set initial lists to store time values
    my_matrix_times = []
    my_vector_times = []
    np_matrix_times = []
    np_vector_times = []
    #Set domain to work on
    domain = 2**np.arange(1, 5)

    for n in domain:
        # Generate matrices and vector for calculation
        A = random_matrix(n)
        B = random_matrix(n)
        x = random_vector(n)

        # Time naive matrix-matrix multiplication
        matrix_start = time.time()
        matrix_matrix_product(A, B)
        my_matrix_times.append(time.time() - matrix_start)

        # Time naive matrix-vector multiplication
        vector_start = time.time()
        matrix_vector_product(A, x)
        my_vector_times.append(time.time() - vector_start)

        # Time numpy matrix-matrix multiplication
        matrix_start = time.time()
        np.dot(A, B)
        np_matrix_times.append(time.time() - matrix_start)

        # Time numpy matrix-vector multiplication
        vector_start = time.time()
        np.dot(A, x)
        np_vector_times.append(time.time() - vector_start)

    #Set first axis to plot on
    ax1 = plt.subplot(121)
    ax1.set_title("Times on Regular Scale")
    ax1.set_xlabel("n")
    ax1.set_ylabel("Seconds")

    #Plot lists against domain
    ax1.plot(domain, my_matrix_times,'g.-', linewidth = 2, markersize = 7, label = "My matrix_matrix product")
    ax1.plot(domain, my_vector_times,'b.-', linewidth = 2, markersize = 7, label = "My matrix_vector product")
    ax1.plot(domain, np_matrix_times,'r.-', linewidth = 2, markersize = 7, label = "Numpy matrix_matrix product")
    ax1.plot(domain, np_vector_times,'y.-', linewidth = 2, markersize = 7, label = "Numpy matrix_vector product")
    ax1.legend(loc = "upper right")

    #Set second axis to plot on
    ax2 = plt.subplot(122)
    ax2.set_title("Times on Log-Log Scale")
    ax2.set_xlabel("n")
    ax2.set_ylabel("Seconds")

    #Plot lists against domain in log-log
    ax2.loglog(domain, my_matrix_times,'g.-', linewidth = 2, markersize = 7, label = "My matrix_vector product", basey=2, basex=2)
    ax2.loglog(domain, my_vector_times,'b.-', linewidth = 2, markersize = 7, label = "My matrix_vector product", basey=2, basex=2)
    ax2.loglog(domain, np_matrix_times,'r.-', linewidth = 2, markersize = 7, label = "Numpy matrix_matrix product", basey=2, basex=2)
    ax2.loglog(domain, np_vector_times,'y.-', linewidth = 2, markersize = 7, label = "Numpy matrix_vector product", basey=2, basex=2)
    ax2.legend(loc = "upper right")

    plt.show()
