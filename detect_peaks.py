from typing import List, Tuple
import numpy as np
from scipy import signal
from scipy.signal import convolve2d
from math import pi
from matplotlib import pyplot as plt

def calc_theta(Z: np.ndarray, filter: bool = False) -> np.ndarray:
    """Computes the theta matrix for a 2-dimensional charge stability diagram.

    The theta matrix indicates the direction of the 2-dimensional gradient.

    Args:
        Z: 2-dimensional charge stability diagram matrix.
        filter: Enables filtering during the calculations.

    Returns:
        theta: 2-dimensional theta matrix.
    """

    ### Filter coefficients
    # filter for S before
    xSfil = np.array([[1], [2], [1]])
    ySfil = np.array([[1, 2, 1]])

    # filter for G after (not used currently)
    xGfil = np.array([[1], [2], [1]])
    yGfil = np.array([[1, 2, 1]])
    Gfil = convolve2d(xGfil, yGfil)

    # filter for Z before (not used currently)
    # xZfil = np.array([[1], [4], [6], [4], [1]])
    # yZfil = np.array([[1, 4, 6, 4, 1]])
    xZfil = np.array([[1], [2], [1]])
    yZfil = np.array([[1, 2, 1]])
    Zfil = convolve2d(xZfil, yZfil)

    # Sobel Operator
    SY = np.array([[1], [0], [-1]])
    SY = convolve2d(SY, ySfil)

    SX = np.array([[1, 0, -1]])
    SX = convolve2d(SX, xSfil)

    if (filter == True):
        Z = convolve2d(Z, Zfil, mode='valid')

    GY = convolve2d(Z, SY, mode='valid')

    GX = convolve2d(Z, SX, mode='valid')

    if (filter == True):
        GY = convolve2d(GY, Gfil, mode='valid')
        GX = convolve2d(GX, Gfil, mode='valid')

    # this isnt even used
    # G = (GX**2 + GY**2)**0.5;

    theta = np.arctan(GY / GX)
    return theta


def calc_transgrad(theta: np.ndarray) -> np.ndarray:
    """Computes the transition gradient matrix from a given theta matrix.

    Args:
        theta: 2-dimensional theta matrix of a charge stability diagram.

    Returns:
        transgrad: 2-dimensional transition gradient matrix.
            x-axis is start position, y-axis is gradient.
    """
    # Low priority: duplicate this function to recalculate transgrad with reduced range.

    # Generate Lines
    ly = theta.shape[0]
    lx = theta.shape[1]

    yl = np.linspace(0, ly, ly, endpoint=False, dtype=int)
    yl = yl.ravel()

    maxdx = np.ceil(ly / 3).astype(int)

    theta_mode = find_mode(theta)

    transgrad = np.zeros((maxdx, lx))

    for x1 in range(lx):
        for dx in range(min([x1 + 1, maxdx])):
            xl = x1 + np.around(-dx * yl / ly).astype(int);
            # Try find the most ideal function. best currently is around(cos^2)
            # transgrad[dx,x1] = np.sum(np.abs(np.sin(theta_mode-theta[yl,xl])))/ly
            # transgrad[dx,x1] = 1-np.mean(np.abs(np.cos(theta_mode-theta[yl,xl])))
            transgrad[dx, x1] = 1 - np.mean(
                np.abs(np.around(np.cos(theta_mode - theta[yl, xl]) ** 2)))

    # these lines filter transgrad, but filtering seems to lose a lot of information
    # filt = np.ones((3,3)) #make this optional
    # transgrad = conv2(transgrad, filt, mode='same')/9
    return transgrad


def find_mode(M: np.ndarray) -> float:
    """Determines the mode of a matrix.

    Ideally used to find the most common theta value.

    Args:
        M: n-dimensional matrix. Ideally a 2-dimensional theta matrix.

    Returns:
        mode: most common element of M.
    """

    H = np.reshape(M, -1)
    hist, hist_edges = np.histogram(H, np.linspace(-pi, pi, 100))
    ind = max_index(hist)
    mode = (hist_edges[ind] + hist_edges[ind + np.array([1])]) * 0.5;
    return mode[0]


def max_index(M: np.ndarray) -> Tuple[int, int]:
    """
    Returns the index of the maximum element in M.

    Args:
        M: n-dimensional matrix.Ideally a 2-dimensional theta matrix,
            2-dimensional charge stability diagram, or 2-dimensional transition
            gradient matrix.

    Returns:
        Array with x and y index of maximum element.
        For a transition gradient matrix, I[0] is dx, and I[1] is x1.
    """
    return np.unravel_index(np.argmax(M, axis=None), M.shape)


def delete_trans(theta: np.ndarray, location: int, gradient: float) -> np.ndarray:
    """Function that attempts to remove a transition from a theta matrix.
    It does so by replacing the transition's theta data with the modal theta.

    Args:
        theta: 2-dimensional theta matrix of a charge stability diagram.
        location: Base index of the charge transfer event in Z
        gradient: Gradient of the charge transfer event in Z

    Returns:
        theta: modified 2-dimensional theta matrix, with the specified transition removed.
    """

    ly = theta.shape[0]
    lx = theta.shape[1]

    yl = np.linspace(0, ly, ly, endpoint=False, dtype=int)
    yl = yl.ravel()

    theta_mode = find_mode(theta)

    # this is naive at the moment
    start = location - 3
    stop = location + 3
    dx = gradient

    if (start < 0):
        start = 0
    if (stop > lx):  # this needs some fix
        stop = lx
    if (start - dx < 0):
        dx = start

    for x1 in range(start, stop):
        xl = x1 + np.around(-dx * yl / ly).astype(int);
        theta[yl, xl] = theta_mode

    return theta


def find_transitions(Z: np.ndarray,
                     x: np.ndarray,
                     y: np.ndarray,
                     trueunits: bool = False,
                     chargetransfer: bool = False,
                     plots: bool = False) -> List[dict]:
    """Function that locates transitions within a 2-dimensional charge
    stability diagram and returns relevant information.

    Args:
        Z: 2-dimensional charge stability diagram matrix.
        x: 1-dimensional voltage vector for the x-axis of Z
        y: 1-dimensional voltage vector for the y-axis of Z
        trueunits:
            if(True):
                Where applicable, return all values in proper units. i.e. voltage and current.
            if(False):
                Return values in calculation specific form. i.e. index and ratios.
        chargetransfer:
            Enables calculation of voltage and current shift information about transitions.
            This is required to calculate dV, dI, dI_x, dI_y
        plots:
            Enables plotting of theta and transition gradient diagrams for each transition found.

    Returns: a list of dictionaries, one entry for each transition found:
    if(trueunits == True):
        location  (float): Voltage at the base of the transition.
        gradient  (float): Gradient of the transition. in y_Voltage/x_Voltage
        intensity (float): Value between 0 and 1, indicating the strength of the transition
        dV        (float): The shift of coulomb peaks from a charge transfer event. dV = dVtop = ∆q/Ctop
        dI        (array): An array of current change from before to after a transition.
                           Returns -1 if error.
        dI_x      (array): An array of x-voltages corresponding to the points in dI.
        dI_y      (array): An array of y-voltages corresponding to the points in dI.
    if(trueunits == False):
        location    (int): Index at the base of the transition.
        gradient  (float): Gradient of the transition. in y-index/x-index
        intensity (float): Value between 0 and 1, indicating the strength of the transition
        dV          (int): The shift of coulomb peaks from a charge transfer event in terms of index in X.
                           dV*(y[1]-y[0]) = dVtop = ∆q/Ctop
        dI        (array): An array of current change from before to after a transition.
                           Returns -1 if error.
        dI_x      (array): An array of x-indices corresponding to the points in dI.
        dI_y      (array): An array of y-indices corresponding to the points in dI.
    """

    theta = calc_theta(Z, filter=True)
    theta_mode = find_mode(theta)
    transgrad = calc_transgrad(theta)
    if (plots):
        plt.figure()
        plt.plot(transgrad, theta, figsize=(14, 5))

    translist = []

    while (np.max(
            transgrad) > 0.4):  # change this value for sensitivity. 0.4 seems to be good

        I = max_index(transgrad)
        M = np.max(transgrad)

        difx = (x.shape[0] - theta.shape[1]) / 2
        dify = (y.shape[0] - theta.shape[0]) / 2
        location = (difx + I[1] + np.around(
            dify * I[0] / theta.shape[0])).astype(int)
        gradient = -(theta.shape[0] / I[0])
        gradient_error = (np.abs(I[0] / (I[0] - 1) - 1) + np.abs(
            I[0] / (I[0] + 1) - 1)) * 50  # same as*100/2
        theta = delete_trans(theta, I[1], I[0])
        transgrad = calc_transgrad(theta)

        if (
        trueunits):  # this makes all the values in actual units, not just indices
            # units in V/V
            gradient = gradient * (y[1] - y[0]) / (x[1] - x[0])  # in V/V
            # units in V
            location = x[location]

        if (chargetransfer):
            # dV = dVtop = ∆q/Ctop
            dV, dI, dI_x, dI_y = get_chargetransfer(Z, location, gradient,
                                                    theta_mode)

            if (
            trueunits):  # this makes all the values in actual units, not just indices
                # units in V
                dV = dV * (y[1] - y[0])
                # units in V
                dI_y = y[dI_y]
                # units in V
                dI_x = x[dI_x]
            trans = {'location': location, 'gradient': gradient, \
                     'grad_err%': gradient_error, 'intensity': M, 'dVtop': dV, \
                     'dI_y': dI_y, 'dI_x': dI_x, 'dI': dI}
        else:
            trans = {'location': location, 'gradient': gradient, \
                     'grad_err%': gradient_error, 'intensity': M}

        translist.append(trans)

        if (plots):
            qc.MatPlot(transgrad, theta, figsize=(14, 5))

    return translist


def get_chargetransfer(Z: np.ndarray,
                       location: int,
                       gradient: float,
                       theta_mode: float) -> Tuple[int, np.ndarray,
                                                   np.ndarray, np.ndarray]:
    """Calculates information about a particular charge transfer event.

    dI information pertains to points on a coulomb peak prior to a charge transfer event.

    Args:
        Z: 2-dimensional charge stability diagram matrix.
        location: Base index of the charge transfer event in Z
        gradient: Gradient of the charge transfer event in Z
        theta_mode: Mode of theta (most common theta value)

    Returns:
        dV: The shift of coulomb peaks from a charge transfer event.
                      Given as a shift of index in Z. When properly scaled:
                          dV = dVtop = ∆q/Ctop
        dI: An array of current change from before to after a transition.
        dI_x: An array of x-indices corresponding to the points in dI.
        dI_y: An array of y-indices corresponding to the points in dI.
    """

    ly = Z.shape[0]
    yl = np.linspace(0, ly, ly, endpoint=False,
                     dtype=int)  # .ravel()  #<<< check out if that works
    yl = yl.ravel()
    xl = (location + np.around(yl / gradient)).astype(int);

    try:
        # lines to check before and after
        shift = 3
        line_pre = Z[yl, xl - shift]
        line_pos = Z[yl, xl + shift]
        # average magnitude difference function
        AMDF = np.zeros(ly)
        for i in range(ly):
            # AMDF[i] = np.mean(np.abs(np.around(np.cos(peak-theta[yl,xl])**2)))
            AMDF[i] = -np.mean(np.abs(
                line_pre[np.array(range(0, ly - i))] - line_pos[
                    np.array(range(i, ly))])) * (ly + i) / ly

        # the 7 in the following line is from the 1 +2*reach of the pos/pre lines
        # qc.MatPlot(AMDF, figsize=(14,5))
        peakshift = np.around(
            np.abs(np.tan(theta_mode - np.pi / 2)) * (1 + 2 * shift)).astype(
            int)
        dV = max_index(AMDF)[0] + peakshift

        shift = 1
        line_pre = Z[yl, xl - shift]
        line_pos = Z[yl, xl + shift]

        # this peak detection could DEFINITELY be fine-tuned (check back for how dI is calculated tooooo)
        # i did it very quick sticks
        # 11pm on a saturday night
        # yes, that quick
        peaks = (
            signal.find_peaks(line_pre - line_pos, distance=25, height=0.2))
        dI_y = peaks[0]
        dI_x = (location + np.around(dI_y / gradient)).astype(int);
        dI = peaks[1]['peak_heights']

        return dV, dI, dI_x, dI_y
    except:
        return -1, -1, -1, -1