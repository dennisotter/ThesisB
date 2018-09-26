from typing import List, Tuple
import numpy as np
from scipy import signal
from scipy.signal import convolve2d
from math import pi
from matplotlib import pyplot as plt


def max_index(M: np.ndarray) -> Tuple[int, int]:
    """Returns the index of the maximum element in M.

    Args:
        M: n-dimensional matrix. Ideally a 2-dimensional theta matrix,
            2-dimensional charge stability diagram, or 2-dimensional transition
            gradient matrix.

    Returns:
        Array with x and y index of maximum element.
        For a transition gradient matrix, I[0] is dx, and I[1] is x1.
    """
    return np.unravel_index(np.argmax(M), M.shape)


def calculate_theta_matrix(Z: np.ndarray, filter: bool = False) -> np.ndarray:
    """Computes the theta matrix for a 2-dimensional charge stability diagram.

    The theta matrix indicates the direction of the 2-dimensional gradient.

    # TODO Please elaborate code. At this moment all these arrays don't make
    #     sense. Either say why the values are chosen, or refer to a
    #     website/paper where they were obtained from.
    # TODO Could modifying these values improve the code?
    # TODO From what I gather, we're applying a kernel, so perhaps it makes sense
    #     to have kernel_size as a keyword argument
    # TODO what does filter do excactly? What type of filtering is applied?
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

    # TODO If something is not used, either remove it or have it as a keyword option
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

    if filter:
        Z = convolve2d(Z, Zfil, mode='valid')

    GY = convolve2d(Z, SY, mode='valid')

    GX = convolve2d(Z, SX, mode='valid')

    if filter:
        GY = convolve2d(GY, Gfil, mode='valid')
        GX = convolve2d(GX, Gfil, mode='valid')

    # TODO what does this do? Why isn't it used?
    # this isnt even used
    # G = (GX**2 + GY**2)**0.5;

    theta = np.arctan(GY / GX)
    return theta


def find_matrix_mode(M: np.ndarray) -> float:
    """Determines the mode of a matrix (most-often occurring element).

    # TODO check if this description is accurate
    Mode is found by first generating a histogram of matrix values, and then
    returning the center value of the bin with the highest count.
    Values are grouped because floating numbers are only approximately equal.

    Args:
        M: n-dimensional matrix. Ideally a 2-dimensional theta matrix.

    Returns:
        mode: most common element of M after grouping via a histogram.
    """
    H = np.reshape(M, -1)
    # TODO have bins (100) as a keyword argument
    hist, hist_edges = np.histogram(H, np.linspace(-pi, pi, 100))
    ind = max_index(hist)
    mode = (hist_edges[ind] + hist_edges[ind + np.array([1])]) / 2
    return mode[0]


def calculate_transition_gradient(theta: np.ndarray) -> np.ndarray:
    """Compute the transition gradient matrix from a given theta matrix.

    # TODO minor explanation of what a transition gradient is

    Args:
        theta: 2-dimensional theta matrix of a charge stability diagram.

    Returns:
        transgrad: 2-dimensional transition gradient matrix.
            x-axis is start position, y-axis is gradient.
    """
    # Low priority: duplicate this function to recalculate transgrad with reduced range.

    # Generate Lines
    ly, lx = theta.shape

    yl = np.arange(ly, dtype=int)

    # TODO where does this value come from?
    dx_max = int(np.ceil(ly / 3))

    theta_mode = find_matrix_mode(theta)

    transition_gradient = np.zeros((dx_max, lx))

    # TODO (Serwan) there's probably a loopless way to implement this
    for x1 in range(lx):
        for dx in range(min([x1 + 1, dx_max])):
            xl = x1 + np.round(-dx * yl / ly).astype(int)
            # Try find the most ideal function. best currently is round(cos^2)
            # transition_gradient[dx,x1] = np.sum(np.abs(np.sin(theta_mode-theta[yl,xl])))/ly
            # transition_gradient[dx,x1] = 1-np.mean(np.abs(np.cos(theta_mode-theta[yl,xl])))
            transition_gradient[dx, x1] = 1 - np.mean(
                np.abs(np.round(np.cos(theta_mode - theta[yl, xl]) ** 2)))

    # TODO make filtering optional
    # these lines filter transgrad, but filtering seems to lose a lot of information
    # filt = np.ones((3,3))
    # transgrad = convolve2d(transgrad, filt, mode='same')/9
    return transition_gradient


def delete_transition(theta: np.ndarray,
                      location: int,
                      gradient: float) -> np.ndarray:
    """Remove a transition from a theta matrix.

    It does so by replacing the transition's theta data with the modal theta.
    # TODO a bit more elaborate explanation of the algorithm

    Args:
        theta: 2-dimensional theta matrix of a charge stability diagram.
        location: Base index of the charge transfer event in Z
        gradient: Gradient of the charge transfer event in Z

    Returns:
        theta: modified 2-dimensional theta matrix, with the specified transition removed.
    """

    ly = theta.shape[0]
    lx = theta.shape[1]

    yl = np.arange(ly, dtype=int)

    theta_mode = find_matrix_mode(theta)

    # this is naive at the moment
    # TODO improve start, stop, why is +-3 chosen?
    start = location - 3
    stop = location + 3
    dx = gradient

    if start < 0:
        start = 0
    if stop > lx:  # this needs some fix
        stop = lx
    if start - dx < 0:
        dx = start

    # TODO (Serwan) there's probably a faster loop-less way to do this
    for x1 in range(start, stop):
        xl = x1 + np.round(-dx * yl / ly).astype(int)
        theta[yl, xl] = theta_mode

    return theta


# TODO (Serwan) Implement this function
def plot_transitions(Z, x, y, transitions, ax=None,
                     transition_gradient=None,):
    if ax is None:
        fig, ax = plt.subplots()


def find_transitions(Z: np.ndarray,
                     x: np.ndarray,
                     y: np.ndarray,
                     min_gradient: float = 0.4,
                     true_units: bool = False,
                     charge_transfer: bool = False,
                     plot: bool = False) -> List[dict]:
    """Locate transitions within a 2-dimensional charge stability diagram

    Args:
        Z: 2-dimensional charge stability diagram matrix.
        x: 1-dimensional voltage vector for the x-axis of Z
        y: 1-dimensional voltage vector for the y-axis of Z
        min_gradient: Minimum gradient to count as a transition
        true_units:
            if True:
                Where applicable, return all values in proper units. i.e. voltage and current.
            if False:
                Return values in calculation specific form. i.e. index and ratios.
        charge_transfer:
            Enables calculation of voltage and current shift information about transitions.
            This is required to calculate dV, dI, dI_x, dI_y
        plot:
            Enables plotting of theta and transition gradient diagrams for each transition found.

    Returns: a list of dictionaries, one entry for each transition found:
    # TODO (Serwan) simplify this part
    if true_units == True:
        location  (float): Voltage at the base of the transition.
        gradient  (float): Gradient of the transition. in y_Voltage/x_Voltage
        intensity (float): Value between 0 and 1, indicating the strength of the transition
        dV        (float): The shift of coulomb peaks from a charge transfer event. dV = dVtop = ∆q/Ctop
        dI        (array): An array of current change from before to after a transition.
                           Returns -1 if error.
        dI_x      (array): An array of x-voltages corresponding to the points in dI.
        dI_y      (array): An array of y-voltages corresponding to the points in dI.
    if true_units == False):
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

    theta = calculate_theta_matrix(Z, filter=True)
    theta_mode = find_matrix_mode(theta)
    transition_gradient = calculate_transition_gradient(theta)

    if plot:
        # TODO add plot of transition in DC scan
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].pcolormesh(transition_gradient)
        axes[1].pcolormesh(theta)

    transitions = []

    # change this value for sensitivity. 0.4 seems to be good
    while np.max(transition_gradient) > min_gradient:
        I = max_index(transition_gradient)
        M = np.max(transition_gradient)

        # TODO this chunk of code is hard to understand, should add documentation
        # between most lines explaining each step
        difx = (x.shape[0] - theta.shape[1]) / 2
        dify = (y.shape[0] - theta.shape[0]) / 2
        location = int(difx + I[1] + np.round(dify * I[0] / theta.shape[0]))
        gradient = -(theta.shape[0] / I[0])
        gradient_error = (np.abs(I[0] / (I[0] - 1) - 1) + np.abs( # TODO Why -1 and +1?
            I[0] / (I[0] + 1) - 1)) * 50  # same as*100/2  # TODO Where does 50 come from?
        theta = delete_transition(theta, I[1], I[0])
        transition_gradient = calculate_transition_gradient(theta)

        if true_units:  # Convert indices to units
            gradient = gradient * (y[1] - y[0]) / (x[1] - x[0])  # in V/V
            location = x[location]  # units in V

        if charge_transfer:  # TODO What is the downside to applying this by default?
            # dV = dVtop = delta_q/Ctop
            dV, dI, dI_x, dI_y = get_charge_transfer_information(
                Z, location, gradient, theta_mode)

            if true_units:  # this makes all the values in actual units, not just indices
                # units in V
                dV = dV * (y[1] - y[0])
                # units in V
                dI_y = y[dI_y]
                # units in V
                dI_x = x[dI_x]
            transition = {'location': location,
                          'gradient': gradient,
                          'gradient_error': gradient_error,
                          'intensity': M,
                          'dVtop': dV,
                          'dI_y': dI_y,
                          'dI_x': dI_x,
                          'dI': dI}
        else:
            transition = {'location': location,
                          'gradient': gradient,
                          'gradient_error': gradient_error,
                          'intensity': M}

        transitions.append(transition)

        if plot:
            # TODO add x, y units to plot
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            axes[0].pcolormesh(transition_gradient)
            axes[1].pcolormesh(theta)

    return transitions


def get_charge_transfer_information(Z: np.ndarray,
                                    location: int,
                                    gradient: float,
                                    theta_mode: float) -> Tuple[int, np.ndarray,
                                                   np.ndarray, np.ndarray]:
    """Calculate information about a particular charge transfer event.

    dI information pertains to points on a coulomb peak prior to a charge transfer event.
    # TODO elaborate how the algorithm works

    Args:
        Z: 2-dimensional charge stability diagram matrix.
        location: Base index of the charge transfer event in Z
        gradient: Gradient of the charge transfer event in Z
        theta_mode: Mode of theta (most common theta value)

    Returns:
        dV: The shift of coulomb peaks from a charge transfer event.
                      Given as a shift of index in Z. When properly scaled:
                          dV = dVtop = delta_q/Ctop
        dI: An array of current change from before to after a transition.
        dI_x: An array of x-indices corresponding to the points in dI.
        dI_y: An array of y-indices corresponding to the points in dI.
    """

    ly = Z.shape[0]
    yl = np.arange(ly, dtype=int)
    xl = (location + np.round(yl / gradient)).astype(int)

    # TODO most of this code is quite hard to follow, please add some documentation
    try:  # TODO is the try except necessary?
        # lines to check before and after
        shift = 3
        line_pre = Z[yl, xl - shift]
        line_pos = Z[yl, xl + shift]
        # average magnitude difference function
        AMDF = np.zeros(ly)
        for i in range(ly):
            # AMDF[i] = np.mean(np.abs(np.round(np.cos(peak-theta[yl,xl])**2)))
            AMDF[i] = -np.mean(np.abs(
                line_pre[np.array(range(0, ly - i))] - line_pos[
                    np.array(range(i, ly))])) * (ly + i) / ly

        # the 7 in the following line is from the 1 +2*reach of the pos/pre lines
        # qc.MatPlot(AMDF, figsize=(14,5))
        peakshift = np.round(
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
        peaks = (signal.find_peaks(line_pre - line_pos, distance=25, height=0.2))
        dI_y = peaks[0]
        dI_x = (location + np.round(dI_y / gradient)).astype(int)
        dI = peaks[1]['peak_heights']

        return dV, dI, dI_x, dI_y
    except:
        return -1, -1, -1, -1
