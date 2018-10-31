### Automated Tracking of Single Atom Qubits for Silicon Quantum Computing ###
# By Dennis Otter
# Honours Thesis 2018.
#
#
#
# If you would like an in-depth explaination of this code, please refer to my thesis paper.
# The paper also includes notes on some vital improvements to this code.

from typing import List, Tuple
import numpy as np
from scipy import signal
from scipy.signal import convolve2d
from math import pi
from matplotlib import pyplot as plt
from qcodes import load_data, MatPlot
import itertools as it

def max_index(M: np.ndarray) -> Tuple[int, int]:
    """Returns the index of the maximum element in M.

    Args:
        M: n-dimensional matrix. Ideally a 2-dimensional theta matrix,
            2-dimensional charge stability diagram, or 2-dimensional
            Hough transform matrix.

    Returns:
        Array with x and y index of maximum element.
        For a transition gradient matrix, I[0] is dx, and I[1] is x1.
    """
    return np.unravel_index(np.argmax(M), M.shape)


def calculate_theta_matrix(Z: np.ndarray, filter: bool = False) -> np.ndarray:
    """Computes the theta matrix for a 2-dimensional charge stability diagram.

    The theta matrix indicates the direction of the 2-dimensional gradient.

    # TODO Fine-tuning these values could definitely improve performance.
    # TODO We are applying a kernel, so perhaps it makes sense to have kernel_size as a keyword argument.
    
    Args:
        Z: 2-dimensional charge stability diagram matrix.
        filter: Enables filtering during the calculations.

    Returns:
        theta: 2-dimensional theta matrix.
    """

    ### Filter coefficients

    # Refer to https://en.wikipedia.org/wiki/Sobel_operator
    # That explains the prinicples used here.

    # Sobel Operator
    # SY and SX are differentiating kernels, while the ySfil and xSfil are averaging.
    SY = np.array([[1], [0], [-1]])
    ySfil = np.array([[1, 2, 1]])
    SY = convolve2d(SY, ySfil)

    SX = np.array([[1, 0, -1]])
    xSfil = np.array([[1], [2], [1]])
    SX = convolve2d(SX, xSfil)

    # Binomial filter kernel for the X and Y gradient matrices.
    xGfil = np.array([[1], [2], [1]])
    yGfil = np.array([[1, 2, 1]])
    Gfil = convolve2d(xGfil, yGfil)

    # Binomial filter kernel for the source matrix Z prior to computing the gradients.
    xZfil = np.array([[1], [2], [1]])
    yZfil = np.array([[1, 2, 1]])
    Zfil = convolve2d(xZfil, yZfil)

    if filter:
        # This will filter the source matrix Z prior to computing the gradients.
        Z = convolve2d(Z, Zfil, mode='valid')

    #Calculate X and Y gradient matrices
    GY = convolve2d(Z, SY, mode='valid')
    GX = convolve2d(Z, SX, mode='valid')

    if filter:
        #This will filter the gradient matrices once they have been calculated.
        GY = convolve2d(GY, Gfil, mode='valid')
        GX = convolve2d(GX, Gfil, mode='valid')

    #Calculate gradient direction.
    theta = np.arctan(GY / GX)
    return theta


def find_matrix_mode(M: np.ndarray, bins: int = 100) -> float:
    """Determines the mode of a matrix (most-often occurring element).

    Mode is found by first generating a histogram of matrix values, and then
    returning the center value of the bin with the highest count.
    Values are grouped because floating numbers are only approximately equal.

    Args:
        M: n-dimensional matrix. Ideally a 2-dimensional theta matrix.

    Returns:
        mode: most common element of M after grouping via a histogram.
    """

    hist, hist_edges = np.histogram(M, np.linspace(-pi, pi, bins))
    ind = max_index(hist)
    mode = (hist_edges[ind] + hist_edges[ind + np.array([1])]) / 2
    return mode[0]


def calculate_hough_transform(theta_dif: np.ndarray) -> np.ndarray:
    """Compute the Hough transform matrix from a given theta matrix.
    
    # TODO Have a variable input for the gradient range to scan for. This is dx_max

    Args:
        theta_dif: Filtered 2-dimensional theta matrix of a charge stability diagram.

    Returns:
        2-dimensional Hough transform matrix.
            x-axis is start position, y-axis is gradient.
    """
   
    # Generate Lines
    len_y, len_x = theta_dif.shape
    y_line = np.arange(len_y, dtype=int)

    # This value was found to be roughly twice the maximum dx value a transition will have
    # Essentially this is the minimum gradient
    dx_max = int(np.ceil(len_y / 3))

    hough = np.zeros((dx_max, len_x))

    # TODO there's probably a loopless way to implement this
    # -> I don't suspect there is.
    for x1 in range(len_x):
        for dx in range(min([x1 + 1, dx_max])):
            x_line = x1 + np.round(-dx * y_line / len_y).astype(int)
            hough[dx, x1] = np.mean(theta_dif[y_line, x_line])

    return hough


def delete_transition(theta_dif: np.ndarray, hough_filt: np.ndarray, hough_raw: np.ndarray) -> np.ndarray:
    """Removes a transition from a theta_dif matrix. In order to find transitions, they are identified one at a time. 
    The most prominent transition is identified first, then removed from the theta_dif matrix so that the second most 
    prominent transition can be found.
    
    Args:
        theta_dif: 2-dimensional filtered theta matrix of a charge stability diagram.
        location: Base index of the charge transfer event in Z
        gradient: Gradient of the charge transfer event in Z

    Returns:
        theta_dif: modified filtered 2-dimensional theta matrix, with the specified transition removed.
    """

    dx, location = max_index(hough_filt) 
    len_y,len_x = theta_dif.shape
    y_line = np.arange(len_y, dtype=int)
    dx_max = hough_filt.shape[0]

    # Start and stop are the base locations from which to delete a transition from.
    start = location
    stop = location
    while((hough_filt[dx,start] >= 0.25)&(start>0)): #this value 0.25 needs to be same as in find_transition
        start -=1
    while((hough_filt[dx,stop] >= 0.25)&(stop<(len_x-1))):
        stop +=1

    # TODO there's probably a faster loop-less way to do this
    for x1 in range(start, stop):
        x_line = x1 + np.round(-dx * y_line / len_y).astype(int)
        theta_dif[y_line, x_line] = 0
        
    x1 = np.array(range(start-dx,min([stop+dx_max-dx, len_x])))
    for i in range(x1.size):
        for dx_scan in range(min([x1[i] + 1, dx_max])):
            x_line = x1[i] + np.round(-dx_scan * y_line / len_y).astype(int)
            hough_raw[dx_scan, x1[i]] = np.mean(theta_dif[y_line, x_line])
            
    

    return theta_dif, hough_raw


def calculate_theta_dif(theta: np.ndarray, theta_mode: float) -> np.ndarray:
    """Calculates the difference between the values of a given 
    theta matrix, and theta_mode. This function is used 
    to highlight pixels that are likely to lie on a transition.
    
    Args:
        theta: 2-dimensional theta matrix of a charge stability diagram.
        theta_mode: Modal theta value, found with find_matrix_mode().

    Returns:
        theta_dif: modified 2-dimensional theta matrix, 
            with the specified transition removed.
    """
    #you can change this method for potential improvements
    theta_dif = 1 - np.cos(theta_mode - theta) ** 2
    return theta_dif


def plot_transitions(x: np.ndarray, y: np.ndarray, Z: np.ndarray, transitions: List[dict]):
    """Plots the source charge stability diagram, next to a charge 
    stability diagram with the given transitions plotted.
    
    Args:
        x: 1-dimensional voltage vector for the x-axis of Z
        y: 1-dimensional voltage vector for the y-axis of Z
        Z: 2-dimensional charge stability diagram matrix.
        transitions: List of transitions found using find_transitions()
    """
    fig0,(ax0,ax1) = plt.subplots(1, 2, figsize=[12,4])
    fig0.suptitle('Transition Identification', fontsize=16, fontweight='semibold')

    ax0.pcolormesh(x, y, Z, cmap='hot')
    ax0.set_xlabel('Fast Gate Voltage (V)', fontsize=14)
    ax0.set_ylabel('TG Voltage (V)', fontsize=14)
    ax0.set_title('Source scan', fontsize=16)

    ax1.pcolormesh(x, y, Z, cmap='hot')
    ax1.set_xlabel('Fast Gate Voltage (V)', fontsize=14)
    ax1.set_title('Transitions Identified', fontsize=16)

    yvals = ax1.get_ylim()
    for transition in transitions:
        x_base = transition['location']
        if (type(x_base) is int) : x_base = x[x_base]

        xvals = [x_base, x_base]
        xvals[1] += (yvals[1] - yvals[0]) / transition['gradient']
        ax1.plot(xvals, yvals, '-', linewidth=4)
    plt.show()


def plot_hough_transform(hough: np.ndarray, theta_dif: np.ndarray, location: int = -1, dx: int = -1):
    """Plots a filtered theta matrix next to its Hough Transform.
    
    Args:
        hough: 2D hough transform matrix
        theta_dif: 2D filtered theta matrix.
    """
    fig, axes = plt.subplots(1, 2, figsize=[13,4])

    c = axes[1].pcolormesh(hough, cmap='inferno')
    axes[1].set_ylabel('∆x value', fontsize=14)
    axes[1].set_xlabel('Fast Gate Voltage index', fontsize=14)
    axes[1].set_title('Hough Transform Matrix', fontsize=16)
    fig.colorbar(c, ax=axes[0])

    axes[0].pcolormesh(theta_dif, cmap='gray')
    axes[0].set_xlabel('Fast Gate voltage index', fontsize=14)
    axes[0].set_ylabel('TG voltage index', fontsize=14)
    axes[0].set_title('Filtered Theta Matrix', fontsize=16)
    
    if location is not -1:
        axes[1].scatter(location, dx, marker= 'o', linewidth=4, color='magenta')
        yvals = axes[0].get_ylim()
        xvals = [location, location-dx]
        axes[0].plot(xvals, yvals, '--', linewidth=2,color='magenta')

    plt.show()


def find_transitions(x: np.ndarray,
                     y: np.ndarray,
                     Z: np.ndarray,
                     true_units: bool = True,
                     plot = False) -> List[dict]:
    """Locate transitions within a 2-dimensional charge stability diagram

    Args:
        x: 1-dimensional voltage vector for the x-axis of Z
        y: 1-dimensional voltage vector for the y-axis of Z
        Z: 2-dimensional charge stability diagram matrix.
        true_units:
            if True:
                Where applicable, return all values in proper units. i.e. voltage and current.
            if False:
                Return values in calculation specific form. i.e. index and ratios.
        charge_transfer:
            Enables calculation of voltage and current shift information about transitions.
            This is required to calculate dV, dI, dI_x, dI_y
        plot:
             - 'Off'     = No plots
             - 'Simple'  = Plot of DC data and transition data next to it
             - 'Complex' = All of simple, plus the transition_gradient and theta plots for each transition.

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
    theta_dif = calculate_theta_dif(theta,theta_mode)

    hough_raw = calculate_hough_transform(theta_dif)
    H_filter = np.ones((3,3))
    hough_filt = convolve2d(hough_raw, H_filter, mode='same')/9

    transitions = []

    # This condition seems good now, could be improved later but i'd say low priority.
    while ((np.max(hough_filt) > 3*np.mean(hough_filt)) & (np.max(hough_filt) >0.25)):
        
        #maximum element of transition_gradient will reveal where the transition is
        raw_gradient, raw_location = max_index(hough_filt) 
        intensity = np.max(hough_filt) 
        
        if (plot == 'Complex'): plot_hough_transform(hough_filt, theta_dif, raw_location, raw_gradient)

        # When filtering with convolution, the size of theta and transition_gradient will differ from the initial Z matrix.
        # The following lines adjust the raw_location from transition_gradient to be a true location in Z
        difx = (x.shape[0] - theta.shape[1]) / 2 #difference in x-axis size
        dify = (y.shape[0] - theta.shape[0]) / 2 #difference in y-axis size
        #Adjusting the location: 
        # raw_location 
        # + difference in x
        # + difference in x from dify due to gradient shift.
        location = int(difx + raw_location + np.round(dify * raw_gradient / theta.shape[0]))
        
        #Recalculate theta and hough with the identified transition removed
        theta_dif, hough_raw = delete_transition(theta_dif, hough_filt, hough_raw)
        hough_filt = convolve2d(hough_raw, H_filter, mode='same')/9
        
        #If the gradient registers as being close to perfectly vertical, skip over this transition, 
        #since transitions are never perfectly vertical. 
        #You can change this if you don't believe me but the algorithm will be more buggy
        if(raw_gradient <2): continue
        
        #gradient = dy/dx = y_length/dx = theta.shape[0]/raw_gradient
        gradient = -(theta.shape[0]/raw_gradient)
        
        # The following lines combine together to come up with the final error calculation
        #minimum_gradient = dy/max_dx = -(theta.shape[0]/(raw_gradient+1))
        #maximum_gradient = dy/min_dx = -(theta.shape[0]/(raw_gradient-1))
        #abs_error        = (maximum_gradient - minimum_gradient)/2
        #percent_error    = abs_error/observed_gradient*100                = abs_error/(-(theta.shape[0]/raw_gradient))*100
        gradient_error    = (np.abs(raw_gradient/(raw_gradient-1)-1) + np.abs(raw_gradient/(raw_gradient+1)-1))*50
        
        dV = get_charge_transfer(Z, location, gradient, theta_mode)
        if true_units:  # Convert indices to units
            gradient = gradient * (y[1] - y[0]) / (x[1] - x[0])  # in V/V
            location = x[location]  # units in V
            dV = dV * (y[1] - y[0]) # units in V

        #Add transition entry onto the output list
        transitions.append({'location': location,
                      'gradient': gradient,
                      'gradient_error': gradient_error,
                      'intensity': intensity,
                      'dVtop': dV})

    if (plot == 'Complex'): plot_hough_transform(hough_filt,theta_dif)
        
    if (plot == True)|(plot == 'Complex'): plot_transitions(x,y,Z,transitions)

    return transitions


def get_charge_transfer(Z: np.ndarray,
                                    location: int,
                                    gradient: float,
                                    theta_mode: float) -> int:
    """Calculates how much the coulomb peaks shift at a charge transfer event.
    It does this by taking two slices either side of the transition, then comparing the shift.

    Args:
        Z         : 2-dimensional charge stability diagram matrix.
        location  : Base index of the charge transfer event in Z
        gradient  : Gradient of the charge transfer event in Z
        theta_mode: Mode of theta (most common theta value)

    Returns:
        dV: The shift of coulomb peaks from a charge transfer event.
                      Given as a shift of index in Z. When properly scaled:
                          dV = dVtop = delta_q/Ctop
    """
    ly = Z.shape[0]
    yl = np.arange(ly, dtype=int)
    xl = (location + np.round(yl / gradient)).astype(int)


    # Take two current lines to the left and right of the transition, these will be compared to see what changes.
    # 3 can be chosen arbitrarily, it doesn't matter too much, 
    # as long as each line is definitely on each side of the transition
    shift = 4
    pre  = xl - shift
    post = xl + shift
    if ((min(pre) < 0)|(max(post)>=Z.shape[1])):
        #if the pre-post lines are out of bounds, then don't bother computing. This could be improved later.
        return -1
    line_pre = Z[yl, pre]
    line_pos = Z[yl, post]
    
    # Average Magnitude Difference Function. 
    # This will shift and compare the lines before and after to see how much the transition shifted the coulomb peaks
    AMDF = np.zeros(ly)
    for i in range(ly):
        
        AMDF[i] = -np.mean(np.abs(
            line_pre[np.array(range(0, ly - i))]
            -line_pos[np.array(range(i, ly))]))  \
            * (ly + i) / ly                      #Adjustment for the decreasing comparison window as lines are shifted

    # peakshift exists to find out how much of the shift in coulomb peaks in the difference is due to the 
    # natural gradient of the coulomb peaks. This can be worked out using tan, the coulomb peak gradient (theta_mode), 
    # and the shift amount
    peakshift = np.round(
        np.abs(np.tan(theta_mode - np.pi/2)) *(1 +2*shift)).astype(
        int)
    dV = max_index(AMDF)[0] + peakshift

########## This code is rudimentary but it is on the way to automatically finding tuning points #######
    
#     #*** This following section of code could be improved.
#     #*** It is a rudimentary implimentation of finding potential tuning points. 
    
#     #Now we will take closer lines to compare, in order to find the biggest difference in SET current.
#     shift = 1
#     line_pre = Z[yl, xl - shift]
#     line_pos = Z[yl, xl + shift]

#     #Compare the lines and find peaks in the difference 
#     #the find_peaks parameters could really be improved.
#     #also, using a %difference could be much better than absolute. Use (line_pre-line_pos)/line_pos when re-evaluating.
#     peaks = (signal.find_peaks(line_pre - line_pos, distance=25, height=0.2))
#     dI_y = peaks[0] #y-index of the peaks location
#     dI_x = (location + np.round(dI_y / gradient)).astype(int) #x-index of the peaks
#     dI = peaks[1]['peak_heights'] #the value of the peaks

    return dV.astype(int)#, dI, dI_x, dI_y

## Start code for 3D tracking

def find_transitions_3D(slow: np.ndarray,
                        fast: np.ndarray,
                        TG: np.ndarray,
                        Z: np.ndarray,
                        plot: bool = False) -> List[List[dict]]:
    """Locate transitions within a 3-dimensional charge stability diagram

    Args:
        slow: 1-dimensional voltage vector for the slow gate voltage of Z
        fast: 1-dimensional voltage vector for the fast gate voltage of Z
        TG  : 1-dimensional voltage vector for the TG voltage of Z
        Z   : 3-dimensional charge stability diagram matrix. [slow,fast,TG]
        plot:
             - False = No plots
             - True  = Plots the 3D transitions.

    Returns: A list of transition lists over the third dimension found with find_transitions()
    """
    transition_list = []
    for i in range(Z.shape[0]): 
        transition_list.append(find_transitions(fast,TG, Z[i,:,:], true_units=True))
    
    if plot:
        plot_transitions_3D(slow,fast,TG,Z,transition_list)
    return transition_list


def plot_transitions_3D(slow: np.ndarray, 
                     fast: np.ndarray, 
                     TG: np.ndarray, 
                     Z: np.ndarray, 
                     transition_list: List[List[dict]],
                     fit_list: List[dict] = None):
    """Plots the transitions found over a 3D scan with find_transitions_3D().
    
    Args:
        slow: 1-dimensional voltage vector for the slow gate voltage of Z
        fast: 1-dimensional voltage vector for the fast gate voltage of Z
        TG  : 1-dimensional voltage vector for the TG voltage of Z
        Z   : 3-dimensional charge stability diagram matrix. [slow,fast,TG]
        transition_list: A list of transition lists. Calculated with find_transitions_3D()
        fit_list: A list of fit lines, linking the transitions in transition_list. 
            fit_list is calculated with track_transitions_multi or track_transitions_single.
    """
    n_slow = Z.shape[0]
    x_points = []
    y_points = []
    grad_points = []
    for i in range(n_slow):
        for T in transition_list[i]:
            fast_val = T['location']
            x_points.append(slow[i])
            y_points.append(fast_val)
            grad_points.append(T['gradient'])
    
    
    fig,ax = plt.subplots(1, 1, figsize=[9,5])
    plot = ax.scatter(x_points, y_points, c=grad_points)
    c = fig.colorbar(plot, ax=ax)
    c.set_label('TG/Fast Gradient', fontsize=14)
    
    xvals = np.array([slow[0], slow[-1]])
    if fit_list is not None:
        for F in fit_list:
            yvals = xvals*F['fast/slow gradient'] +F['fast intercept']
            ax.plot(xvals,yvals)#,color='red')
    
    xmargin = (np.max(x_points)-np.min(x_points))/13
    xlim = [(np.min(x_points)-xmargin), (np.max(x_points)+xmargin)]
    ymargin = (np.max(y_points)-np.min(y_points))/10
    ylim = [(np.min(y_points)-ymargin), (np.max(y_points)+ymargin)]
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_ylabel('Fast Gate Voltage (V)', fontsize=14)
    ax.set_xlabel('Slow Gate Voltage (V)', fontsize=14);
    ax.set_title('3D Transition Plot', fontsize=14, fontweight='semibold')


def track_transitions_single(slow: np.ndarray,
                     fast: np.ndarray,
                     TG: np.ndarray, 
                     Z: np.ndarray,
                     transition_list: List[List[dict]],
                     plot: bool = False) -> List[dict]:
    """Tracks the transitions found with find_transitions_2D(). 
    
    There must be a single transition over the entire 3D scan for this to work. 
    Any more or less will cause the algorithm to work incorrectly.
    This algorithm is very reliable when used appropriately.
    
    Args:
        slow: 1-dimensional voltage vector for the slow gate voltage of Z
        fast: 1-dimensional voltage vector for the fast gate voltage of Z
        TG  : 1-dimensional voltage vector for the TG voltage of Z
        Z   : 3-dimensional charge stability diagram matrix. [slow,fast,TG]
        transition_list: A list of transition lists. Calculated with find_transitions_2D()
        plot: If plot==True, then plot_transitions_2D() is automatically called with the respective values.
    """
    
    n_slow = Z.shape[0]
    x_points = []
    y_points = []
    grad_points = []
    for i in range(n_slow): #length of slow gate
        for T in transition_list[i]:
            x_points.append(i)
            y_points.append(T['location'])
            grad_points.append(T['gradient'])
    
    
    X = np.swapaxes(np.array((slow[x_points], np.ones((len(x_points)),dtype=int))),0,1)
    Y = np.asarray(y_points)
    
    r = np.linalg.lstsq(X,Y)[0]
    m_slow = r[0]
    b = r[1]
    m_fast = np.mean(grad_points)

    retval = []
    retval.append({'fast intercept': b,
              'fast/slow gradient': m_slow,
              'TG/fast gradient': m_fast,
              'TG/slow gradient': m_fast/m_slow})#,
              
#               'Slow points': X[:,0],
#               'Fast points': Y,
#               'Fit points': X@r,
#               'Gradient vector': grad_points})
    
    if plot:
        plot_transitions_3D(slow,fast,TG,Z,transition_list,retval)
    return retval


#The following code is unreliable. Do not use. Only included to demonstrate how it works for some cases
def track_transitions_multi(slow: np.ndarray,
                     fast: np.ndarray,
                     TG: np.ndarray, 
                     Z: np.ndarray,
                     transition_list: List[List[dict]],
                     plot: bool = False) -> List[dict]:
    """Tracks the transitions found in find_transitions_3D(). 
    
    This will only work for any amount of transitions over the entire 3D scan
    However, this algorithm is not accurate; it can link transitions which are not correlated 
    and it can miss transitions that are.
    
    *** WARNING: DO NOT USE THIS ALGORITHM FOR LARGE DATASETS ***
    This algorithm is incredibly inefficient. 
    It calculates every possible combination of transitions and computes the least squares fit line, finding the best fit.
    This relates to approx (n_transitions)^2*(n_scans)!
    
    I have tested this for sets up to 6 scans (6 charge stability diagrams)
    A test of 100 scans was sufficient to use over 16GB RAM and 100% of a 4-core processor.
    
    Basically, don't use this function. I have included it only to demonstrate my attempt. 
    If you desire to correlate multiple transitions, use find_transitions_3D() with plot=True, 
    then visually decide which ones are correlated.
    
    Args:
        slow: 1-dimensional voltage vector for the slow gate voltage of Z
        fast: 1-dimensional voltage vector for the fast gate voltage of Z
        TG  : 1-dimensional voltage vector for the TG voltage of Z
        Z   : 3-dimensional charge stability diagram matrix. [slow,fast,TG]
        transition_list: A list of transition lists. Calculated with find_transitions_3D()
        plot: If plot==True, then plot_transitions_3D() is automatically called with the respective values.
    """
    
    n_slow = Z.shape[0]
    points_list = []
    for i in range(n_slow): #length of slow gate
        for T in transition_list[i]:
            points_list.append((i,
                                T['location'],
                                T['gradient']))
    
    X_slow_0 = np.swapaxes(np.array((slow, np.ones((len(slow)),dtype=int))),0,1)
#     np.ones((n_slow,2),dtype=int)
#     X_slow_0[:,0] = np.arange(n_slow,dtype=int)
    least_squares = []

    combs = [] #combinations of points
    for i in range((int)(n_slow*2/3), n_slow+1):
        combs.extend(list(it.combinations(range(n_slow),i))) #creates a list of possibilities
        
    for X_slow in combs:
        slow_index = []
        for k in X_slow:
            slow_index.append(np.arange(len(transition_list[k])))
        slow_index = np.asarray(list(it.product(*slow_index)))

        for k in range(slow_index.shape[0]):

            Y_fast = [transition_list[X_slow[j]][slow_index[k,j]]['location'] for j in range(len(X_slow))]
            G_fast = [transition_list[X_slow[j]][slow_index[k,j]]['gradient'] for j in range(len(X_slow))]

            X_slow_1 = X_slow_0[X_slow,:]

            r = np.linalg.lstsq(X_slow_1,Y_fast)[0]
            m = r[0]
            b = r[1]
            Y_fit = X_slow_1@r
            if 0 in Y_fit:
                continue
            Y_dif = np.mean(np.abs(Y_fast/Y_fit-1)) #np.mean(np.abs(Y_fit-Y_fast))
            G_dif = np.max(np.abs((G_fast/np.mean(G_fast)-1)))

            Tot_dif = Y_dif/len(Y_fast)*(1+G_dif) #change this for different sorting

            lsq = {'m': m,
                   'b': b,
                   'Y_fast': Y_fast,
                   'X_slow': X_slow,
                   'Y_fit': Y_fit,
    #                'Y_dif': Y_dif,
                   'G_fast': G_fast,
    #                'G_dif': G_dif,
                   'Tot_dif': Tot_dif,}
            least_squares.append(lsq)
        
    least_squares = sorted(least_squares, key=lambda ls: ls['Tot_dif'])
    retval = []
    numbrs = []
    

    for L in least_squares:
        if (L['Tot_dif'] >0.005): break
        ok = True
        for Y in L['Y_fast']:
            if Y in numbrs:
                ok = False
                break
        if(ok):
            retval.append({'fast intercept': L['b'],
                      'fast/slow gradient': L['m'],
                      'TG/fast gradient': np.mean(L['G_fast']),
                      'TG/slow gradient': np.mean(L['G_fast'])/L['m']})#,
              
#                       'Slow points': slow[np.asarray(L['X_slow'])],
#                       'Fast points': L['Y_fast'],
#                       'Fit points': L['Y_fit'],
#                       'Gradient vector': L['G_fast']})
            numbrs.extend(L['Y_fast'])
            
    
    if plot:
        plot_transitions_2D(slow,fast,TG,Z,transition_list, retval)
            
    return retval

