import numpy as np
from scipy.interpolate import PchipInterpolator, griddata


def create_PTt_path(P, T, t, nsteps):
    """
    Create a PTt path array by interpolating the given PTt data.
    Parameters:
        PTt_data (ndarray): An array containing PTt data with shape (n, 3), where n is the number of data points. Each row should contain pressure, temperature, and time values.
        n_pts (int): The number of points to be generated in the PTt path. Defaults to 101.
    Returns:
        ndarray: A 2D array representing the PTt path. Each row contains the interpolated pressure, temperature, and time values at a specific point along the path.
    """
    Pi     = P
    Ti     = T
    tw     = t


    n_rows = len(Pi)
    n_rows

    ### get the number of points to interp the PTt path along
    d = n_rows/(nsteps-1)
    s = np.arange(0,(n_rows-1)+d, d)

    
    ### create the interp
    temp_interp = PchipInterpolator(np.arange(0,n_rows), Ti)
    pressure_interp = PchipInterpolator(np.arange(0,n_rows), Pi)
    time_interp = PchipInterpolator(np.arange(0,n_rows), tw)
    ### do the interp
    P_interp = pressure_interp(s)
    T_interp = temp_interp(s)
    t_interp = time_interp(s)


    ### Cannot contain identical PT points
    epsP=.1 
    epsT=.01
    
    dPi = P_interp[1:]-P_interp[:-1]
    dTi = T_interp[1:]-T_interp[:-1]
    
    #### checks for identical points and adds a small amount to the pressure and temperature
    P_interp[1:][(dPi < epsP) & (dTi < epsT)] = P_interp[1:][(dPi < epsP) & (dTi < epsT)]+2*epsP
    T_interp[1:][(dPi < epsP) & (dTi < epsT)] = T_interp[1:][(dPi < epsP) & (dTi < epsT)]+2*epsT

    # Create PTt path array
    PTt_path = np.zeros(shape=(len(P_interp), nsteps))
    PTt_path[:, 0] = P_interp
    PTt_path[:, 1] = T_interp
    PTt_path[:, 2] = t_interp

    return PTt_path