import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator, griddata

from UWGarnetDiffusion.functions.mineral_growth_functions import *
from UWGarnetDiffusion.functions.MAGEMin_functions import *

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
    temp_interp = scipy.interpolate.PchipInterpolator(np.arange(0,n_rows), Ti)
    pressure_interp = scipy.interpolate.PchipInterpolator(np.arange(0,n_rows), Pi)
    time_interp = scipy.interpolate.PchipInterpolator(np.arange(0,n_rows), tw)
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


def generate_garnet_from_perpleX(Pi, 
                                 Ti,
                                 ti, 
                                 data_loc,
                                 garnet_r=0, 
                                 garnet_classes=99,
                                 nR_diff=99,
                                 garnet_no=50,
                                 plot_figs=False,):
    """
    Generate a Garnet crystal from a PTt path.

    Args:
        Pi (array): Pressure component along PTt path.
        Ti (array): Temperature component of PTt path.
        ti (array): time component of PTt path.
        data_loc (path): location of Garnet data [Vol (vol%), Fe, Mg, Mn, Ca].
        plot_figs (bool): Plot figures summarising the Garnet generation [optional].
        n_pts (int): Number of PTt points to resample at [optional].
        garnet_r (int): Radius of the selected Garnet crystal [optional].
        garnet_no (int): select a Garnet crystal from the normal distribution [optional]. 
    Returns:
        Rr : Garnet radius
        tr : time of model
        Pr : Pressure at each radius
        Tr : Temperature at each radius
        Mnr: Mn concentration at each radius
        Mgr: Mg concentration at each radius
        Fer: Fe concentration at each radius
        Car: Ca concentration at each radius

    """ 
    # if PTt_path == None:
    #     print("PTt_path must be specified")
    #     exit()
    
    if type(data_loc) == None:
        RuntimeError("Please supply path to location of Garnet data")
    
    if garnet_no >= 100:
        RuntimeError("garnet_no must be < 100 (0 is the largest garnet)")


    ### read in garnet 2D griddata data from perplex'''
    gt_vol = pd.read_csv(f'{data_loc}/vol.tab', sep='\s+', skiprows=12)

    gt_Fe = pd.read_csv(f'{data_loc}/Fe.tab', sep='\s+', skiprows=12)
    gt_Mg = pd.read_csv(f'{data_loc}/Mg.tab', sep='\s+', skiprows=12)
    gt_Mn = pd.read_csv(f'{data_loc}/Mn.tab', sep='\s+', skiprows=12)
    gt_Ca = pd.read_csv(f'{data_loc}/Ca.tab', sep='\s+', skiprows=12)


    #### grid the Perplex data
    V_grid, T_grid, P_grid = grid_perplex_data(gt_vol.iloc[:,0], gt_vol.iloc[:,1], gt_vol.iloc[:,2])
    Fe_grid, T_grid, P_grid = grid_perplex_data(gt_Fe.iloc[:,0], gt_Fe.iloc[:,1], gt_Fe.iloc[:,2])
    Mn_grid, T_grid, P_grid = grid_perplex_data(gt_Mn.iloc[:,0], gt_Mn.iloc[:,1], gt_Mn.iloc[:,2])
    Mg_grid, T_grid, P_grid = grid_perplex_data(gt_Mg.iloc[:,0], gt_Mn.iloc[:,1], gt_Mg.iloc[:,2])
    Ca_grid, T_grid, P_grid = grid_perplex_data(gt_Ca.iloc[:,0], gt_Ca.iloc[:,1], gt_Ca.iloc[:,2])


    ### numpy array with the P and T points to interpolate the data to
    points = np.vstack([Ti, Pi]).T

    # Interpolate using griddata
    GVi = np.nan_to_num(griddata((T_grid.flatten(), P_grid.flatten()), V_grid.flatten(), points, method='linear'))

    Fei = np.nan_to_num(griddata((T_grid.flatten(), P_grid.flatten()), Fe_grid.flatten(), points, method='linear'))


    Mni = np.nan_to_num(griddata((T_grid.flatten(), P_grid.flatten()), Mn_grid.flatten(), points, method='linear'))


    Mgi = np.nan_to_num(griddata((T_grid.flatten(), P_grid.flatten()), Mg_grid.flatten(), points, method='linear'))


    Cai = np.nan_to_num(griddata((T_grid.flatten(), P_grid.flatten()), Ca_grid.flatten(), points, method='linear'))


    # The first GVi must be 0, we will call it reduction (we neglect the previous history - we recommend starting from a zero field of garnet)
    GVi = GVi - GVi[0]  # reduction (previous history neglected)

    Rr, tr, Pr, Tr, Mnr, Mgr, Fer, Car  = generate_garnets(GVi,  Mgi, Mni, Fei, Cai, ti, Ti, Pi, garnet_classes, garnet_no, garnet_r, nR_diff, plot_figs)



    return Rr, tr, Pr, Tr, Mnr, Mgr, Fer, Car 


def generate_garnet_from_MAGEMin(Pi, 
                                 Ti,
                                 ti,
                                 data,
                                 X,
                                 Xoxides,
                                 sys_in,
                                 garnet_r=0, 
                                 garnet_classes=99,
                                 nR_diff=99,
                                 garnet_no=50,
                                 plot_figs=False,
                                 fractionate=False,
                                 verbose=0,):
    """
    Generate garnet from MAGEMin data.

    Parameters:
        Pi (float): Pressure in kbar
        Ti (float): Temperature in K
        ti (float): Time in Myr
        db (str): MAGEMin database object
        X (array): Bulk rock composition
        Xoxides (array): Oxide composition of bulk rock
        garnet_r (int, optional): Garnet radius. Defaults to 0.
        garnet_classes (int, optional): Number of garnet classes. Defaults to 99.
        nR_diff (int, optional): Number of shells to create in the Garnet. Defaults to 99.
        garnet_no (int, optional): Number of garnets. Defaults to 50.
        plot_figs (bool, optional): Whether to plot figures. Defaults to False.
        fractionate (bool, optional): Whether to fractionate the bulk rock as garnet forms. Defaults to False.
        verbose (int, optional): Verbosity level. Defaults to False.

    Returns:
        tuple: Rr, tr, Pr, Tr, Mnr, Mgr, Fer, Car
    """
        
    

    # gt_mol_frac, py_arr, alm_arr, spss_arr, gr_arr, kho_arr = garnet_over_path(Pi, Ti, data, X, Xoxides, sys_in, verbose=0, fractionate=0)

    GVi, Mgi, Mni, Fei, Cai = garnet_over_path(Pi, Ti, data, X, Xoxides, sys_in, verbose=0, fractionate=fractionate)

     # # The first GVi must be 0, we will call it reduction (we neglect the previous history - we recommend starting from a zero field of garnet)
    ### This is now done in the fractionation path


    # Rr, tr, Pr, Tr, Mnr, Mgr, Fer, Car  = generate_garnets(GVi,  py_arr, spss_arr, alm_arr, gr_arr, ti, Ti, Pi, garnet_classes, garnet_no, garnet_r, nR_diff, plot_figs)
    Rr, tr, Pr, Tr, Mnr, Mgr, Fer, Car  = generate_garnets(GVi,  Mgi, Mni, Fei, Cai, ti, Ti, Pi, garnet_classes, garnet_no, garnet_r, nR_diff, plot_figs)

    
    



    


    


    return Rr, tr, Pr, Tr, Mnr, Mgr, Fer, Car 



def generate_garnets(GVi, Mgi, Mni, Fei, Cai, ti, Ti, Pi, garnet_classes, garnet_no, garnet_r, nR_diff, plot_figs):

    nPT = len(GVi)


    # find the indices where the garnet does not grow
    k = 0
    GVG = [GVi[0]]  # GVG must be 0 at the beginning
    ind_no_increase = []

    for i in range(1, nPT):
        if GVi[i] <= max(GVG):
            k += 1
            ind_no_increase.append(i)
            GVG.append(max(GVG))
        else:
            GVG.append(GVi[i])


    ### Normalise values to max volume of garnet
    GVG = np.array(GVG) / max(GVG) #### normalise to 1

    ### find where GVG first is equal to 1
    indGVG1 = np.where(GVG == 1)[0]

    ### end of growth stage
    iend = min(indGVG1)

    ### indices of the growth stage
    ind = list(range(iend))

    ### Find where the garnet is growing (GVG < 1)
    ind_increase = np.where(GVG < 1)[0] 

    ### select data from the growth stage
    tG = ti[ind]
    TG = Ti[ind]
    PG = Pi[ind]
    MnG = np.array(Mni)[ind]
    MgG = np.array(Mgi)[ind]
    FeG = np.array(Fei)[ind]
    CaG = np.array(Cai)[ind]
    GVG = np.array(GVG)[ind]


    size_dist = 'N'  # type of garnet size distribution
    n_classes = garnet_classes  # number of garnet classes

    if garnet_no >= 100:
        print("garnet_no must be < 100 (1 is the largest garnet)")
        exit()

    # radius classes
    r_min = 10  # minimum (initial) garnet radius (~microns)
    r_max = 500  # maximum radius
    r = np.linspace(r_min, r_max, n_classes)
    dr = r[1] - r[0]
    nr = n_classes

    if garnet_r == 0:
        r_resc = r[n_classes - garnet_no]
    else:
        r_resc = garnet_r


    # prepare size_distribution
    if size_dist == 'N':  # normal distribution (cut off)
        mi = (r_min + r_max) / 2
        s = (mi - r_min) / 2
        finp = np.exp(-(r - mi)**2 / 2 / s**2) / np.sqrt(2 * np.pi) / s
        # show_dilute(mi, s, 200) # This function is not defined in the provided code
    elif size_dist == 'U':  # user defined
        finp = np.ones(n_classes)  # /n_classes



    # normalization
    v = 4/3 * np.pi * r**3
    V = np.sum(v * finp)  # dr not necessary
    fn = finp / V
    check_fn = np.sum(v * fn)
    fnr = fn[::-1]  # the largest garnet goes first

    # check GVG
    if len(GVG) <= 3:
        print("GVG is too short - end of make_garnets")
        exit()

    # monotonicity of GVG
    difGVG = np.diff(GVG)
    for i, val in enumerate(difGVG):
        if val < 0:
            print(i, val, "diff(GVG)<0")
            exit()

    # normalization
    Gn = GVG / np.max(GVG)
    tGn = tG 
    
    
    G, t, r, R = generate_garnet_distribution(n_classes, r_min, dr, fnr, Gn, tGn)


    # PT and concentrations along equidistant r
    PGrw = np.interp(t, tG, PG)
    TGrw = np.interp(t, tG, TG)
    Mnrw = np.interp(t, tG, MnG)
    Mgrw = np.interp(t, tG, MgG)
    Ferw = np.interp(t, tG, FeG)
    # Carw = np.interp(t, tG, CaG)
    Carw = 1 - Mnrw - Mgrw - Ferw

    # Preparation of selected garnet for make_diffusion
    ind = np.arange(garnet_no, nr)  

    Rr1 = R[garnet_no, ind]  # numpy uses 0-based indexing

    # Extract data corresponding to the growth of the selected garnet (garnet_no)
    tr1 = np.array(t)[ind] 
    Pr1 = PGrw[ind]
    Tr1 = TGrw[ind]
    Mnr1 = Mnrw[ind]
    Mgr1 = Mgrw[ind]
    Fer1 = Ferw[ind]
    # Car1 = Carw[ind]
    Car1 = 1 - Mnr1 - Mgr1 - Fer1

    # Final profiles of the selected garnet - nR_diff shells
    dRr = Rr1[-1] / nR_diff
    Rrz = np.arange(dRr, Rr1[-1] + dRr, dRr)

    ### interp values from old radius to new one
    trz = np.interp(Rrz, Rr1, tr1)
    Prz = np.interp(Rrz, Rr1, Pr1)
    Trz = np.interp(Rrz, Rr1, Tr1) 
    Mnrz = np.interp(Rrz, Rr1, Mnr1) 
    Mgrz = np.interp(Rrz, Rr1, Mgr1) 
    Ferz = np.interp(Rrz, Rr1, Fer1)
    Carz = 1 - Mnrz - Mgrz - Ferz

    # # Add 1st point - the centre
    Rr = np.concatenate([[0], Rrz])
    tr = np.concatenate([[trz[0]], trz])
    Pr = np.concatenate([[Prz[0]], Prz])
    Tr = np.concatenate([[Trz[0]], Trz])
    Mnr = np.concatenate([[Mnrz[0]], Mnrz])
    Mgr = np.concatenate([[Mgrz[0]], Mgrz])
    Fer = np.concatenate([[Ferz[0]], Ferz])
    Car = np.concatenate([[Carz[0]], Carz])

    ### scale the values to the scaled 
    Rr = Rr / np.max(Rr) * r_resc 


    # Extract data corresponding to the growth of the selected garnet (garnet_no)
    tr1 = np.array(t)[ind] 
    Pr1 = PGrw[ind]
    Tr1 = TGrw[ind]
    Mnr1 = Mnrw[ind]
    Mgr1 = Mgrw[ind]
    Fer1 = Ferw[ind]
    # Car1 = Carw[ind]
    Car1 = 1 - Mnr1 - Mgr1 - Fer1

    # Final profiles of the selected garnet - nR_diff shells
    dRr = Rr1[-1] / nR_diff
    Rrz = np.arange(dRr, Rr1[-1] + dRr, dRr)


    ### interp values from old radius to new one
    trz = np.interp(Rrz, Rr1, tr1)
    Prz = np.interp(Rrz, Rr1, Pr1)
    Trz = np.interp(Rrz, Rr1, Tr1) 
    Mnrz = np.interp(Rrz, Rr1, Mnr1) 
    Mgrz = np.interp(Rrz, Rr1, Mgr1) 
    Ferz = np.interp(Rrz, Rr1, Fer1)
    Carz = 1 - Mnrz - Mgrz - Ferz

    # # Add 1st point - the centre
    Rr = np.concatenate([[0], Rrz])
    tr = np.concatenate([[trz[0]], trz])
    Pr = np.concatenate([[Prz[0]], Prz])
    Tr = np.concatenate([[Trz[0]], Trz])
    Mnr = np.concatenate([[Mnrz[0]], Mnrz])
    Mgr = np.concatenate([[Mgrz[0]], Mgrz])
    Fer = np.concatenate([[Ferz[0]], Ferz])
    Car = np.concatenate([[Carz[0]], Carz])

    ### scale the values to the scaled 
    Rr = Rr / np.max(Rr) * r_resc 


    if plot_figs == True:
        fig, axs = plt.subplots(3, 2, figsize=(10, 15))  # Create a figure with 3x2 subplots
        fig.suptitle('Garnet formation summary')

        # Subplot 1
        axs[0, 0].set_title('Garnet size')
        axs[0, 0].set_xlabel('r')
        axs[0, 0].set_ylabel('f')
        axs[0, 0].set_xlim([0, r.max()])
        axs[0, 0].plot(r, finp, '-')
        for i in range(n_classes):
            axs[0, 0].plot([r[i], r[i]], [0, finp[i]], '-')

        # Subplot 2
        axs[0, 1].set_title('classes birth place')
        axs[0, 1].set_xlabel('T')
        axs[0, 1].set_ylabel('P')
        axs[0, 1].plot(Ti, Pi)
        axs[0, 1].plot(TGrw, PGrw, 'r.')  # where the garnet classes were formed
        # selected garnet
        axs[0, 1].plot(TGrw[garnet_no], PGrw[garnet_no], 'ko')
        axs[0, 1].plot(TGrw[garnet_no], PGrw[garnet_no], 'kx')
        axs[0, 1].text(TGrw[garnet_no]*1.02, PGrw[garnet_no], str(garnet_no))

        # Subplot 3
        axs[1, 0].set_title('classes growth')
        axs[1, 0].set_xlabel('t')
        axs[1, 0].set_ylabel('r')

        for i in range(0, nr, 10):
            axs[1, 0].plot(t[i:], R[i, i:], 'r.-')

        if garnet_no <= nr:
            axs[1, 0].plot(t[garnet_no:], R[garnet_no, garnet_no:], 'k.:', linewidth=1)
            axs[1, 0].text(t[-1]*1.02, R[garnet_no, -1], str(garnet_no))
        else:
            print('wrong garnet_no')

        # Subplot 4
        axs[1, 1].set_title('volume consumption')
        axs[1, 1].set_xlabel('t')
        axs[1, 1].set_ylabel('GV')
        axs[1, 1].set_ylim([0, 1.01])
        axs[1, 1].plot(tGn, Gn, 'k')
        axs[1, 1].plot(t, G, 'mx')  # formation of classes
        axs[1, 1].plot(t[garnet_no], G[garnet_no], 'ko')  # selected garnet
        axs[1, 1].plot(t[garnet_no], G[garnet_no], 'kx')  # selected garnet
        axs[1, 1].grid(True)

        # Subplot 5
        axs[2, 0].set_title('Mn(b) Mg(r) Fe(m) Ca(g)')
        axs[2, 0].set_xlabel('r')
        axs[2, 0].set_ylabel('c')
        axs[2, 0].set_xlim([0, max(r)])
        for i in np.arange(1, nr, 10):  # 1:nr
            ind = np.arange(i, nr)
            r = R[i, :]
            axs[2, 0].plot(r[ind], Mnrw[ind], '-b', r[ind], Mgrw[ind], 'r-', r[ind], Ferw[ind], 'm-', r[ind], Carw[ind], 'g-')

        i = garnet_no
        ind = np.arange(i, nr)
        r = R[i, :]
        axs[2, 0].plot(r[ind], Mnrw[ind], 'b:', r[ind], Mgrw[ind], 'r:', r[ind], Ferw[ind], 'm:', r[ind], Carw[ind], 'g:', linewidth=2)
        rshow = r
        rshowmax = max(rshow)


        # Subplot 6
        axs[2, 1].set_title(f'Garnet No {garnet_no}')
        axs[2, 1].set_xlabel('r')
        axs[2, 1].set_ylabel('c')
        axs[2, 1].set_xlim([0, max(Rr)])
        axs[2, 1].set_ylim([0, 1])
        axs[2, 1].plot(Rr, Mnr, '-b', Rr, Mgr, 'r-', Rr, Fer, 'm-', Rr, Car, 'g-', linewidth=2)

        plt.show()


        return Rr, tr, Pr, Tr, Mnr, Mgr, Fer, Car 



















