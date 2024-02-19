# %%
### Useful functions for garnet/mineral growth calculations

# %%
import numpy as np

import scipy


# pyrope (py) = MgO
# almandine (alm) = FeO
# spessartine (sp) = MnO
# grossular (gr) = CaO
# khoharite (kho) = MgO


def grid_perplex_data(T, P, data):
    Tmin = T.min()
    Tmax = T.max()
    Tn   = len(np.unique(T))
    Tg   = np.linspace(Tmin, Tmax, Tn)
    
    Pmin = P.min()
    Pmax = P.max()
    Pn   = len(np.unique(P))
    Pg   = np.linspace(Pmin, Pmax, Pn)


    np.nan_to_num(data, copy=False)

    T_grid, P_grid = np.meshgrid(Tg, Pg)


    data_grid = np.reshape(data, [Tn, Pn])

    return data_grid, T_grid, P_grid


### Create a PTt path from the PTt path file and interpolate it to nsteps points
# def create_PTt_path(P, T, t, nsteps):
#     """
#     Perform path interpolation for given PTt path data and number of steps.
#     """

#     Pi     = P
#     Ti     = T
#     tw     = t
#     ntime  = nsteps


#     n_rows = len(Pi)
#     n_rows

#     ### get the number of points to interp the PTt path along
#     d = n_rows/(ntime-1)
#     s = np.arange(0,(n_rows-1)+d, d)

    
#     ### create the interp
#     temp_interp = scipy.interpolate.PchipInterpolator(np.arange(0,n_rows), Ti)
#     pressure_interp = scipy.interpolate.PchipInterpolator(np.arange(0,n_rows), Pi)
#     time_interp = scipy.interpolate.PchipInterpolator(np.arange(0,n_rows), tw)
#     ### do the interp
#     P_interp = pressure_interp(s)
#     T_interp = temp_interp(s)
#     t_interp = time_interp(s)


#     ### Cannot contain identical PT points
#     epsP=.1 
#     epsT=.01
    
#     dPi = P_interp[1:]-P_interp[:-1]
#     dTi = T_interp[1:]-T_interp[:-1]
    
#     #### checks for identical points and adds a small amount to the pressure and temperature
#     P_interp[1:][(dPi < epsP) & (dTi < epsT)] = P_interp[1:][(dPi < epsP) & (dTi < epsT)]+2*epsP
#     T_interp[1:][(dPi < epsP) & (dTi < epsT)] = T_interp[1:][(dPi < epsP) & (dTi < epsT)]+2*epsT

#     #### returns the P and T points along the interpolated path
#     return P_interp, T_interp, t_interp

def generate_garnet_distribution(n_classes, r_min, dr, fnr, Gn, tGn):
    """
    Generate garnet distribution.

    Args:
        n_classes (int): Number of classes.
        r_min (float): Minimum radius.
        dr (float): Delta radius.
        fnr (numpy array): FNR values.
        Gn (numpy array): Gn values.
        tGn (numpy array): tGn values.

    Returns:
        tuple: Tuple containing G, t, r, and R arrays.
    """
        
    v0 = 4/3 * np.pi * r_min**3
    G = np.zeros(n_classes)
    t = np.zeros(n_classes)
    r = np.full(n_classes, r_min, dtype=float)
    R = np.zeros((n_classes, n_classes))

    for i in range(n_classes):
        if i == 0:
            G[i] = v0 * fnr[i]
        else:
            v = 4/3 * np.pi * ((r[:i] + dr)**3 - r[:i]**3)
            V = v0 * fnr[i] + np.sum(v * fnr[:i])
            G[i] = G[i-1] + V
            r[:i] += dr
            r[i] = r_min
            R[:i, i] = r[:i]
            R[i, i] = r[i]

        ind1 = np.where(Gn <= G[i])[0]
        ind2 = np.where(Gn >= G[i])[0]
        i1 = ind1[-1] if ind1.size > 0 else ind2[0]
        i2 = ind2[0] if ind2.size > 0 else ind1[-1]

        t[i] = tGn[i2] if i1 == i2 else np.interp(G[i], [Gn[i1], Gn[i2]], [tGn[i1], tGn[i2]])

    return G, t, r, R


def estimate_PT_path(TPpoints, 
                     values, 
                     garnet_profile):
    
    import numpy as np
    from scipy.optimize import minimize, differential_evolution
    from scipy.interpolate import griddata
    
    ### Example known points and their z values
    # points = np.column_stack([Tgrid.flatten(), Pgrid.flatten()])
    # values = np.column_stack([Fei, Cai, Mgi, Mni]) # Fe, Ca, Mg, Mn
    
    
    points = TPpoints
    results = np.zeros( shape = (len(garnet_profile), 2)  )*np.nan
    
    
    # Define a function for interpolated z values at any given (x, y)
    def interpolated_z(xy, points, values):
        return griddata(points, values, xy, method='linear')
    
    # Define the objective function for minimization
    def objective(xy, points, values, z0):
        L2_norm = 0
        for i in range(len(z0)):
            L2_norm += (interpolated_z(xy, points, values[:,i]) - z0[i])**2
            
        return L2_norm
    
    ## Initial guess for (T, P), centre of the PT grid
    T = np.average(TPpoints[:,0])
    P = np.average(TPpoints[:,1])
    
    initial_guess = [T, P]
    
    # # # z0 is the specific z value you're looking for the (T, P) coordinates of
    for i in range(len(garnet_profile)):
        z0 = garnet_profile[i]
        # result = minimize(objective, initial_guess, method='BFGS'
        #                   bounds=([points[:,0].min(), points[:,0].max()], [points[:,1].min(), points[:,1].max()]), 
        #                   args=(points, values, z0))
        
        result = minimize(fun=objective, x0=initial_guess, method='CG',
                          # bounds=([points[:,0].min(), points[:,0].max()], [points[:,1].min(), points[:,1].max()]), 
                          args=(points, values, z0))
        
        results[i][0] = result.x[0]
        results[i][1] = result.x[1]

    ### returns the estimated PT point
    return results


# %%

# %%

    
    
    

# %%
