import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def generate_distribution(n_classes, r_min, dr, fnr, Gn, tGn):
    """
    Generates a distribution of radial sizes and associated garnet volumes and formation times.
    
    Parameters:
        n_classes (int): Number of classes.
        r_min (float): Minimum mineral/crystal radius.
        dr (float): Increment by which radii are increased.
        fnr (array-like): Array of normalized fractions for new volume added per class.
        Gn (array-like): Total volume at discrete time steps.
        tGn (array-like): Formation times corresponding to volumes in Gn.
    
    Returns:
        cumulative_volumes (np.array): Cumulative mineral volume for each class.
        formation_times (np.array): Formation time for each class.
        radii (np.array): Final radius values for each class.
        radius_matrix (2D np.array): Matrix of radius values between classes.
    """
    # Calculate the initial volume for a garnet of minimum radius.
    initial_volume = 4/3 * np.pi * r_min**3

    # Initialize arrays for cumulative volumes, formation times, and radii.
    cumulative_volumes = np.zeros(n_classes)
    formation_times = np.zeros(n_classes)
    # Every new class starts with r_min.
    radii = np.full(n_classes, r_min, dtype=float)
    # Create an empty matrix for radii between classes.
    radius_matrix = np.full((n_classes, n_classes), np.nan)
    
    # Loop over each garnet class.
    for i in range(n_classes):
        if i == 0:
            # For the first class, scale the initial volume using the fraction.
            current_volume = initial_volume * fnr[i]
            cumulative_volumes[i] = current_volume
            # Save the volume increment for reference.
            vol_increment = initial_volume
        else:
            # For subsequent classes, compute the volume increment for each existing class.
            # This computes the change in volume when the radius increases by dr.
            volume_increments = 4/3 * np.pi * ((radii[:i] + dr)**3 - (radii[:i])**3)
            
            current_volume = initial_volume * fnr[i] + np.sum(volume_increments * fnr[:i])

            # Update cumulative volume: each class adds its computed volume.
            cumulative_volumes[i] = cumulative_volumes[i-1] + current_volume
        
        # Update the radii for the already processed classes by adding dr,
        # then reset the current class radius to the minimum.
        radii[:i] += dr
        radii[i] = r_min

        # Build the radius matrix: for column i, store the radii for previous classes.
        radius_matrix[:i, i] = radii[:i]
        radius_matrix[i, i] = radii[i]
        
        # Determine the formation time for this class by finding which 
        # time index in tGn corresponds to the cumulative volume.
        lower_inds = np.where(Gn <= cumulative_volumes[i])[0]
        upper_inds = np.where(Gn >= cumulative_volumes[i])[0]
        i_lower = lower_inds[-1] if lower_inds.size > 0 else upper_inds[0]
        i_upper = upper_inds[0] if upper_inds.size > 0 else lower_inds[-1]
        
        # If the cumulative volume exactly matches one value, use that formation time;
        # otherwise, interpolate between the two nearest time points.
        if i_lower == i_upper:
            formation_times[i] = tGn[i_upper]
        else:
            formation_times[i] = np.interp(cumulative_volumes[i],
                                           [Gn[i_lower], Gn[i_upper]],
                                           [tGn[i_lower], tGn[i_upper]])
    
    return cumulative_volumes, formation_times, radii, radius_matrix

class GarnetGenerator:
    def __init__(self, Pi, Ti, ti, data, X, Xoxides, sys_in,
                 r_min=10, r_max=100, garnet_classes=99, nR_diff=99, fractionate=False):
        self.Pi = Pi
        self.Ti = Ti
        self.ti = ti
        self.data = data
        self.X = X
        self.Xoxides = Xoxides
        self.sys_in = sys_in
        self.r_min = r_min
        self.r_max = r_max
        self.garnet_classes = garnet_classes
        self.nR_diff = nR_diff
        self.fractionate = fractionate

        self.extract_garnet_data()

    def extract_garnet_data(self):
        from .MAGEMin_functions import MAGEMinGarnetCalculator
        garnet_generator = MAGEMinGarnetCalculator()

        (self.gt_mol_frac, self.gt_wt_frac, self.gt_vol_frac,
         self.d_gt_mol_frac, self.d_gt_wt_frac,
         self.Mgi, self.Mni, self.Fei, self.Cai) = garnet_generator.gt_over_path(
            self.Pi, self.Ti, self.data, self.X, self.Xoxides,
            self.sys_in, fractionate=self.fractionate
        )

    def _reset_variables(self):
        ### reset everything to None
        (self.gt_mol_frac, self.gt_wt_frac, self.gt_vol_frac,
         self.d_gt_mol_frac, self.d_gt_wt_frac,
         self.Mgi, self.Mni, self.Fei, self.Cai) = [None]*9
    
    def _compute_normalized_GVG(self, GVi):
        """Compute and return normalized garnet volume sequence (GVG)"""
        GVG = [GVi[0]]
        for i in range(1, len(GVi)):
            if GVi[i] <= max(GVG):
                GVG.append(max(GVG))
            else:
                GVG.append(GVi[i])
        return np.array(GVG) / np.max(GVG)
    
    def _get_size_distribution(self, size_dist, r):
        """Compute the garnet size distribution based on size_dist input"""
        n_classes = len(r)
        if isinstance(size_dist, str):
            if size_dist == 'N':  # normal distribution
                mi = (self.r_min + self.r_max) / 2
                s = (mi - self.r_min) / 2
                finp = np.exp(-(r - mi)**2 / (2 * s**2)) / (np.sqrt(2 * np.pi) * s)
            elif size_dist == 'U':  # uniform distribution
                finp = np.ones(n_classes)
            else:
                raise ValueError("When provided as a string, size_dist must be 'N' or 'U'")
        elif isinstance(size_dist, (list, np.ndarray)):
            user_dist = np.array(size_dist, dtype=float)
            if user_dist.shape[0] != n_classes:
                raise ValueError("User-defined distribution must have length equal to garnet_classes")
            finp = user_dist
        else:
            raise ValueError("size_dist must be a string ('N' or 'U') or a numeric array")
        return finp

    def _normalize_distribution(self, finp, r):
        """Normalize the size distribution using volume and return reversed array (fnr)."""
        v = 4/3 * np.pi * r**3
        V = np.sum(v * finp)
        fn = finp / V
        return fn[::-1]  # reverse order: largest garnet goes first
        
    def get_retrograde_concentrations(self, new_t=None):
        """Get the retrograde concentrations of garnet-forming elements.

        Parameters:
            new_t (array-like, optional): New time values to interpolate the data
                and return the concentrations at these times. If None, the original
                data is returned. Uses a linear interpolation between datapoints.
            
        Returns:
            Concentrations (array): An array with the element concentrations and PTt data at each retrograde step.
        """

        # Compute normalized GVG
        GVi = np.array(self.gt_vol_frac)

        GVG = self._compute_normalized_GVG(GVi)
        ind = np.where((GVG == 1))[0]

        self._ind = ind
        
        tG = self.ti[ind]
        TG = self.Ti[ind]
        PG = self.Pi[ind]
        MnG = np.array(self.Mni)[ind]
        MgG = np.array(self.Mgi)[ind]
        FeG = np.array(self.Fei)[ind]
        CaG = np.array(self.Cai)[ind]
        GVG = np.array(GVG)[ind]
        
        if new_t is not None:
            # interpolate onto new time values
            interp_TG = interp1d(tG, TG, kind='linear', fill_value='extrapolate')
            interp_PG = interp1d(tG, PG, kind='linear', fill_value='extrapolate')
            interp_MnG = interp1d(tG, MnG, kind='linear', fill_value='extrapolate')
            interp_MgG = interp1d(tG, MgG, kind='linear', fill_value='extrapolate')
            interp_FeG = interp1d(tG, FeG, kind='linear', fill_value='extrapolate')
            interp_CaG = interp1d(tG, CaG, kind='linear', fill_value='extrapolate')
            interp_GVG = interp1d(tG, GVG, kind='linear', fill_value='extrapolate')

            new_TG = interp_TG(new_t)
            new_PG = interp_PG(new_t)
            new_MnG = interp_MnG(new_t)
            new_MgG = interp_MgG(new_t)
            new_FeG = interp_FeG(new_t)
            new_CaG = interp_CaG(new_t)
            new_GVG = interp_GVG(new_t)

            data = np.column_stack([new_t, new_TG, new_PG, new_MnG, new_MgG, new_FeG, new_CaG]).T

        else:
             data = np.column_stack([tG, TG, PG, MnG, MgG, FeG, CaG]).T



        return data



    def generate_garnets(self, size_dist='N'):
        """Generates garnet distributions.

        Parameters:
            size_dist (str or array-like): 
                'N' for a normal distribution, 
                'U' for a uniform distribution, or 
                a user-defined numeric distribution array of length equal to garnet_classes.
            
        Returns:
            garnets (list): List of garnet data dictionaries.
        """

        # Compute normalized GVG
        GVi = np.array(self.gt_vol_frac)
        GVG = self._compute_normalized_GVG(GVi)

        ind = np.where((GVG > 0.) & (GVG < 1))[0]

        self._ind = ind
        
        tG = self.ti[ind]
        TG = self.Ti[ind]
        PG = self.Pi[ind]
        MnG = np.array(self.Mni)[ind]
        MgG = np.array(self.Mgi)[ind]
        FeG = np.array(self.Fei)[ind]
        Cai = np.array(self.Cai)[ind]
        GVG = np.array(GVG)[ind]

        # Generate radius classes
        n_classes = self.garnet_classes
        r = np.linspace(self.r_min, self.r_max, n_classes, endpoint=True)
        dr = r[1] - r[0]

        # Determine distribution for garnet sizes
        if isinstance(size_dist, str):
            if size_dist == 'N':  # normal distribution (cut-off)
                mi = (self.r_min + self.r_max) / 2
                s = (mi - self.r_min) / 2
                finp = np.exp(-(r - mi)**2 / 2 / s**2) / np.sqrt(2 * np.pi) / s
            elif size_dist == 'U':  # uniform distribution
                finp = np.ones(n_classes)
            else:
                raise ValueError("When provided as a string, size_dist must be 'N' or 'U'")
        elif isinstance(size_dist, (list, np.ndarray)):
            user_dist = np.array(size_dist, dtype=float)
            if user_dist.shape[0] != n_classes:
                raise ValueError("User-defined distribution must have length equal to garnet_classes")
            finp = user_dist
        else:
            raise ValueError("size_dist must be a string ('N' or 'U') or a numeric array")
        

        # Normalize the distribution by volume
        fnr = self._normalize_distribution(finp, r)

        Gn = GVG / np.max(GVG)
        tGn = tG

        G, t, r_r, R = generate_distribution(n_classes, self.r_min, dr, fnr, Gn, tGn)


        # Interpolate physical properties along the garnet growth
        PGrw = np.interp(t, tG, PG)
        TGrw = np.interp(t, tG, TG)
        Mnrw = np.interp(t, tG, MnG)
        Mgrw = np.interp(t, tG, MgG)
        Ferw = np.interp(t, tG, FeG)
        Carw = 1 - Mnrw - Mgrw - Ferw

        garnets = []
        for i in range(n_classes):
            ind_range = np.arange(i, n_classes)
            Rr1 = R[i, ind_range]
            tr1 = t[ind_range]
            Pr1 = PGrw[ind_range]
            Tr1 = TGrw[ind_range]
            Mnr1 = Mnrw[ind_range]
            Mgr1 = Mgrw[ind_range]
            Fer1 = Ferw[ind_range]
            Car1 = 1 - Mnr1 - Mgr1 - Fer1

            dRr = Rr1[-1] / self.nR_diff
            Rrz = np.arange(dRr, Rr1[-1] + dRr, dRr)
            trz = np.interp(Rrz, Rr1, tr1)
            Prz = np.interp(Rrz, Rr1, Pr1)
            Trz = np.interp(Rrz, Rr1, Tr1)
            Mnrz = np.interp(Rrz, Rr1, Mnr1)
            Mgrz = np.interp(Rrz, Rr1, Mgr1)
            Ferz = np.interp(Rrz, Rr1, Fer1)
            Carz = 1 - Mnrz - Mgrz - Ferz

            Rr_full = np.concatenate([[0], Rrz])
            tr_full = np.concatenate([[trz[0]], trz])
            Pr_full = np.concatenate([[Prz[0]], Prz])
            Tr_full = np.concatenate([[Trz[0]], Trz])
            Mnr_full = np.concatenate([[Mnrz[0]], Mnrz])
            Mgr_full = np.concatenate([[Mgrz[0]], Mgrz])
            Fer_full = np.concatenate([[Ferz[0]], Ferz])
            Car_full = np.concatenate([[Carz[0]], Carz])

            iteration_data = {
                "Rr": Rr_full,
                "tr": tr_full,
                "Pr": Pr_full,
                "Tr": Tr_full,
                "Mnr": Mnr_full,
                "Mgr": Mgr_full,
                "Fer": Fer_full,
                "Car": Car_full
            }
            garnets.append(iteration_data)
        
        return garnets

    def plot_garnet_summary(self, size_dist='N', Rmax=None, garnet_no=0, path=None):
        """
        Plot a summary of the garnet formation results.
        
        Parameters:
            size_dist (str or array-like): 
                'N' for a normal distribution, 'U' for uniform,
                or a user-defined numeric array (length=garnet_classes).
            Rmax (float, optional): Maximum radius for subplot 5.
            path (str, optional): Path to save the figure.
        """
        # Compute normalized GVG
        GVi = np.array(self.gt_vol_frac)
        GVG = self._compute_normalized_GVG(GVi)

        ind = np.where((GVG > 0.) & (GVG < 1))[0]

        tG = self.ti[ind]
        TG = self.Ti[ind]
        PG = self.Pi[ind]
        MnG = np.array(self.Mni)[ind]
        MgG = np.array(self.Mgi)[ind]
        FeG = np.array(self.Fei)[ind]
        # GVG is already normalized
        GVG = GVG[ind]

        n_classes = self.garnet_classes
        r = np.linspace(self.r_min, self.r_max, n_classes, endpoint=True)
        dr = r[1] - r[0]

        # Determine distribution for garnet sizes
        if isinstance(size_dist, str):
            if size_dist == 'N':  # normal distribution (cut-off)
                mi = (self.r_min + self.r_max) / 2
                s = (mi - self.r_min) / 2
                finp = np.exp(-(r - mi)**2 / 2 / s**2) / np.sqrt(2 * np.pi) / s
            elif size_dist == 'U':  # uniform distribution
                finp = np.ones(n_classes)
            else:
                raise ValueError("When provided as a string, size_dist must be 'N' or 'U'")
        elif isinstance(size_dist, (list, np.ndarray)):
            user_dist = np.array(size_dist, dtype=float)
            if user_dist.shape[0] != n_classes:
                raise ValueError("User-defined distribution must have length equal to garnet_classes")
            finp = user_dist
        else:
            raise ValueError("size_dist must be a string ('N' or 'U') or a numeric array")
        
        # Normalize the distribution by volume
        fnr = self._normalize_distribution(finp, r)

        Gn = GVG / np.max(GVG)
        tGn = tG

        # Compute garnet distribution characteristics (using generate_distribution)
        G, t_arr, r_r, R = generate_distribution(n_classes, self.r_min, dr, fnr, Gn, tGn)
        
        # Interpolate physical properties along garnet growth
        PGrw = np.interp(t_arr, tG, PG)
        TGrw = np.interp(t_arr, tG, TG)
        Mnrw = np.interp(t_arr, tG, MnG)
        Mgrw = np.interp(t_arr, tG, MgG)
        Ferw = np.interp(t_arr, tG, FeG)
        
        # Set a default for Rmax if not provided
        if Rmax is None:
            Rmax = r_r.max()

        # --- Create the summary subplots ---
        fig, axs = plt.subplots(3, 2, figsize=(10, 15))
        fig.suptitle('Garnet formation summary')

        # Subplot 1: Garnet size
        axs[0, 0].set_title('Garnet size')
        axs[0, 0].set_xlabel('r')
        axs[0, 0].set_ylabel('f')
        axs[0, 0].set_xlim([r_r.min(), r_r.max()])
        axs[0, 0].plot(r_r, finp, '-')
        for i in range(n_classes):
            axs[0, 0].plot([r_r[i], r_r[i]], [0, finp[i]], '-')

        # Subplot 2: Classes' birth place
        axs[0, 1].set_title('Classes birth place')
        axs[0, 1].set_xlabel('T')
        axs[0, 1].set_ylabel('P')
        axs[0, 1].plot(self.Ti, self.Pi)
        axs[0, 1].plot(self.Ti, self.Pi, 'kx')
        axs[0, 1].plot(TGrw, PGrw, 'r.')  # locations where garnet classes formed

        # Subplot 3: Classes growth
        axs[1, 0].set_title('Classes growth')
        axs[1, 0].set_xlabel('t')
        axs[1, 0].set_ylabel('r')
        for i in range(0, n_classes, 10):
            axs[1, 0].plot(t_arr[i:], R[i, i:], 'r.-')
        
        # Subplot 4: Volume consumption
        axs[1, 1].set_title('Volume consumption')
        axs[1, 1].set_xlabel('t')
        axs[1, 1].set_ylabel('GV')
        axs[1, 1].set_ylim([0, 1.01])
        axs[1, 1].plot(tGn, Gn, 'k')
        axs[1, 1].plot(t_arr, G, 'mx')
        axs[1, 1].grid(True)

        # Subplot 5: Elemental compositions
        axs[2, 0].set_title('Mn(b) Mg(r) Fe(m) Ca(g)')
        axs[2, 0].set_xlabel('r')
        axs[2, 0].set_ylabel('c')
        axs[2, 0].set_xlim([0, Rmax])
        for i in np.arange(0, n_classes, 10):
            ind_local = np.arange(i, n_classes)
            rplt = R[i, :]
            axs[2, 0].plot(rplt[ind_local], Mnrw[ind_local], '-b',
                           rplt[ind_local], Mgrw[ind_local], 'r-',
                           rplt[ind_local], Ferw[ind_local], 'm-',
                           rplt[ind_local], (1 - Mnrw[ind_local] - Mgrw[ind_local] - Ferw[ind_local]), 'g-')
        # Subplot 6: Garnet composition
        for ax in [axs[2,0], axs[2,1]]:
            ax.set_title('Mn(b) Mg(r) Fe(m) Ca(g)')
            ax.set_xlabel('r')
            ax.set_ylabel('c')
            ax.set_xlim([0, Rmax])
            i = garnet_no ### highlight the defined garnet number
            ind_local = np.arange(i, n_classes)
            rplt = R[i, :]
            ax.plot(rplt[ind_local], Mnrw[ind_local], 'bx',
                        rplt[ind_local], Mgrw[ind_local], 'rx',
                        rplt[ind_local], Ferw[ind_local], 'mx',
                        rplt[ind_local], (1 - Mnrw[ind_local] - Mgrw[ind_local] - Ferw[ind_local]), 'gx', linewidth=2)
            
        axs[2,1].plot(rplt[ind_local], Mnrw[ind_local], 'b-',
            rplt[ind_local], Mgrw[ind_local], 'r-',
            rplt[ind_local], Ferw[ind_local], 'm-',
            rplt[ind_local], (1 - Mnrw[ind_local] - Mgrw[ind_local] - Ferw[ind_local]), 'g-', linewidth=2)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        if path is not None:
            plt.savefig(path)


        plt.show()




