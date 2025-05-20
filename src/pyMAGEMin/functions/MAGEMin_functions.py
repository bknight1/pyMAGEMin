import numpy as np
import sys
import warnings
import juliacall
from juliacall import Main as jl, convert as jlconvert
from .bulk_rock_functions import *


from pyMAGEMin import MAGEMin_C

# class MAGEMinDataExtractor:
#     def __init__(self):
#         # self.MAGEMin_C = juliacall.newmodule("MAGEMin")
#         # self.MAGEMin_C.seval("using MAGEMin_C")
#         pass

    # Helper functions that remain (they can also be made static or class methods if needed)
def extract_end_member(phase, MAGEMinOutput, end_member, sys_out):
    try:
        phase_ind = MAGEMinOutput.ph.index(phase)
        em_index = MAGEMinOutput.SS_vec[phase_ind].emNames.index(end_member)
        if sys_out.casefold() == 'wt':
            data = MAGEMinOutput.SS_vec[phase_ind].emFrac_wt[em_index]
        else:
            data = MAGEMinOutput.SS_vec[phase_ind].emFrac[em_index]
    except:
        data = 0.
    return data

def phase_frac(phase, MAGEMinOutput, sys_out):
    try:
        phase_ind = MAGEMinOutput.ph.index(phase)
        if sys_out.casefold() == 'wt':
            data = MAGEMinOutput.ph_frac_wt[phase_ind]
        elif sys_out.casefold() == 'vol':
            ph_names = MAGEMinOutput.ph
            n_ph = len(ph_names)
            V = np.zeros((n_ph, 1))
            for i, ph in enumerate(ph_names):
                ids = [j for j, p in enumerate(MAGEMinOutput.ph) if p == ph]
                if ids:
                    rho = sum(MAGEMinOutput.SS_vec[j].rho if j < MAGEMinOutput.n_SS 
                            else MAGEMinOutput.PP_vec[j - MAGEMinOutput.n_SS].rho for j in ids) / len(ids)
                    V[i, 0] = sum(MAGEMinOutput.ph_frac_wt[j] for j in ids) / rho
            V /= np.sum(V)
            data = V[phase_ind]
        else:
            data = MAGEMinOutput.ph_frac[phase_ind]
    except:
        data = 0.
    return data

class MAGEMinGarnetCalculator:
    def __init__(self):
        pass

    def generate_2D_grid_gt_endmembers(self, P, T, data, X, Xoxides, sys_in, sys_out='mol'):
        """
        Generates a 2D grid of garnet endmember mole fractions by calling MAGEMin.
        Returns arrays for garnet mole, weight, volume fractions and endmember fractions.
        """
        Xoxides = jlconvert(jl.Vector[jl.String], Xoxides)
        X = jlconvert(jl.Vector[jl.Float64], X)
        out = MAGEMin_C.multi_point_minimization(P, T, data, X=X, Xoxides=Xoxides, sys_in=sys_in)
        sys.stdout.flush()

        gt_mol_frac = np.zeros_like(P)
        gt_wt_frac = np.zeros_like(P)
        gt_vol_frac = np.zeros_like(P)
        py_arr  = np.zeros_like(P)
        alm_arr  = np.zeros_like(P)
        spss_arr  = np.zeros_like(P)
        gr_arr  = np.zeros_like(P)
        kho_arr  = np.zeros_like(P)

        for i in range(len(T)):
            gt_mol_frac[i] = phase_frac(phase="g", MAGEMinOutput=out[i], sys_out='mol')
            gt_wt_frac[i]  = phase_frac(phase="g", MAGEMinOutput=out[i], sys_out='wt')
            gt_vol_frac[i] = phase_frac(phase="g", MAGEMinOutput=out[i], sys_out='vol')

            py_arr[i]   = extract_end_member(phase="g", MAGEMinOutput=out[i], end_member="py", sys_out=sys_out)
            alm_arr[i]  = extract_end_member(phase="g", MAGEMinOutput=out[i], end_member="alm", sys_out=sys_out)
            spss_arr[i] = extract_end_member(phase="g", MAGEMinOutput=out[i], end_member="spss", sys_out=sys_out)
            gr_arr[i]   = extract_end_member(phase="g", MAGEMinOutput=out[i], end_member="gr", sys_out=sys_out)
            kho_arr[i]  = extract_end_member(phase="g", MAGEMinOutput=out[i], end_member="kho", sys_out=sys_out)

        return gt_mol_frac, gt_wt_frac, gt_vol_frac, py_arr, alm_arr, spss_arr, gr_arr, kho_arr

    def generate_2D_grid_gt_elements(self, P, T, data, X, Xoxides, sys_in, sys_out='mol'):
        """
        Generates 2D grids for garnet elements based on endmember fractions.
        Returns garnet fractions plus element molar fractions.
        """
        gt_mol_frac, gt_wt_frac, gt_vol_frac, py_arr, alm_arr, spss_arr, gr_arr, kho_arr = \
            self.generate_2D_grid_gt_endmembers(P, T, data, X, Xoxides, sys_in, sys_out)
            
        Mgi = np.zeros_like(P)
        Mni = np.zeros_like(P)
        Fei = np.zeros_like(P)
        Cai = np.zeros_like(P)
        for i in range(len(py_arr)):

            garnet_fractions = {"py": py_arr[i], "alm": alm_arr[i], "gr": gr_arr[i], "spss": spss_arr[i], "kho": kho_arr[i]}
            
            # Calculate the molar fractions of the elements in garnet
            mole_fractions = calculate_molar_fractions(garnet_fractions)

            if sys_out == 'wt':
                mole_fractions_list = [mole_fractions["Mg"], mole_fractions["Mn"], mole_fractions["Fe"], mole_fractions["Ca"]]
                wt_percent_list = convert_mol_percent_to_wt_percent(mole_fractions_list, ["Mg", "Mn", "Fe", "Ca"], atomic_mass_dict)

                Mgi[i] = wt_percent_list[0]
                Mni[i] = wt_percent_list[1]
                Fei[i] = wt_percent_list[2]
                Cai[i] = wt_percent_list[3]

            else:
                Mgi[i] = mole_fractions["Mg"]
                Mni[i] = mole_fractions["Mn"]
                Fei[i] = mole_fractions["Fe"]
                Cai[i] = mole_fractions["Ca"]

        return gt_mol_frac, gt_wt_frac, gt_vol_frac, Mgi, Mni, Fei, Cai

    def gt_single_point_calc_endmembers(self, P, T, data, X, Xoxides, sys_in, sys_out='mol'):
        """
        Calculate garnet endmember fractions for a single P-T point.
        Returns fractions and the full MAGEMin output.
        """
        Xoxides = jlconvert(jl.Vector[jl.String], Xoxides)
        X = jlconvert(jl.Vector[jl.Float64], X)
        out = MAGEMin_C.single_point_minimization(P, T, data, X=X, Xoxides=Xoxides, sys_in=sys_in)
        sys.stdout.flush()
        gt_frac = gt_wt = gt_vol = py = alm = spss = gr = kho = 0.
        if 'g' in out.ph:
            gt_frac  = phase_frac(phase="g", MAGEMinOutput=out, sys_out='mol')
            gt_wt    = phase_frac(phase="g", MAGEMinOutput=out, sys_out='wt')
            gt_vol   = phase_frac(phase="g", MAGEMinOutput=out, sys_out='vol')

            py  = extract_end_member(phase="g", MAGEMinOutput=out, end_member="py", sys_out=sys_out)
            alm = extract_end_member(phase="g", MAGEMinOutput=out, end_member="alm", sys_out=sys_out)
            spss= extract_end_member(phase="g", MAGEMinOutput=out, end_member="spss", sys_out=sys_out)
            gr  = extract_end_member(phase="g", MAGEMinOutput=out, end_member="gr", sys_out=sys_out)
            kho = extract_end_member(phase="g", MAGEMinOutput=out, end_member="kho", sys_out=sys_out)
        
        return gt_frac, gt_wt, gt_vol, py, alm, spss, gr, kho, out

    def gt_single_point_calc_elements(self, P, T, data, X, Xoxides, sys_in, sys_out='mol'):
        """
        Calculate garnet element fractions for a single P-T point.
        Returns garnet fractions and molar fractions of elements.
        """
        gt_frac, gt_wt, gt_vol, py, alm, spss, gr, kho, out = \
            self.gt_single_point_calc_endmembers(P, T, data, X, Xoxides, sys_in, sys_out)
        
        garnet_fractions = {"py": py, "alm": alm, "gr": gr, "spss": spss, "kho": kho}
        
        # Calculate the molar fractions of the elements in garnet
        mole_fractions = calculate_molar_fractions(garnet_fractions)

        if sys_out == 'wt':
            mole_fractions_list = [mole_fractions["Mg"], mole_fractions["Mn"], mole_fractions["Fe"], mole_fractions["Ca"]]
            wt_percent_list = convert_mol_percent_to_wt_percent(mole_fractions_list, ["Mg", "Mn", "Fe", "Ca"], atomic_mass_dict)

            Mg = wt_percent_list[0]
            Mn = wt_percent_list[1]
            Fe = wt_percent_list[2]
            Ca = wt_percent_list[3]

        else:
            Mg = mole_fractions["Mg"]
            Mn = mole_fractions["Mn"]
            Fe = mole_fractions["Fe"]
            Ca = mole_fractions["Ca"]
        
        return gt_frac, gt_wt, gt_vol, Mg, Mn, Fe, Ca, out

    def gt_along_path(self, P, T, data, X, Xoxides, sys_in, sys_out='mol', fractionate=False):
        """
        Calculates garnet composition and fractionation along a P-T path.
        Returns arrays of garnet fractions and element mole fractions along the path.
        """
        Xoxides = jlconvert(jl.Vector[jl.String], Xoxides)
        X = jlconvert(jl.Vector[jl.Float64], X)
        n_points = len(P)
        gt_wt_frac = np.zeros(n_points)
        gt_mol_frac = np.zeros(n_points)
        gt_vol_frac = np.zeros(n_points)
        d_gt_mol_frac = np.zeros(n_points)
        d_gt_wt_frac = np.zeros(n_points)
        Mgi = np.zeros(n_points)
        Mni = np.zeros(n_points)
        Fei = np.zeros(n_points)
        Cai = np.zeros(n_points)
        
        ### loop over P-T points to get the garnet data
        for i in range(n_points):
            gt_frac, gt_wt, gt_vol, py, alm, spss, gr, kho, out = \
                self.gt_single_point_calc_endmembers(P[i], T[i], data, X, Xoxides, sys_in, sys_out='mol')
            

            gt_mol_frac[i] = gt_frac
            gt_wt_frac[i] = gt_wt
            gt_vol_frac[i] = gt_vol


            garnet_fractions = {"py": py, "alm": alm, "gr": gr, "spss": spss, "kho": kho}
            
            # Calculate the molar fractions of the elements in garnet
            mole_fractions = calculate_molar_fractions(garnet_fractions)

            if sys_out == 'wt':
                mole_fractions_list = [mole_fractions["Mg"], mole_fractions["Mn"], mole_fractions["Fe"], mole_fractions["Ca"]]
                wt_percent_list = convert_mol_percent_to_wt_percent(mole_fractions_list, ["Mg", "Mn", "Fe", "Ca"], atomic_mass_dict)

                Mgi[i] = wt_percent_list[0]
                Mni[i] = wt_percent_list[1]
                Fei[i] = wt_percent_list[2]
                Cai[i] = wt_percent_list[3]

            else:
                Mgi[i] = mole_fractions["Mg"]
                Mni[i] = mole_fractions["Mn"]
                Fei[i] = mole_fractions["Fe"]
                Cai[i] = mole_fractions["Ca"]

            if fractionate:
                ### Should we fractionate other phases too ???
                from .MAGEMin_functions import PhaseFunctions
                phase_functions = PhaseFunctions()
                # Calculate the derivative of garnet fraction.
                if sys_in == "wt":
                    d_frac, X = phase_functions.fractionate_phase(out.X, phase="g", frac_array=gt_wt_frac, i=i, out=out, sys_in=sys_in)
                    d_gt_wt_frac[i] = d_frac
                elif sys_in == "mol":
                    d_frac, X = phase_functions.fractionate_phase(out.X, phase="g", frac_array=gt_mol_frac, i=i, out=out, sys_in=sys_in)
                    d_gt_mol_frac[i] = d_frac
                else:
                    Warnings.warn("Invalid system input. No fractionation applied.")
    

        return gt_mol_frac, gt_wt_frac, gt_vol_frac, d_gt_mol_frac, d_gt_wt_frac, Mgi, Mni, Fei, Cai


class PhaseFunctions:
    def __init__(self):
        pass

    def find_phase_in(self, P, initial_T, data, phase, sys_out='mol', precision=1., verbose=False):
        """
        Finds the solidus temperature (where the liquid fraction becomes zero).
        """
        solidus_T = float(initial_T)
        out = MAGEMin_C.single_point_minimization(P, solidus_T, data)
        phasefrac = phase_frac(phase=phase, MAGEMinOutput=out, sys_out=sys_out)
        if phasefrac == 0:
            raise ValueError(f'Increase initial temperature guess; phase fraction = {phasefrac}')
        if phasefrac == 1:
            warnings.warn(f'Decrease initial temperature guess; phase fraction = {phasefrac}')
        while phasefrac > 0:
            solidus_T -= precision
            out = MAGEMin_C.single_point_minimization(P, solidus_T, data)
            phasefrac = phase_frac(phase=phase, MAGEMinOutput=out, sys_out=sys_out)
            if verbose:
                print(f"phase frac: {phasefrac:.4f}, T: {solidus_T:.2f}")
        return solidus_T

    def find_phase_saturation(self, P, initial_T, data, phase, sys_out='mol', precision=1., verbose=False):
        """
        Finds the liquidus temperature (where the liquid fraction becomes one).
        """
        liquidus_T = float(initial_T)
        out = MAGEMin_C.single_point_minimization(P, liquidus_T, data)
        phasefrac = phase_frac(phase=phase, MAGEMinOutput=out, sys_out=sys_out)
        if phasefrac == 1:
            raise ValueError(f'Decrease initial temperature guess; liquid fraction = {phasefrac}')
        if phasefrac == 0:
            warnings.warn(f'Increase initial temperature guess; liquid fraction = {phasefrac}')
        while phasefrac < 1:
            liquidus_T += precision
            out = MAGEMin_C.single_point_minimization(P, liquidus_T, data)
            phasefrac = phase_frac(phase=phase, MAGEMinOutput=out, sys_out=sys_out)
            if verbose:
                print(f"phase frac: {phasefrac:.4f}, T: {liquidus_T:.2f}")
        return liquidus_T

    def fractionate_phase(self, X, phase, frac_array, i, out, sys_in='mol'):
        """
        Calculates the derivative of the phase fraction and adjusts the bulk composition X.
        If the phase is missing from out.ph, the fractionation is skipped.
        """
        # Compute the fraction derivative.
        d_frac = frac_array[i] - frac_array[i-1] if i > 0 else frac_array[0]
        
        # Only adjust if the phase fraction is positive and the phase exists.
        if (frac_array[i] > 0) & (phase in out.ph) & (d_frac > 0):
            phase_index = out.ph.index(phase)
            if sys_in == "wt":
                X = out.bulk_wt
                new_X = out.bulk_wt - (np.array(out.bulk_wt) *
                                    np.array(out.SS_vec[phase_index].Comp_wt) *
                                    d_frac)
            elif sys_in == "mol":
                X = out.bulk
                new_X = out.bulk - (np.array(out.bulk) *
                                    np.array(out.SS_vec[phase_index].Comp) *
                                    d_frac)
            else:
                Warnings.warn("Invalid system input. No fractionation applied.")
                d_frac, new_X = 0, out.bulk
        else:
            d_frac, new_X = 0, out.bulk

        return d_frac, new_X