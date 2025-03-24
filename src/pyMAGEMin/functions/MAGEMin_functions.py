import numpy as np
import sys
import warnings
import juliacall
from juliacall import Main as jl, convert as jlconvert
from .bulk_rock_functions import *

# class MAGEMinDataExtractor:
#     def __init__(self):
#         # self.MAGEMin_C = juliacall.newmodule("MAGEMin")
#         # self.MAGEMin_C.seval("using MAGEMin_C")
#         pass

    # Helper functions that remain (they can also be made static or class methods if needed)
def extract_end_member(phase, MAGEMinOutput, end_member, sys_in):
    try:
        phase_ind = MAGEMinOutput.ph.index(phase)
        em_index = MAGEMinOutput.SS_vec[phase_ind].emNames.index(end_member)
        if sys_in.casefold() == 'wt':
            data = MAGEMinOutput.SS_vec[phase_ind].emFrac_wt[em_index]
        else:
            data = MAGEMinOutput.SS_vec[phase_ind].emFrac[em_index]
    except:
        data = 0.
    return data

def phase_frac(phase, MAGEMinOutput, sys_in):
    try:
        phase_ind = MAGEMinOutput.ph.index(phase)
        if sys_in.casefold() == 'wt':
            data = MAGEMinOutput.ph_frac_wt[phase_ind]
        elif sys_in.casefold() == 'vol':
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
        self.MAGEMin_C = juliacall.newmodule("MAGEMin")
        self.MAGEMin_C.seval("using MAGEMin_C")

    def generate_2D_grid_gt_endmembers(self, P, T, data, X, Xoxides, sys_in):
        """
        Generates a 2D grid of garnet endmember mole fractions by calling MAGEMin.
        Returns arrays for garnet mole, weight, volume fractions and endmember fractions.
        """
        Xoxides = jlconvert(jl.Vector[jl.String], Xoxides)
        X = jlconvert(jl.Vector[jl.Float64], X)
        out = self.MAGEMin_C.multi_point_minimization(P, T, data, X=X, Xoxides=Xoxides, sys_in=sys_in)
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
            gt_mol_frac[i] = phase_frac(phase="g", MAGEMinOutput=out[i], sys_in='mol')
            gt_wt_frac[i]  = phase_frac(phase="g", MAGEMinOutput=out[i], sys_in='wt')
            gt_vol_frac[i] = phase_frac(phase="g", MAGEMinOutput=out[i], sys_in='vol')

            py_arr[i]   = extract_end_member(phase="g", MAGEMinOutput=out[i], end_member="py", sys_in=sys_in)
            alm_arr[i]  = extract_end_member(phase="g", MAGEMinOutput=out[i], end_member="alm", sys_in=sys_in)
            spss_arr[i] = extract_end_member(phase="g", MAGEMinOutput=out[i], end_member="spss", sys_in=sys_in)
            gr_arr[i]   = extract_end_member(phase="g", MAGEMinOutput=out[i], end_member="gr", sys_in=sys_in)
            kho_arr[i]  = extract_end_member(phase="g", MAGEMinOutput=out[i], end_member="kho", sys_in=sys_in)

        return gt_mol_frac, gt_wt_frac, gt_vol_frac, py_arr, alm_arr, spss_arr, gr_arr, kho_arr

    def generate_2D_grid_gt_elements(self, P, T, data, X, Xoxides, sys_in):
        """
        Generates 2D grids for garnet elements based on endmember fractions.
        Returns garnet fractions plus element molar fractions.
        """
        gt_mol_frac, gt_wt_frac, gt_vol_frac, py_arr, alm_arr, spss_arr, gr_arr, kho_arr = \
            self.generate_2D_grid_gt_endmembers(P, T, data, X, Xoxides, sys_in)
            
        Mgi = np.zeros_like(P)
        Mni = np.zeros_like(P)
        Fei = np.zeros_like(P)
        Cai = np.zeros_like(P)
        for i in range(len(py_arr)):
            garnet_fractions = {"py": py_arr[i], "alm": alm_arr[i], "gr": gr_arr[i], 
                                "spss": spss_arr[i], "kho": kho_arr[i]}
            mole_fractions = calculate_molar_fractions(garnet_fractions)
            Mgi[i] = mole_fractions["Mg"]
            Mni[i] = mole_fractions["Mn"]
            Fei[i] = mole_fractions["Fe"]
            Cai[i] = mole_fractions["Ca"]
        return gt_mol_frac, gt_wt_frac, gt_vol_frac, Mgi, Mni, Fei, Cai

    def gt_single_point_calc_endmembers(self, P, T, data, X, Xoxides, sys_in):
        """
        Calculate garnet endmember fractions for a single P-T point.
        Returns fractions and the full MAGEMin output.
        """
        Xoxides = jlconvert(jl.Vector[jl.String], Xoxides)
        X = jlconvert(jl.Vector[jl.Float64], X)
        out = self.MAGEMin_C.single_point_minimization(P, T, data, X=X, Xoxides=Xoxides, sys_in=sys_in)
        sys.stdout.flush()
        gt_frac = gt_wt = gt_vol = py = alm = spss = gr = kho = 0.
        if 'g' in out.ph:
            gt_frac  = phase_frac(phase="g", MAGEMinOutput=out, sys_in='mol')
            gt_wt    = phase_frac(phase="g", MAGEMinOutput=out, sys_in='wt')
            gt_vol   = phase_frac(phase="g", MAGEMinOutput=out, sys_in='vol')

            py  = extract_end_member(phase="g", MAGEMinOutput=out, end_member="py", sys_in=sys_in)
            alm = extract_end_member(phase="g", MAGEMinOutput=out, end_member="alm", sys_in=sys_in)
            spss= extract_end_member(phase="g", MAGEMinOutput=out, end_member="spss", sys_in=sys_in)
            gr  = extract_end_member(phase="g", MAGEMinOutput=out, end_member="gr", sys_in=sys_in)
            kho = extract_end_member(phase="g", MAGEMinOutput=out, end_member="kho", sys_in=sys_in)
        return gt_frac, gt_wt, gt_vol, py, alm, spss, gr, kho, out

    def gt_single_point_calc_elements(self, P, T, data, X, Xoxides, sys_in):
        """
        Calculate garnet element fractions for a single P-T point.
        Returns garnet fractions and molar fractions of elements.
        """
        gt_frac, gt_wt, gt_vol, py, alm, spss, gr, kho, out = \
            self.gt_single_point_calc_endmembers(P, T, data, X, Xoxides, sys_in)
        garnet_fractions = {"py": py, "alm": alm, "gr": gr, "spss": spss, "kho": kho}
        mole_fractions = calculate_molar_fractions(garnet_fractions)
        Mg = mole_fractions["Mg"]
        Mn = mole_fractions["Mn"]
        Fe = mole_fractions["Fe"]
        Ca = mole_fractions["Ca"]
        return gt_frac, gt_wt, gt_vol, Mg, Mn, Fe, Ca, out

    def gt_over_path(self, P, T, data, X, Xoxides, sys_in, fractionate=False):
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
        for i in range(n_points):
            gt_frac, gt_wt, gt_vol, py, alm, spss, gr, kho, out = \
                self.gt_single_point_calc_endmembers(P[i], T[i], data, X, Xoxides, sys_in)
            if i == 0:
                gt_mol_frac_initial = gt_frac
                gt_wt_frac_initial = gt_wt
                gt_vol_frac_initial = gt_vol
            gt_mol_frac[i] = gt_frac - gt_mol_frac_initial
            gt_wt_frac[i] = gt_wt - gt_wt_frac_initial
            gt_vol_frac[i] = gt_vol - gt_vol_frac_initial

            garnet_fractions = {"py": py, "alm": alm, "gr": gr, "spss": spss, "kho": kho}
            mole_fractions = calculate_molar_fractions(garnet_fractions)
            Mgi[i] = mole_fractions["Mg"]
            Mni[i] = mole_fractions["Mn"]
            Fei[i] = mole_fractions["Fe"]
            Cai[i] = mole_fractions["Ca"]

            if fractionate:
                # Additional fractionation logic can be incorporated here.
                if i > 0:
                    d_gt_mol_frac[i] = gt_mol_frac[i] - gt_mol_frac[i-1]
                    d_gt_wt_frac[i] = gt_wt_frac[i] - gt_wt_frac[i-1]
                else:
                    d_gt_mol_frac[i] = gt_mol_frac[0]
                    d_gt_wt_frac[i] = gt_wt_frac[0]
                # Example: adjust bulk composition 'X' using the change in garnet fraction.
                if gt_frac > 0:
                    garnet_ind = out.ph.index("g")
                    Xoxides = out.oxides
                    if sys_in == "wt":
                        X = out.bulk_wt - (np.array(out.bulk_wt) * np.array(out.SS_vec[garnet_ind].Comp_wt) * d_gt_wt_frac[i])
                    elif sys_in == "mol":
                        X = out.bulk - (np.array(out.bulk) * np.array(out.SS_vec[garnet_ind].Comp) * d_gt_mol_frac[i])
        return gt_mol_frac, gt_wt_frac, gt_vol_frac, d_gt_mol_frac, d_gt_wt_frac, Mgi, Mni, Fei, Cai


class MAGEMinCalculator:
    def __init__(self):
        self.MAGEMin_C = juliacall.newmodule("MAGEMin")
        self.MAGEMin_C.seval("using MAGEMin_C")

    def find_phase_in(self, P, initial_T, data, phase, sys_in='mol', precision=1., verbose=False):
        """
        Finds the solidus temperature (where the liquid fraction becomes zero).
        """
        solidus_T = float(initial_T)
        out = self.MAGEMin_C.single_point_minimization(P, solidus_T, data)
        phasefrac = phase_frac(phase=phase, MAGEMinOutput=out, sys_in=sys_in)
        if phasefrac == 0:
            raise ValueError(f'Increase initial temperature guess; phase fraction = {phasefrac}')
        if phasefrac == 1:
            warnings.warn(f'Decrease initial temperature guess; phase fraction = {phasefrac}')
        while phasefrac > 0:
            solidus_T -= precision
            out = self.MAGEMin_C.single_point_minimization(P, solidus_T, data)
            phasefrac = phase_frac(phase=phase, MAGEMinOutput=out, sys_in=sys_in)
            if verbose:
                print(f"phase frac: {phasefrac:.4f}, T: {solidus_T:.2f}")
        return solidus_T

    def find_phase_saturation(self, P, initial_T, data, phase, sys_in='mol', precision=1., verbose=False):
        """
        Finds the liquidus temperature (where the liquid fraction becomes one).
        """
        liquidus_T = float(initial_T)
        out = self.MAGEMin_C.single_point_minimization(P, liquidus_T, data)
        phasefrac = phase_frac(phase=phase, MAGEMinOutput=out, sys_in=sys_in)
        if phasefrac == 1:
            raise ValueError(f'Decrease initial temperature guess; liquid fraction = {phasefrac}')
        if phasefrac == 0:
            warnings.warn(f'Increase initial temperature guess; liquid fraction = {phasefrac}')
        while phasefrac < 1:
            liquidus_T += precision
            out = self.MAGEMin_C.single_point_minimization(P, liquidus_T, data)
            phasefrac = phase_frac(phase=phase, MAGEMinOutput=out, sys_in=sys_in)
            if verbose:
                print(f"phase frac: {phasefrac:.4f}, T: {liquidus_T:.2f}")
        return liquidus_T