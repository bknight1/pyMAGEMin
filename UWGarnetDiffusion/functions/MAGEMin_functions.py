import numpy as np


from julia import MAGEMin_C

from UWGarnetDiffusion.functions.bulk_rock_functions import *

from UWGarnetDiffusion.functions.mineral_growth_functions import *


def generate_2D_grid_endmembers(P, T, data, X, Xoxides, sys_in, verbose=0):
    """
    Generate a 2D grid for multi-point minimization.

    Parameters:
    - P: pressure
    - T: temperature
    - db: database
    - X: unknown
    - Xoxides: unknown
    - verbose: verbosity level (default 0)

    Returns:
    - gt_vol: garnet volume
    - py_frac: pyrope fraction (MgO)
    - alm_frac: almandine fraction (FeO)
    - spss_frac: spessartine fraction (MnO)
    - gr_frac: grossular fraction (CaO)
    - kho_frac: khoharite fraction (???)
    """

        
    # data    =   MAGEMin_C.Initialize_MAGEMin(db, verbose=verbose);
        
    out     =   MAGEMin_C.multi_point_minimization(P, T, data, X=X, Xoxides=Xoxides, sys_in=sys_in);


    ### release the data
    # MAGEMin_C.Finalize_MAGEMin(data);

    gt_mol_frac    = np.zeros_like(P)

    py_arr  = np.zeros_like(P)
    alm_arr  = np.zeros_like(P)
    spss_arr  = np.zeros_like(P)
    gr_arr  = np.zeros_like(P)
    kho_arr  = np.zeros_like(P)




    for i in range(len(T)):
        ### get garnet data
        gt_mol_frac[i] = ( extract_phase_mol_frac(phase="g", MAGEMinOutput=out[i]) )

        ### get end member data of the garnet
        py_arr[i]   = (extract_end_member_mol_frac(phase="g", MAGEMinOutput=out[i], end_member="py"))  # pyrope (py) = MgO
        alm_arr[i]  = (extract_end_member_mol_frac(phase="g", MAGEMinOutput=out[i], end_member="alm")) # almandine (alm) = FeO
        spss_arr[i] = (extract_end_member_mol_frac(phase="g", MAGEMinOutput=out[i], end_member="spss")) # spessartine (sp) = MnO
        gr_arr[i]   = (extract_end_member_mol_frac(phase="g", MAGEMinOutput=out[i], end_member="gr")) # grossular (gr) = CaO
        kho_arr[i]  = (extract_end_member_mol_frac(phase="g", MAGEMinOutput=out[i], end_member="kho")) # khoharite (kho) = MgO



    
    return gt_mol_frac, py_arr, alm_arr, spss_arr, gr_arr, kho_arr
    # return gt_mol_frac, Mgi, Mni, Fei, Cai

def generate_2D_grid_elements(P, T, data, X, Xoxides, sys_in, verbose=0):
    Mgi = np.zeros_like(P)
    Mni = np.zeros_like(P)
    Fei = np.zeros_like(P)
    Cai = np.zeros_like(P)

    gt_mol_frac, py_arr, alm_arr, spss_arr, gr_arr, kho_arr = generate_2D_grid_endmembers(P, T, data, X, Xoxides, sys_in, verbose=0)

    for i in range(len(py_arr)):
        garnet_fractions = {"py": py_arr[i], "alm": alm_arr[i], "gr": gr_arr[i], "spss": spss_arr[i], "kho": kho_arr[i]}

        # ### Calculate molar fractions of Mg, Ca, Fe, and Mn
        mole_fractions = calculate_molar_fractions(garnet_fractions)

        # ### append to lists for each element
        Mgi[i] = (mole_fractions["Mg"])
        Mni[i] = (mole_fractions["Mn"])
        Fei[i] = (mole_fractions["Fe"])
        Cai[i] = (mole_fractions["Ca"])

    return gt_mol_frac, Mgi, Mni, Fei, Cai


    

def gt_single_point_calc_endmembers(P, T, data, X, Xoxides, sys_in, verbose=0):
    """
    Calculate garnet fractions over a given path, includes fractionation as the garnet forms.

    Args:
        P (list): List of pressures.
        T (list): List of temperatures.
        db (object): Database object.
        X (list): List of mole fractions.
        Xoxides (list): List of oxide mole fractions.
        verbose (bool, optional): Verbosity level. Defaults to False.
        fractionate (bool, optional): If True, fractionate the garnet. Defaults to False.

    Returns:
        tuple: Tuple containing lists of garnet mole fractions, Mg fractions, Mn fractions, Fe fractions, and Ca fractions.
    """

    ### do calculation
    out = MAGEMin_C.single_point_minimization(P, T, data, X=X, Xoxides=Xoxides, sys_in=sys_in)

    gt_frac = 0.
    py = 0.
    alm = 0.
    spss = 0.
    gr = 0.
    kho = 0.

    Mg = 0.
    Mn = 0.
    Fe = 0.
    Ca = 0.

    if 'g' in out.ph:
        gt_frac  = extract_phase_mol_frac("g", out) 
        py  = extract_end_member_mol_frac(phase="g", MAGEMinOutput=out, end_member="py")
        alm = extract_end_member_mol_frac(phase="g", MAGEMinOutput=out, end_member="alm")
        spss= extract_end_member_mol_frac(phase="g", MAGEMinOutput=out, end_member="spss")
        gr  = extract_end_member_mol_frac(phase="g", MAGEMinOutput=out, end_member="gr")
        kho = extract_end_member_mol_frac(phase="g", MAGEMinOutput=out, end_member="kho")

        # garnet_fractions = {"Py": py, "Alm": alm, "Gr": gr, "Spss": spss, "Kho": kho}

        # ### Calculate molar fractions of Mg, Ca, Fe, and Mn
        # mole_fractions = calculate_molar_fractions(garnet_fractions)
        # #### append to lists for each element
        # Mg = (mole_fractions["Mg"])
        # Mn = (mole_fractions["Mn"])
        # Fe = (mole_fractions["Fe"])
        # Ca = (mole_fractions["Ca"])


    

    # return gt_frac, Mg, Mn, Fe, Ca
    return gt_frac, py, alm, spss, gr, kho

def gt_single_point_calc_elements(P, T, data, X, Xoxides, sys_in, verbose=0):
    """
    Calculate garnet fractions over a given path, includes fractionation as the garnet forms.

    Args:
        P (list): List of pressures.
        T (list): List of temperatures.
        db (object): Database object.
        X (list): List of mole fractions.
        Xoxides (list): List of oxide mole fractions.
        verbose (bool, optional): Verbosity level. Defaults to False.
        fractionate (bool, optional): If True, fractionate the garnet. Defaults to False.

    Returns:
        tuple: Tuple containing lists of garnet mole fractions, Mg fractions, Mn fractions, Fe fractions, and Ca fractions.
    """

    ### use the previous function to get the EM fractions
    gt_frac, py, alm, spss, gr, kho = gt_single_point_calc_endmembers(P, T, data, X=X, Xoxides=Xoxides, sys_in=sys_in)


    Mg = 0.
    Mn = 0.
    Fe = 0.
    Ca = 0.

    garnet_fractions = {"py": py, "alm": alm, "gr": gr, "spss": spss, "kho": kho}

    ### Calculate molar fractions of Mg, Ca, Fe, and Mn
    mole_fractions = calculate_molar_fractions(garnet_fractions)
    #### append to lists for each element
    Mg = (mole_fractions["Mg"])
    Mn = (mole_fractions["Mn"])
    Fe = (mole_fractions["Fe"])
    Ca = (mole_fractions["Ca"])


    

    return gt_frac, Mg, Mn, Fe, Ca



# def garnet_over_path_no_frac(P, T, data, X, Xoxides, sys_in, verbose=0, ):
#     """
#     Perform a multi-point minimization over the given parameters and return the garnet mole fractions and element data.

#     Args:
#         P (list): List of pressures.
#         T (list): List of temperatures.
#         db (str): Database name.
#         X (dict): Dictionary of element names and their molar fractions.
#         Xoxides (dict): Dictionary of oxide names and their molar fractions.
#         verbose (int, optional): Verbosity level. Defaults to 0.

#     Returns:
#         tuple: Tuple containing garnet mole fractions and molar fractions of Mg, Mn, Fe, and Ca.
#     """


#     out     =   MAGEMin_C.multi_point_minimization(P, T, data, X=X, Xoxides=Xoxides, sys_in=sys_in)


#     ### release the data
#     # MAGEMin_C.Finalize_MAGEMin(data)

#     gt_mol_frac    = []

#     Mgi = []  # pyrope (py) & Khoharite (kho) = MgO
#     Fei = []  # almandine (alm) = FeO
#     Mni = []  # spessartine (sp) = MnO
#     Cai = []  # grossular (gr) = CaO

#     ### TODO work out if the loop over each PT point is the best option
#     for i in range(len(P)):
#         ### get garnet data
#         if i ==0:
#             gt_mol_frac_initial = extract_phase_mol_frac("g", out[i]) 

#         gt_mol_frac_at_PT = extract_phase_mol_frac("g", out[i]) 

#         # The first gt_frac must be 0, we will call it reduction (we neglect the previous history - we recommend starting from a zero field of garnet)
#         gt_frac = gt_mol_frac_at_PT - gt_mol_frac_initial
    
#         gt_mol_frac.append( gt_frac )



#         ### get end member data of the garnet
#         py   = (extract_end_member_mol_frac(phase="g", MAGEMinOutput=out[i], end_member="py"))
#         alm  = (extract_end_member_mol_frac(phase="g", MAGEMinOutput=out[i], end_member="alm"))
#         spss = (extract_end_member_mol_frac(phase="g", MAGEMinOutput=out[i], end_member="spss"))
#         gr   = (extract_end_member_mol_frac(phase="g", MAGEMinOutput=out[i], end_member="gr"))
#         kho  = (extract_end_member_mol_frac(phase="g", MAGEMinOutput=out[i], end_member="kho"))


#         garnet_fractions = {"Py": py, "Alm": alm, "Gr": gr, "Spss": spss, "Kho": kho}

#         ### Calculate molar fractions of Mg, Ca, Fe, and Mn
#         mole_fractions = calculate_molar_fractions(garnet_fractions)

#         ### append to lists for each element
#         Mgi.append(mole_fractions["Mg"])
#         Mni.append(mole_fractions["Mn"])
#         Fei.append(mole_fractions["Fe"])
#         Cai.append(mole_fractions["Ca"])
        


#     return gt_mol_frac, Mgi, Mni, Fei, Cai

def garnet_over_path(P, T, data, X, Xoxides, sys_in, verbose=0, fractionate=0):
    """
    Calculate garnet fractions over a given path, includes fractionation as the garnet forms.

    Args:
        P (list): List of pressures.
        T (list): List of temperatures.
        db (object): Database object.
        X (list): List of mole fractions.
        Xoxides (list): List of oxide mole fractions.
        verbose (bool, optional): Verbosity level. Defaults to False.
        fractionate (bool, optional): If True, fractionate the garnet. Defaults to False.

    Returns:
        tuple: Tuple containing lists of garnet mole fractions, Mg fractions, Mn fractions, Fe fractions, and Ca fractions.
    """


    # gt_wt_frac      = []
    gt_mol_frac  = np.zeros(len(P))

    Mgi = np.zeros(len(P))  # pyrope (py) & Khoharite (kho) = MgO
    Fei = np.zeros(len(P)) # almandine (alm) = FeO
    Mni = np.zeros(len(P))  # spessartine (sp) = MnO
    Cai = np.zeros(len(P))  # grossular (gr) = CaO

    py_arr   = np.zeros(len(P))
    alm_arr  = np.zeros(len(P))
    spss_arr = np.zeros(len(P))
    gr_arr   = np.zeros(len(P))
    kho_arr  = np.zeros(len(P))

    for i in range(len(P)):
        gt_fr, py, alm, spss, gr, kho = gt_single_point_calc(P[i], T[i], data, X, Xoxides, sys_in, verbose=verbose)

        py_arr[i] = py
        alm_arr[i] = alm
        spss_arr[i] = spss
        gr_arr[i] = gr
        kho_arr[i] = kho

        ### convert end members to Fe, Mg, Mn, Ca amounts
        garnet_fractions = {"py": py, "alm": alm, "gr": gr, "spss": spss, "kho": kho}


        ### Calculate molar fractions of Mg, Ca, Fe, and Mn
        mole_fractions = calculate_molar_fractions(garnet_fractions)

        # ### append to lists for each element
        Mgi[i] = (mole_fractions["Mg"])
        Mni[i] = (mole_fractions["Mn"])
        Fei[i] = (mole_fractions["Fe"])
        Cai[i] = (mole_fractions["Ca"])


        ### get garnet data
        if i ==0:
            gt_mol_frac_initial = gt_fr

        gt_mol_frac_at_PT = gt_fr

        # The first gt_frac must be 0, we will call it reduction (we neglect the previous history - we recommend starting from a zero field of garnet)
        gt_frac = gt_mol_frac_at_PT  - gt_mol_frac_initial
        
        gt_mol_frac[i] = ( gt_frac )

        ### Fractionate the bulk rock due to the garnet growth
        if fractionate == True:
            """
            This approach is based on the assumption that the change in garnet fraction from one 
            P-T point to the next represents the amount of garnet growth (or reduction) 
            that occurs due to the changes in pressure and temperature.
            """
            if i ==0:
                d_gt_frac = gt_mol_frac[0] ### for the initial point
            else:
                d_gt_frac = gt_mol_frac[-1] - gt_mol_frac[-2] ### for every other point along the PT path
                

            ### recalculate bulk rock with new molar% due to fractionation
            bulk_rock = dict(zip(Xoxides, X))
            # print(bulk_rock)

            X_dict = modify_bulk_rock_due_to_garnet(bulk_rock, garnet_fraction=d_gt_frac, garnet_end_member_fractions=garnet_fractions)
            X = list(X_dict.values())
            
            # print(X)

            # X = adjust_bulk_composition_for_garnet(Xoxides, X, d_gt_frac, garnet_fractions)
            

    # ### release the data
    # MAGEMin_C.Finalize_MAGEMin(data)



    return gt_mol_frac, Mgi, Mni, Fei, Cai




def extract_end_member_mol_frac(phase, MAGEMinOutput, end_member):
    """
    Extracts end member data from MAGEMinOutput based on phase and end member.
    Args:
        phase (str): The phase to extract data for.
        MAGEMinOutput (MAGEMinOutput): The MAGEMinOutput object containing the data.
        endMember (str): The end member to extract data for.
    Returns:
        float: The molar fraction of the specified end member for the given phase. If the specified
        phase or end member is not found, returns 0.0.
    """
    try:
        phase_ind = MAGEMinOutput.ph.index(phase)
        em_index = MAGEMinOutput.SS_vec[phase_ind].emNames.index(end_member)
        data = MAGEMinOutput.SS_vec[phase_ind].emFrac[em_index]
    except:
        data = 0.
    return data

def extract_end_member_wt_frac(MAGEMinOutput, phase, end_member):
    """
    Extracts the weight fraction of a specified end member for a given phase from the input MAGEMinOutput.

    Args:
        MAGEMinOutput: The input MAGEMinOutput object.
        phase: The phase for which the end member weight fraction is to be extracted.
        end_member: The end member for which the weight fraction is to be extracted.

    Returns:
        The weight fraction of the specified end member for the given phase. If the specified
        phase or end member is not found, returns 0.0.
    """

    try:
        phase_ind = MAGEMinOutput.ph.index(phase)
        em_index = MAGEMinOutput.SS_vec[phase_ind].emNames.index(end_member)
        data = MAGEMinOutput.SS_vec[phase_ind].emFrac_wt[em_index]
    except:
        data = 0.
    return data


def extract_phase_mol_frac(phase, MAGEMinOutput):
    """
    Extracts the phase mol from the MAGEMinOutput.

    Args:
        phase: The phase to extract.
        MAGEMinOutput: The MAGEMinOutput object containing phase information.

    Returns:
        The extracted phase molar fraction, or 0. if not found.
    """
    try:
        gt_ind = MAGEMinOutput.ph.index(phase)
        data = (MAGEMinOutput.ph_frac[gt_ind])
    except:
        data = 0.
    return data

def extract_phase_wt_frac(phase, MAGEMinOutput):
    """
    Function to extract phase weight from MAGEMinOutput.
    
    Parameters:
    - phase: the phase to extract weight for
    - MAGEMinOutput: the MAGEMinOutput object
    
    Returns:
    - data: the weight fraction of the specified phase, or 0. if not found
    """
    try:
        gt_ind = MAGEMinOutput.ph.index(phase)
        data = (MAGEMinOutput.ph_frac_wt[gt_ind])
    except:
        data = 0.
    return data
