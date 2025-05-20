# %%
import numpy as np

# %%
### Useful functions for bulk rock conversions

''' Molecular weights of oxides and elements '''

ref_ox = ["SiO2", "Al2O3", "CaO", "MgO", "FeO", "Fe2O3", "K2O", "Na2O", "TiO2", "O", "Cr2O3", "MnO", "H2O", "S", "O2"]
ref_molar_mass = [60.08, 101.96, 56.08, 40.30, 71.85, 159.69, 94.2, 61.98, 79.88, 16.0, 151.99, 70.937, 18.015, 32.06, 31.998]

molar_mass_dict = dict(zip(ref_ox, ref_molar_mass))

ref_elements = ["Si", "O", "Al", "Ca", "Mg", "Fe", "K", "Na", "Ti", "Cr", "Mn", "H", "S"]
ref_atomic_mass = [28.085, 15.999, 26.982, 40.078, 24.305, 55.845, 39.0983, 22.98977, 47.867, 51.9961, 54.938, 1.00794, 32.065]

atomic_mass_dict = dict(zip(ref_elements, ref_atomic_mass))


def convert_mol_percent_to_wt_percent(mol_percents, components, mass_dict):
    """
    Generic conversion from mole percent to weight (mass) percent.
    
    Args:
        mol_percents (list): Mole percentages for each component.
        components (list): List of component names (e.g., oxides or elements).
        mass_dict (dict): Dictionary mapping each component to its mass (g/mol or atomic mass).
        
    Returns:
        list: Weight percentages for each component.
    """
    total_mass = 0
    for comp, mol in zip(components, mol_percents):
        total_mass += mol * mass_dict[comp]
    wt_percents = [(mol * mass_dict[comp] / total_mass) * 100 for comp, mol in zip(components, mol_percents)]
    return wt_percents


def convert_wt_percent_to_mol_percent(wt_percents, components, mass_dict):
    """
    Generic conversion from weight (mass) percent to mole percent.
    
    Args:
        wt_percents (list): Weight percentages for each component.
        components (list): List of component names.
        mass_dict (dict): Dictionary mapping each component to its mass (g/mol or atomic mass).
        
    Returns:
        list: Mole percentages for each component.
    """
    total_moles = 0
    for comp, wt in zip(components, wt_percents):
        total_moles += wt / mass_dict[comp]
    mol_percents = [((wt / mass_dict[comp]) / total_moles) * 100 for comp, wt in zip(components, wt_percents)]
    return mol_percents


def convert_wt_percent_to_moles(wt_percents, components, mass_dict, total_weight):
    """
    Convert weight (mass) percentages to moles.
    
    Args:
        wt_percents (list): Weight percentages for each component.
        components (list): List of component names.
        mass_dict (dict): Dictionary mapping each component to its mass.
        total_weight (float): Total weight of the sample.
        
    Returns:
        list: Moles for each component.
    """
    moles = []
    for comp, wt in zip(components, wt_percents):
        # (wt%/100 * total_weight) gives the weight of the component.
        moles.append((wt / 100 * total_weight) / mass_dict[comp])
    return moles


def convert_mol_percent_to_moles(mol_percent_dict, mass_dict, total_mass=100):
    """
    Convert mole percentages (given as a dictionary) to moles for each component.
    
    Args:
        mol_percent_dict (dict): Dictionary where keys are components and values are mole percentages.
        mass_dict (dict): Dictionary mapping each component to its mass.
        total_mass (float): Total mass (default 100 g) implied for the mole percentage conversion.
        
    Returns:
        dict: Dictionary of moles for each component.
    """
    moles = {}
    for comp, mol_percent in mol_percent_dict.items():
        # Assume total_mass is proportional to the sum of the percentages (e.g. 100 g)
        comp_mass = total_mass * (mol_percent / 100)
        moles[comp] = comp_mass / mass_dict[comp]
    return moles


def convert_moles_to_mol_percent(moles, components):
    """
    Convert absolute moles (given as a list or dict) to mole percentages.
    
    Args:
        moles (list or dict): Absolute moles for each component.
            If list, the order must correspond to 'components'.
        components (list): List of component names.
        
    Returns:
        dict: Dictionary of mole percentages for each component.
    """
    if isinstance(moles, list):
        moles_dict = dict(zip(components, moles))
    else:
        moles_dict = moles
        
    total = sum(moles_dict.values())
    return {comp: (moles_dict[comp] / total) * 100 for comp in components}



# %%
def convert_FeOt_to_FeO_Fe2O3(FeOt_wt_percent, FeO_ratio, Fe2O3_ratio, total_mass=1):
    """
    Convert total iron expressed as FeOt to FeO and Fe2O3 based on given ratios.

    :param FeOt_wt_percent: Weight percent of FeOt in the sample.
    :param FeO_ratio: Ratio of FeO in the total iron.
    :param Fe2O3_ratio: Ratio of Fe2O3 in the total iron.
    :param total_mass: Total mass of the sample (default 100g).
    :return: Weight percent of FeO and Fe2O3.
    """
    if FeO_ratio + Fe2O3_ratio != 1:
        raise ValueError("The sum of FeO and Fe2O3 ratios must equal 1.")

    FeOt_mass = (FeOt_wt_percent / 100) * total_mass

    FeO_wt_percent = (FeOt_mass * FeO_ratio) / total_mass * 100
    Fe2O3_wt_percent = (FeOt_mass * Fe2O3_ratio) / total_mass * 100

    return FeO_wt_percent, Fe2O3_wt_percent


# %%
### extract the molar fraction of each element we want to diffuse in the garnet from the end member crystals
def calculate_molar_fractions(endmember_fractions):
    """
    Calculate molar fractions of Mg, Ca, Fe, and Mn in a garnet mixture.

    :param garnet_fractions: Dictionary with fractionation amounts of Py, Alm, Gr, Spss, and Kho.
    :return: Dictionary with molar fractions XMg, XCa, XFe, and XMn.

    ### Example usage:
    garnet_fractions = {'Py': 0.2, 'Alm': 0.3, 'Gr': 0.1, 'Spss': 0.2, 'Kho': 0.2}
    result = calculate_molar_fractions(garnet_fractions)

    ### Based on TC metapelite description:
     E-m    Formula                   Mixing sites
                        X                         Y
                        Mg    Fe    Mn    Ca      Al    Fe3
    py     Mg3Al2Si3O12   3     0     0     0       2     0
    alm    Fe3Al2Si3O12   0     3     0     0       2     0
    spss   Mn3Al2Si3O12   0     0     3     0       2     0
    gr     Ca3Al2Si3O12   0     0     0     3       2     0
    kho    Mg3Fe2Si3O12   3     0     0     0       0     2

    """
    # Initialize molar amounts of Mg, Ca, Fe, and Mn
    molar_amounts = {'Mg': 0, 'Ca': 0, 'Fe': 0, 'Mn': 0}

    # Add contributions from each garnet type
    ### Based on CFMMnASO system
    molar_amounts['Mg'] += endmember_fractions['py'] * 3  # 3 Mg in Pyrope
    molar_amounts['Fe'] += endmember_fractions['alm'] * 3  # 3 Fe in Almandine
    molar_amounts['Ca'] += endmember_fractions['gr'] * 3   # 3 Ca in Grossular
    molar_amounts['Mn'] += endmember_fractions['spss'] * 3  # 3 Mn in Spessartine
    molar_amounts['Mg'] += endmember_fractions['kho'] * 3   # 3 Mg in Khoharite 
    # molar_amounts['Fe'] += endmember_fractions['Kho'] * 2   #  Effects of Fe3 are on the Y mixing site, which can be ignored (???)

    # Calculate total moles of these elements
    total_moles = sum(molar_amounts.values())

    if total_moles == 0.:
        molar_fractions = {element: 0. for element in molar_amounts.keys()}
    else:
        # Calculate molar fractions
        molar_fractions = {element: amount / total_moles for element, amount in molar_amounts.items()}

    return molar_fractions

    
def recalculate_bulk_rock_composition_due_to_fractionation(garnet_fraction, endmember_fractions, initial_composition_oxides):
    """
    Recalculates the bulk rock composition in terms of oxides including the effect on Al2O3 due to garnet fractionation.
    
    Parameters:
    - garnet_fraction: Fraction of garnet fractionated from the rock in mole percent
    - alm, pyr, grss, spss: Proportions of Almandine, Pyrope, Grossular, and Spessartine in the garnet
    - initial_composition_oxides: Dictionary with the initial mole percentages of SiO2, Al2O3, MgO, FeO, MnO, CaO
    
    Returns:
    - new_composition_oxides: Dictionary with the new mole percentages of SiO2, Al2O3, MgO, FeO, MnO, CaO

        # Define the end-member formulas in terms of moles of oxides
    #     Based on TC metapelite description:
    #      E-m    Formula                   Mixing sites
    #                         X                         Y
    #                         Mg    Fe    Mn    Ca      Al    Fe3
    #     py     Mg3Al2Si3O12   3     0     0     0       2     0
    #     alm    Fe3Al2Si3O12   0     3     0     0       2     0
    #     spss   Mn3Al2Si3O12   0     0     3     0       2     0
    #     gr     Ca3Al2Si3O12   0     0     0     3       2     0
    #     kho    Mg3Fe2Si3O12   3     0     0     0       0     2

    """
    
    # Garnet endmember contributions to oxides
    garnet_contribution_to_oxides = {
        'alm': {'FeO': 3, 'Al2O3': 2},
        'py': {'MgO': 3, 'Al2O3': 2},
        'gr': {'CaO': 3, 'Al2O3': 2},
        'spss': {'MnO': 3, 'Al2O3': 2},
        'kho': {'MgO': 3, 'Al2O3': 0}
        # 'kho': {'MgO': 3, 'Fe2O3': 0, 'Al2O3': 0}
    }
    
    # Calculate the total contribution of each oxide by garnet
    total_oxide_contribution = {oxide: 0 for oxide in initial_composition_oxides}
    for endmember, contribution in garnet_contribution_to_oxides.items():
        for oxide, moles in contribution.items():
            fraction = endmember_fractions[endmember]
            total_oxide_contribution[oxide] += moles * fraction * garnet_fraction
            
    # Subtract the garnet contribution from the initial oxide composition
    new_composition_oxides = initial_composition_oxides.copy()
    for oxide, contribution in total_oxide_contribution.items():
        if oxide in new_composition_oxides:
            new_composition_oxides[oxide] -= contribution
            # Ensure no negative values
            new_composition_oxides[oxide] = max(new_composition_oxides[oxide], 0)

    sum_composition_oxides = sum(new_composition_oxides.values())


    normalized_composition_oxides = {oxide: (concentration / sum_composition_oxides) * 100 for oxide, concentration in new_composition_oxides.items()}

    
    return normalized_composition_oxides

