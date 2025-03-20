# %%
import numpy as np

# %%
### Useful functions for bulk rock conversions

''' Molecular weights of oxides and compounds '''

ref_ox = ["SiO2", "Al2O3", "CaO", "MgO", "FeO", "Fe2O3", "K2O", "Na2O", "TiO2", "O", "Cr2O3", "MnO", "H2O", "S", "O2"]
ref_MolarMass = [60.08, 101.96, 56.08, 40.30, 71.85, 159.69, 94.2, 61.98, 79.88, 16.0, 151.99, 70.937, 18.015, 32.06, 31.998]


molar_mass_dict = dict(zip(ref_ox, ref_MolarMass))

def convert_mol_percent_to_wt_percent(mol_percents, ref_ox):
    """
    Convert mol% to wt% for the given oxides based on provided molar percentages and a reference oxide.
    
    Args:
        mol_percents (list): List of molar percentages for each oxide.
        ref_ox (list): List of reference oxides.

    Returns:
        list: List of weight percentages for each oxide.
    """


    # Filter out the molar masses for the provided ref_ox
    filtered_molar_mass_dict = {oxide: molar_mass_dict[oxide] for oxide in ref_ox}

    total_wt = 0
    wt_percents = []

    # Calculate total weight
    for oxide, mol_percent in zip(ref_ox, mol_percents):
        total_wt += mol_percent * filtered_molar_mass_dict[oxide]

    # Convert each mol% to wt%
    for oxide, mol_percent in zip(ref_ox, mol_percents):
        wt_percent = (mol_percent * filtered_molar_mass_dict[oxide] / total_wt) * 100
        wt_percents.append(wt_percent)

    return wt_percents

def convertBulk4MAGEMin(bulk_in, bulk_in_ox, sys_in, db):
    """
    Reproduces the MAGEMin conversion in python.

    Convert the bulk composition to mol% for the specified oxide components using the MAGEMin database.

    Parameters:
    - bulk_in: array-like, shape (n,)
        The composition of the sample.
    - bulk_in_ox: array-like, shape (m,)
        The composition of the sample in terms of oxides.
    - sys_in: str
        The input system unit, either "wt" for weight fraction or "mol" for molar fraction.
    - db: str
        The name of the database to use for the conversion.

    Returns:
    - MAGEMin_bulk: array-like, shape (m,)
        The composition of the sample in mol% for the specified oxides.
    - MAGEMin_ox: list of str
        The list of oxides used for the conversion.
    """


    MAGEMin_ox_map = {
        "ig": ["SiO2", "Al2O3", "CaO", "MgO", "FeO", "K2O", "Na2O", "TiO2", "O", "Cr2O3", "H2O"],
        "igd": ["SiO2", "Al2O3", "CaO", "MgO", "FeO", "K2O", "Na2O", "TiO2", "O", "Cr2O3", "H2O"],
        "ige": ["SiO2", "Al2O3", "CaO", "MgO", "FeO", "K2O", "Na2O", "TiO2", "O", "Cr2O3", "H2O"],
        "alk": ["SiO2", "Al2O3", "CaO", "MgO", "FeO", "K2O", "Na2O", "TiO2", "O", "Cr2O3", "H2O"],
        "mb": ["SiO2", "Al2O3", "CaO", "MgO", "FeO", "K2O", "Na2O", "TiO2", "O", "H2O"],
        "um": ["SiO2", "Al2O3", "MgO", "FeO", "O", "H2O", "S"],
        "mp": ["SiO2", "Al2O3", "CaO", "MgO", "FeO", "K2O", "Na2O", "TiO2", "O", "MnO", "H2O"]
    }

    MAGEMin_ox = MAGEMin_ox_map.get(db, None)
    if MAGEMin_ox is None:
        print("Database not implemented...")
        return None, None

    MAGEMin_bulk = np.zeros(len(MAGEMin_ox))
    bulk = np.zeros(len(MAGEMin_ox))

    # Convert to mol, if system unit = wt
    if sys_in == "wt":
        for i, ox in enumerate(bulk_in_ox):
            if ox in ref_ox:
                id = ref_ox.index(ox)
                bulk[i] = bulk_in[i] / ref_MolarMass[id]
    else:
        bulk = np.copy(bulk_in)


    bulk = bulk / bulk.sum()

    for i, ox in enumerate(MAGEMin_ox):
        if ox in bulk_in_ox:
            id = bulk_in_ox.index(ox)
            MAGEMin_bulk[i] = bulk[id]

    if "Fe2O3" in bulk_in_ox:
        idFe2O3 = bulk_in_ox.index("Fe2O3")
        idFeO = MAGEMin_ox.index("FeO")
        idO = MAGEMin_ox.index("O")

        MAGEMin_bulk[idFeO] += bulk[idFe2O3] * 2.0
        MAGEMin_bulk[idO] += bulk[idFe2O3]

    MAGEMin_bulk = MAGEMin_bulk / MAGEMin_bulk.sum()

    # Check which component can safely be put to 0.0
    indices = range(len(MAGEMin_ox))
    idNonH2O = [i for i in indices if MAGEMin_ox[i] != "H2O"]

    if db in ["ig", "igd", "ige", "alk"]:
        idNonCr2O3 = [i for i in indices if MAGEMin_ox[i] != "Cr2O3"]
        idNonTiO2 = [i for i in indices if MAGEMin_ox[i] != "TiO2"]
        idNonO = [i for i in indices if MAGEMin_ox[i] != "O"]
        c = set(idNonH2O).intersection(idNonCr2O3, idNonTiO2, idNonO)
    elif db == "mb":
        idNonTiO2 = [i for i in indices if MAGEMin_ox[i] != "TiO2"]
        idNonO = [i for i in indices if MAGEMin_ox[i] != "O"]
        c = set(idNonO).intersection(idNonTiO2)
    else:
        c = idNonH2O

    for i in c:
        if MAGEMin_bulk[i] == 0.0:
            MAGEMin_bulk[i] = 1e-4

    MAGEMin_bulk = MAGEMin_bulk / MAGEMin_bulk.sum() * 100.0

    return MAGEMin_bulk, MAGEMin_ox



def convert_wt_percent_to_mol_percent(wt_percents, ref_ox):
    """
    Calculate the mole percentage of each oxide in the given weight percentages 
    based on the provided reference oxides and their molar masses. 

    Args:
    - wt_percents (list): The weight percentages of the oxides.
    - ref_ox (list): The reference oxides for the calculation.

    Returns:
    - list: The mole percentages of the oxides.
    """

    # Create a dictionary of full reference oxides and their molar masses
    # full_molar_mass_dict = dict(zip(ref_ox, ref_MolarMass))

    # Filter out the molar masses for the provided ref_ox
    filtered_molar_mass_dict = {oxide: molar_mass_dict[oxide] for oxide in ref_ox}

    total_moles = 0
    mol_percents = []

    # Calculate total moles
    for oxide, wt_percent in zip(ref_ox, wt_percents):
        total_moles += wt_percent / filtered_molar_mass_dict[oxide]

    # Convert each wt% to mol%
    for oxide, wt_percent in zip(ref_ox, wt_percents):
        mol_percent = (wt_percent / filtered_molar_mass_dict[oxide]) / total_moles * 100
        mol_percents.append(mol_percent)

    return mol_percents



def convert_wt_percent_to_moles(wt_percents, ref_ox, total_weight):
    """
    Convert weight percentages to moles using provided reference oxides and total weight.

    Args:
        wt_percents (list): List of weight percentages for each oxide.
        ref_ox (list): List of reference oxides.
        total_weight (float): Total weight of the compound.

    Returns:
        list: List of moles for each oxide.
    """
        
    # Filter out the molar masses for the provided ref_ox
    filtered_molar_mass_dict = {oxide: molar_mass_dict[oxide] for oxide in ref_ox}

    moles = []

    # Convert each wt% to moles
    for oxide, wt_percent in zip(ref_ox, wt_percents):
        oxide_weight = wt_percent / 100 * total_weight
        mole = oxide_weight / filtered_molar_mass_dict[oxide]
        moles.append(mole)

    return moles

def convert_mol_percent_to_moles(mole_percents, total_mass=100):
    """
    Convert mole percent to moles for each component in a mixture.

    :param mole_percents: Dictionary of mole percent for each component.
    :param molar_masses: Dictionary of molar mass (g/mol) for each component.
    :param total_mass: Total mass of the mixture (default 100g for direct conversion from mole percent).
    :return: Dictionary of moles for each component.
    """
    
    moles = {}
    for component, mol_percent in mole_percents.items():
        if component not in ref_MolarMass:
            raise ValueError(f"Molar mass for {component} is not specified.")
        component_mass = total_mass * (mol_percent / 100)  # Convert mol% to mass assuming 100g total
        moles[component] = component_mass / ref_MolarMass[component]  # Convert mass to moles
    return moles


def convert_moles_to_mol_percent(moles, ref_ox):
    """
    Convert moles to mol% for the given oxides based on provided absolute moles and a reference oxide list.
    
    Args:
        moles (list or dict): Absolute moles for each oxide.
        ref_ox (list): List of reference oxides.

    Returns:
        list: List of mole percentages for each oxide.
    """
    
    # Ensure 'moles' is a dictionary for easier processing
    if isinstance(moles, list):
        moles_dict = dict(zip(ref_ox, moles))
    else:
        moles_dict = moles
    
    # Calculate total moles
    total_moles = sum(moles_dict.values())
    
    # Calculate mole percent for each oxide
    mol_percents = [(moles_dict[oxide] / total_moles) * 100 for oxide in ref_ox]
    
    return mol_percents

# Example usage:
# ref_ox = ["SiO2", "Al2O3", "CaO", "MgO", "FeO", "Fe2O3", "K2O", "Na2O", "TiO2", "O", "Cr2O3", "MnO", "H2O", "S", "O2"]
# moles = [0.1, 0.15, 0.1, 0.05, 0.2, 0.02, 0.08, 0.1, 0.05, 0.01, 0.01, 0.03, 0.02, 0.005, 0.005]  # Example absolute moles for each oxide

# mol_percents = convert_moles_to_mol_percent(moles, ref_ox)
# print("Mole percentages:", mol_percents)



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

# # Example usage
# FeOt_wt_percent = 8.50  # Example weight percent of FeOt
# FeO_ratio = 0.6  # Assuming 60% of FeOt is FeO
# Fe2O3_ratio = 0.4  # Assuming 40% of FeOt is Fe2O3

# FeO_wt_percent, Fe2O3_wt_percent = convert_FeOt_to_FeO_Fe2O3(FeOt_wt_percent, FeO_ratio, Fe2O3_ratio)
# print(f"FeO weight percent: {FeO_wt_percent:.2f}, Fe2O3 weight percent: {Fe2O3_wt_percent:.2f}")



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

