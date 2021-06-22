from pymatgen.analysis.local_env import VoronoiNN
from pymatgen.entries.compatibility import Compatibility
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core.periodic_table import Element
from gaspy.mongo import make_atoms_from_doc
from gaspy.atoms_operators import fingerprint_adslab, remove_adsorbate
from ase.io import read,write
import os
import numpy as np
import pymatgen as mg
from ase import Atoms
import ase




def block_to_num(block):
    if block == 's':
        return 1
    elif block == 'p':
        return 2
    elif block == 'd':
        return 3
    elif block == 'f':
        return 4
    
def generalized_mean(x, p, N):
    """Generalized mean function capture the mean value of atomic properties by considering the ratio of each element in the structure.                                                                                     
    Args:                                                                                                  
       x (array): array of atomic properties for each atom in the structure.                                           
       p (int): power parameter, e.g., harmonic mean (-1), geometric mean(0), arithmetic mean(1), quadratic mean(2). 
       N (int): total number of atoms in the structure.
    Returns:                                                                                                 
       float: generalized mean value.                                                                        
    """
    if p != 0:
        D = 1/(N)
        out = (D*sum(x**p))**(1/p)
    else:
        D = 1/(N)
        out = np.exp(D*sum(np.log(x)))
    return out

def get_prop(element,properties):
    """Get First ionization energy from element
    Args:
        element (str) : element e.g. 'H'
        properties (str) : IF, HF,
    Return:
        float: First ionization energy (eV) or Heat of Fusion (kJ/mol)
    """
    elemental_list = ['H','He','Li','Be','B','C','N','O','F','Ne','Na','Mg','Al','Si','P','S','Cl','Ar','K','Ca','Sc','Ti','V','Cr','Mn',
                     'Fe','Co','Ni','Cu','Zn','Ga','Ge','As','Se','Br','Kr','Rb','Sr','Y','Zr','Nb','Mo','Tc','Ru','Rh','Pd','Ag','Cd',
                     'In','Sn','Sb','Te','I','Xe','Cs','Ba','La','Ce','Pr','Nd','Pm','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu',
                     'Hf','Ta','W','Re','Os','Ir','Pt','Au','Hg','Tl','Pb','Bi','Po','At','Rn','Fr','Ra','Ac','Th','Pa','U','Np','Pu','Am','Cm', 
                     'Bk','Cf','Es','Fm','Md','No','Lr','Rf','Db','Sg','Bh','Hs','Mt','Ds','Rg']
    First_ionization_energy = [135984,245874,53917,93227,8298,112603,145341,136181,174228,215645,
                              51391,76462,59858,81517,104867,103600,129679,157596,43407,61132,
                              65615,68281,67462,67665,74340,79024,78810,76398,77264,93942,
                               59993,78894,97886,97524,118138,139996,41771,56949,62173,66339,
                               67589,70924,72800,73605,74589,83369,75762,89938,57864,73439,
                               86084,90096,104513,121298,38939,52117,55769,55387,54730,55250,
                               55820,56437,56704,61501,58638,59389,60215,61077,61834,62542,
                               54259,68251,75496,78640,78335,84382,89670,89587,92255,104375,
                               61082,74167,72856,84170,93000,107485,40727,52784,51700,63067,
                               58900,61941,62657,60262,59738,59915,61979,62817,64200,65000,
                               65800,66500,49000,'N/A','N/A','N/A','N/A','N/A','N/A','N/A','N/A']
    
    Heat_of_Fusion = [0.05868,'N/A',3.00,12.20,50.20,'N/A',0.3604,0.22259,0.2552,0.3317,2.598,8.954,10.790,50.550,0.657,
                      1.7175,3.203,1.188,2.334,8.540,14.10,15.450,20.90,16.90,12.050,13.80,16.190,17.470,13.050,7.322,5.590,36.940,
                      'N/A',6.694,5.286,1.638,2.192,8.30,11.40,16.90,26.40,32.0,24.0,24.0,21.50,17.60,11.30,6.192,3.263,7.029,19.870,
                      17.490,7.824,2.297,2.092,7.750,6.20,5.460,6.890,7.140,'N/A',8.630,9.210,10.050,10.80,11.060,12.20,19.90,16.840,
                      7.660,18.60,24.060,31.60,35.40,33.20,31.80,26.10,19.60,12.550,2.295,4.142,4.799,11.30,'N/A','N/A',2.890,'N/A','N/A',
                      'N/A',16.10,12.30,8.520,5.190,2.840,14.40,15.0,'N/A','N/A','N/A','N/A','N/A','N/A','N/A','N/A','N/A','N/A','N/A',
                      'N/A','N/A','N/A','N/A']
    
    for i, v in enumerate(elemental_list):
        if v == element:
            if properties == 'IF':
                return First_ionization_energy[i]
            elif properties == 'HF':
                return Heat_of_Fusion[i]
            else:
                print('Nothing')

def mean_descriptor(element_dict):
    """ convert elements to elemetal properties
    Args:
        element_dict (dictionary) : element dictionary obtained from get_element_dict function
    Return:
        properties list
    """    
    N = sum(element_dict.values())
    atomic_number_list = []
    average_ionic_radius_list=[]
    common_oxidation_states_list=[]
    Pauling_electronegativity_list = []
    group_list = []
    row_list = []
    thermal_conductivity_list = []
    melting_point_list = []
    boiling_point_list = []
    block_list = []
    IE_list = []
    dict = {}
    for item in element_dict:
        if item != 'Empty':
            natom = N
            ele = mg.Element(item)
            atomic_number = ele.Z
            average_ionic_radius = float(str(ele.average_ionic_radius)[:-4])
            common_oxidation_states = ele.common_oxidation_states[0]
            Pauling_electronegativity = ele.X
            row = ele.row
            group = ele.group
            thermal_conductivity = float(str(ele.thermal_conductivity)[:-12])
            boiling_point = float(str(ele.boiling_point)[: -2])
            melting_point = float(str(ele.melting_point)[: -2])
            block = block_to_num(ele.block)
            IE = get_prop(item,'IF')

            atomic_number_list += [atomic_number]*element_dict[item]
            #atomic_mass_list += [atomic_mass]*element_dict[item]
            average_ionic_radius_list += [average_ionic_radius]*element_dict[item]
            common_oxidation_states_list += [common_oxidation_states]*element_dict[item]
            Pauling_electronegativity_list += [Pauling_electronegativity]*element_dict[item]
            row_list += [row]*element_dict[item]
            group_list += [group]*element_dict[item]
            thermal_conductivity_list += [thermal_conductivity]*element_dict[item]
            boiling_point_list += [boiling_point]*element_dict[item]
            melting_point_list += [melting_point]*element_dict[item]
            block_list += [block]*element_dict[item]
            IE_list += [IE]*element_dict[item]
        elif item == 'Empty':
            natom = 0
            ele = 0
            atomic_number = 0
            average_ionic_radius = 0
            common_oxidation_states = 0
            Pauling_electronegativity = 0
            row = 0
            group = 0
            thermal_conductivity = 0
            boiling_point = 0
            melting_point = 0
            block = 0
            IE = 0

            atomic_number_list += [atomic_number]*element_dict[item]
            average_ionic_radius_list += [average_ionic_radius]*element_dict[item]
            common_oxidation_states_list += [common_oxidation_states]*element_dict[item]
            Pauling_electronegativity_list += [Pauling_electronegativity]*element_dict[item]
            row_list += [row]*element_dict[item]
            group_list += [group]*element_dict[item]
            thermal_conductivity_list += [thermal_conductivity]*element_dict[item]
            boiling_point_list += [boiling_point]*element_dict[item]
            melting_point_list += [melting_point]*element_dict[item]
            block_list += [block]*element_dict[item]
            IE_list += [IE]*element_dict[item]
        
    atomic_number_mean = generalized_mean(np.array(atomic_number_list), 1, N)
    average_ionic_radius_mean = generalized_mean(np.array(average_ionic_radius_list), 1, N)
    common_oxidation_states_mean = generalized_mean(np.array(common_oxidation_states_list), 1, N)
    Pauling_electronegativity_mean = generalized_mean(np.array(Pauling_electronegativity_list), 1, N)
    row_mean =generalized_mean(np.array(row_list), 1, N)
    group_mean =generalized_mean(np.array(group_list), 1, N)
    thermal_conductivity_mean = generalized_mean(np.array(thermal_conductivity_list), 1, N)
    boiling_point_mean = generalized_mean(np.array(boiling_point_list), 1, N)
    melting_point_mean = generalized_mean(np.array(melting_point_list), 1, N)
    block_mean = generalized_mean(np.array(block_list),1,N)
    IE_mean = generalized_mean(np.array(IE_list),1,N)
    

    return [atomic_number_mean]  + [block_mean] + [average_ionic_radius_mean] + [common_oxidation_states_mean] + [Pauling_electronegativity_mean] + [thermal_conductivity_mean] + [group_mean] + [row_mean] + [boiling_point_mean] + [melting_point_mean] + [IE_mean] + [natom]


def tight_mean_descriptor(element_dict):
    """ convert elements to elemetal properties but does not want empty dictionary
    Args:
        element_dict (dictionary) : element dictionary obtained from get_element_dict function
    Return:
        properties list
    """    
    N = sum(element_dict.values())
    atomic_number_list = []
    average_ionic_radius_list=[]
    common_oxidation_states_list=[]
    Pauling_electronegativity_list = []
    group_list = []
    row_list = []
    thermal_conductivity_list = []
    melting_point_list = []
    boiling_point_list = []
    block_list = []
    IE_list = []
    dict = {}
    for item in element_dict:
        natom = N
        ele = mg.Element(item)
        atomic_number = ele.Z
        average_ionic_radius = float(str(ele.average_ionic_radius)[:-4])
        common_oxidation_states = ele.common_oxidation_states[0]
        Pauling_electronegativity = ele.X
        row = ele.row
        group = ele.group
        thermal_conductivity = float(str(ele.thermal_conductivity)[:-12])
        boiling_point = float(str(ele.boiling_point)[: -2])
        melting_point = float(str(ele.melting_point)[: -2])
        block = block_to_num(ele.block)
        IE = get_prop(item,'IF')

        atomic_number_list += [atomic_number]*element_dict[item]

        average_ionic_radius_list += [average_ionic_radius]*element_dict[item]
        common_oxidation_states_list += [common_oxidation_states]*element_dict[item]
        Pauling_electronegativity_list += [Pauling_electronegativity]*element_dict[item]
        row_list += [row]*element_dict[item]
        group_list += [group]*element_dict[item]
        thermal_conductivity_list += [thermal_conductivity]*element_dict[item]
        boiling_point_list += [boiling_point]*element_dict[item]
        melting_point_list += [melting_point]*element_dict[item]
        block_list += [block]*element_dict[item]
        IE_list += [IE]*element_dict[item]

        
    atomic_number_mean = generalized_mean(np.array(atomic_number_list), 1, N)
    average_ionic_radius_mean = generalized_mean(np.array(average_ionic_radius_list), 1, N)
    common_oxidation_states_mean = generalized_mean(np.array(common_oxidation_states_list), 1, N)
    Pauling_electronegativity_mean = generalized_mean(np.array(Pauling_electronegativity_list), 1, N)
    row_mean =generalized_mean(np.array(row_list), 1, N)
    group_mean =generalized_mean(np.array(group_list), 1, N)
    thermal_conductivity_mean = generalized_mean(np.array(thermal_conductivity_list), 1, N)
    boiling_point_mean = generalized_mean(np.array(boiling_point_list), 1, N)
    melting_point_mean = generalized_mean(np.array(melting_point_list), 1, N)
    block_mean = generalized_mean(np.array(block_list),1,N)
    IE_mean = generalized_mean(np.array(IE_list),1,N)
        
    return [atomic_number_mean]  + [block_mean] + [average_ionic_radius_mean] + [common_oxidation_states_mean] + [Pauling_electronegativity_mean] + [thermal_conductivity_mean] + [group_mean] + [row_mean] + [boiling_point_mean] + [melting_point_mean] + [IE_mean] + [natom]
