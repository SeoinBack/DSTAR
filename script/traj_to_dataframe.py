from .descriptor import *
from .active_motif import *
from ase.io import read,write
import os
from tqdm import notebook
import numpy as np
import pymatgen as mg
from ase import Atoms
import ase
import pandas as pd
from gaspy.mongo import make_atoms_from_doc


def make_docs_to_dataframe(docs, name_list = None, Use_initial = False):
    """ convert docs list to input descriptor
    Args:
        docs (list) : Dictionary/json/Mongo document created by the `make_doc_from_atoms` function.
        name_list (list) : convert atom to descriptor only in name_list
        Use_initial : If True, Use initial structure from docs
    Return:
        dataframe
        error (list) : name list of atom causing error
    """    
    atomic_number = [[],[],[]]
    block = [[],[],[]]
    average_ionic_radius = [[],[],[]]
    common_oxidation_states = [[],[],[]]
    Pauling_electronegativity = [[],[],[]]
    row = [[],[],[]]
    group = [[],[],[]]
    thermal_conductivity = [[],[],[]]
    boiling_point = [[],[],[]]
    melting_point = [[],[],[]]
    IE = [[],[],[]]
    natom = [[],[],[]]
    error = []   
    names=[]
    energies = []
    if name_list == None:
      name_list = []
      for datum in docs:
        name_list.append(str(datum['mpid'])+'_'+str(datum['mongo_id']))
    
    for datum in notebook.tqdm(docs):
        name = str(datum['mpid'])+'_'+str(datum['mongo_id'])
        if name in name_list:
          if Use_initial:
            atoms = make_atoms_from_doc(datum['initial_configuration'])
          else:  
            atoms = make_atoms_from_doc(datum)
          dict,b= get_element_dict(atoms)
          try:
              active_property = tight_mean_descriptor(dict[0])
              floor_property = mean_descriptor(dict[1])
              down_property = mean_descriptor(dict[2])
              prop_list = [active_property,floor_property,down_property]
              for prop, j in zip(prop_list,range(0,3)):
                  atomic_number[j].append(prop[0])
                  block[j].append(prop[1])
                  average_ionic_radius[j].append(prop[2])
                  common_oxidation_states[j].append(prop[3])
                  Pauling_electronegativity[j].append(prop[4])
                  row[j].append(prop[5])
                  group[j].append(prop[6])
                  thermal_conductivity[j].append(prop[7])
                  boiling_point[j].append(prop[8])
                  melting_point[j].append(prop[9])
                  IE[j].append(prop[10])
                  natom[j].append(prop[11])
              names.append(name)
              energies.append(datum['energy'])
          except:
              error.append(name)

    df = pd.DataFrame({'name' : list(names)},columns=['name'])        
    column_name_list = ['active','floor','down']
    for column_name, i in zip(column_name_list,range(0,3)):
        df[column_name+'_number'] = atomic_number[i]
        df[column_name+'_block'] = block[i]
        df[column_name+'_radi'] = average_ionic_radius[i]
        df[column_name+'_oxi'] = common_oxidation_states[i]
        df[column_name+'_X'] = Pauling_electronegativity[i]
        df[column_name+'_row'] = row[i]
        df[column_name+'_group'] = group[i]
        df[column_name+'_thermal'] = thermal_conductivity[i]
        df[column_name+'_bp'] =boiling_point[i]
        df[column_name+'_mp'] = melting_point[i]
        df[column_name+'_IE'] = IE[i] 
        df[column_name+'_natom'] = natom[i]
    df['energy'] = energies
    return df, error
    
def make_atom_to_properties(atom):
    """ convert atom to input descriptor
    Args:
        atom (ase.Atoms)
        name (str) : refers to atom
    Return:
        list : [properties] 
    """    
    atomic_number = [[],[],[]]
    block = [[],[],[]]
    average_ionic_radius = [[],[],[]]
    common_oxidation_states = [[],[],[]]
    Pauling_electronegativity = [[],[],[]]
    row = [[],[],[]]
    group = [[],[],[]]
    thermal_conductivity = [[],[],[]]
    boiling_point = [[],[],[]]
    melting_point = [[],[],[]]
    IE = [[],[],[]]
    natom = [[],[],[]]

    dict,b= get_element_dict(atom)

    active_property = tight_mean_descriptor(dict[0])
    floor_property = mean_descriptor(dict[1])
    down_property = mean_descriptor(dict[2])
    prop_list = [active_property,floor_property,down_property]
    for prop, j in zip(prop_list,range(0,3)):
      atomic_number[j].append(prop[0])
      block[j].append(prop[1])
      average_ionic_radius[j].append(prop[2])
      common_oxidation_states[j].append(prop[3])
      Pauling_electronegativity[j].append(prop[4])
      row[j].append(prop[5])
      group[j].append(prop[6])
      thermal_conductivity[j].append(prop[7])
      boiling_point[j].append(prop[8])
      melting_point[j].append(prop[9])
      IE[j].append(prop[10])
      natom[j].append(prop[11])

    return [atomic_number,block,average_ionic_radius,common_oxidation_states,Pauling_electronegativity,row,group,thermal_conductivity,boiling_point,melting_point,IE,natom]