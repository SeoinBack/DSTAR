from .active_motif import get_element_dict
from .descriptor import *
from tqdm import notebook

def get_unique_active_motif(atoms):
    """ Get active motif of atom
    Args:
        atoms (ase.atoms)
    Return:
        list : [active site dictionary, same layer dictionary, sub layer dictionary]
    """    
    dict,c= get_element_dict(atoms)
    elements = []
    for dic in dict:
        for key in dic.keys():
            if (key != 'Empty') & (key not in elements):
                elements.append(key)
    if len(elements) == 2:
        a = elements[0]
        b = elements[1]
    if len(elements) == 1:
        a = elements[0]
    new_dict = []
    for dic in dict:
        new_dic ={}
        try:
            new_dic['A'] = dic[a]
        except:
            pass
        try:
            new_dic['B'] = dic[b]
        except:
            pass
        try:
            new_dic['Empty'] = dic['Empty']
        except:
            pass
        new_dict.append(new_dic)
        
    return new_dict
    
def active_motif_to_fingerprint(elemental_combinations,configurations):
    """ get active motif consisted with input element combination 
    Args:
        elemental_combinations (list) : [Element A, Element B], e.g. ['Cu','Al']
        configurations (list) : list from get_unique_active_motif function
    Return:
        list 1 : properties of corresponing element 
        list 2 : error 
    """    
    docs = []
    a = elemental_combinations[0]
    b = elemental_combinations[1]
    for config in configurations:
        dict = []
        if {} in config:
            continue
        for confi in config:
            dic ={}
            try:
                dic[a] = confi['A']
            except:
                pass
            try:
                dic[b] = confi['B']
            except:
                pass
            if list(confi.keys())[0] == 'Empty':
                dic['Empty'] = 1
            dict.append(dic)
        docs.append(dict)
    
    atomic_number = [[],[],[]]
    block = [[],[],[]]
    average_ionic_radius = [[],[],[]]
    common_oxidation_states = [[],[],[]]
    Pauling_electronegativity= [[],[],[]]
    row = [[],[],[]]
    group = [[],[],[]]
    thermal_conductivity = [[],[],[]]
    boiling_point = [[],[],[]]
    melting_point = [[],[],[]]
    IE = [[],[],[]]
    natom = [[],[],[]]    
    error = []
    name=[]
    
    for i in notebook.tqdm(docs):
        doc = i
        try:
            active_property = tight_mean_descriptor(doc[0])
            floor_property = mean_descriptor(doc[1])
            down_property = mean_descriptor(doc[2])
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
            name.append(i)
        except:
            error.append(i)
    properties = [atomic_number,block,average_ionic_radius,common_oxidation_states,Pauling_electronegativity,row,group,thermal_conductivity,
                   boiling_point,melting_point,IE,natom]
  
    return properties, error