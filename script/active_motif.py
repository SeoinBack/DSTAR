from ase import Atoms
import ase
from pymatgen.analysis.local_env import VoronoiNN
from pymatgen.io.ase import AseAtomsAdaptor
from gaspy.atoms_operators import fingerprint_adslab, remove_adsorbate

def get_active_site(atom):
    """Get active site index                                                                                     
    Args:                                                                                                  
       atom (ase.atoms)                                    
    Returns:                                                                                                 
       dictionary: acitve site index and positon of atom
       atoms: atom with adsorbate replaced to uranium                                                                   
    """
    
    atoms,binding_positions = remove_adsorbate(atom)
    atoms += Atoms('U', positions=[binding_positions[1]])
    uranium_index = atoms.get_chemical_symbols().index('U')
    struc = AseAtomsAdaptor.get_structure(atoms)
    
    voro = VoronoiNN(allow_pathological=True, tol=0.8, cutoff=10)
    nn = voro.get_nn_info(struc,n=uranium_index)
    dict = {}
    active_index = []
    for i in nn:
        if i['site_index'] not in active_index:
            active_index.append(i['site_index'])
    
    dict['active_site_index'] = active_index
    if len(active_index) == 1:
        dict['position'] = 'Top'
    if len(active_index) == 2:
        dict['position'] = 'Bridge'
    if len(active_index) == 3:
        dict['position'] = '3-Hollow'
    if len(active_index) == 4:
        dict['position'] = '4-Hollow'
    if len(active_index) == 5:
        dict['postion'] = '5?'
    return dict, atoms

def get_location(index_a,index_b,atom, distance_base = 0.8):
    """At the structure, Return realtive positon of relative atom for criteria atom
    Args:
        index_a (int) : Criteria atom index of structure
        index_b (int) : Relative atom index of structure
        atom (ase.atoms) : Corresponding atom
        distance_base (int) = 0.5 : Base distance to seperate layer(Angstrom)
    Return:
        str: same or sub
    """
    cell = atom.cell
    pos_a = atom[index_a].position
    pos_b = atom[index_b].position
    a_rel_coor=cell.scaled_positions(pos_a)
    b_rel_coor=cell.scaled_positions(pos_b)

    unit_frac_base =0.5
    unit_z = cell[2][2]
    z_frac_base = distance_base/abs(unit_z)
    if unit_z >= 0:
        z_distance = b_rel_coor[2] - a_rel_coor[2]
    if unit_z <= 0:
        z_distance = a_rel_coor[2] - b_rel_coor[2]
    if (z_distance >= -z_frac_base) : 
        return 'same'
    if (z_distance <= -z_frac_base) :
        return 'sub'
    
def get_layer(atom, index):
    """Seperate layer and get index of each layer atoms
    Args:
        atom (ase.atoms)
        index (int) : Atom index of structure
    Return:
        dictionary
    """
    structure = AseAtomsAdaptor.get_structure(atom)
    voro = VoronoiNN(allow_pathological=False, tol=0.2, cutoff=13)
    active_site_index = index
    active_site_nn = voro.get_nn_info(structure,active_site_index)
    fnn_index = []
    same_index = []
    sub_index = []
    active_elem = structure.species[active_site_index]
    radi = active_elem.atomic_radius
    
    for i in active_site_nn:
        if i['weight'] >= 0.05:
            fnn_index.append(i['site_index'])
    for i in fnn_index:
        if get_location(active_site_index,i,atom,distance_base = radi) == 'same':
            if i not in same_index:
                same_index.append(i)
        if get_location(active_site_index,i,atom,distance_base = radi) == 'sub':
            if i not in sub_index:
                sub_index.append(i)
 
    layer_dict = {}
    location = ['same','sub']
    location_index = [same_index,sub_index]
    for i,v in enumerate(location):
        layer_dict[v] = location_index[i]
    
    return layer_dict

def get_element_dict(atom):
    """Get element and atom index corresponded to each site (FNN, sub SNN, same SNN)
    Args:
        atom (ase.atoms)
    Return:
        list : list of dictionary
    """

    active_dict,atoms = get_active_site(atom)
    atoms = atoms.repeat((3,3,1))
    structure = AseAtomsAdaptor.get_structure(atoms)
    active_index = active_dict['active_site_index']
    active_elem = []
    active_unique = []
    same_index = []
    same_elem = []
    same_unique = []
    sub_index = []
    sub_elem = []
    sub_unique = []
    active_element_dict = {}
    same_element_dict = {}
    sub_element_dict = {}
    uranium_index =len(structure.atomic_numbers)-1

    for active in active_index:
        
        if active != uranium_index :
            active_element = str(structure.species[active])
            active_elem.append(active_element)
            if active_element not in active_unique:
                active_unique.append(active_element)
        
        
        
        dict = get_layer(atoms,active)
        for index in dict['same']:
            if (index != uranium_index) and (index not in active_index) :
                if index not in same_index:
                    same_index.append(index)
                    element = str(structure.species[index])
                    if element != 'U':
                        same_elem.append(element)
                        if element not in same_unique:
                            same_unique.append(element)

    for active in active_index:
        dict = get_layer(atoms,active)
        for index in dict['sub']:
            if (index != uranium_index) and ( index not in active_index) :
                if (index not in sub_index) and (index not in same_index):
                    sub_index.append(index)
                    element = str(structure.species[index])
                    if element != 'U':
                        sub_elem.append(element)
                        if element not in sub_unique:
                            sub_unique.append(element)
                        
    for elem in active_unique:
        counts = active_elem.count(elem)
        active_element_dict[elem] = counts

    for elem in same_unique:
        counts = same_elem.count(elem)
        same_element_dict[elem] = counts

    for elem in sub_unique:
        counts = sub_elem.count(elem)
        sub_element_dict[elem] = counts
    
    dict_list = [active_element_dict, same_element_dict, sub_element_dict]
    for dicts in dict_list:
        if len(dicts) == 0:
            dicts['Empty'] = 1
            
        
    
    return [active_element_dict,same_element_dict,sub_element_dict], [active_index,same_index,sub_index]