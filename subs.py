import argparse
import pandas as pd
import os
from datetime import datetime
import joblib
import tqdm 
from ast import literal_eval
from itertools import combinations, product

from dstar.substitution import AtomAlter

def parse_args():
    parser =argparse.ArgumentParser(description='DSTAR Subsitution')
    
    parser.add_argument('--data-path', type=str, default='./data/fp.csv',
                        help='figerprint data path that will be subsituted, (default: ./data/fp.csv/')
    parser.add_argument('--subs-path', type=str, default='./subs/fp/',
                        help='path to save substituted fingerprint data, (default: ./subs.fp/')
    parser.add_argument('--subs-type', type=str, default='comb',
                        help='Susbsitution type, comb = combinations of elemental set A, prod = product elemental set A and B, (default: comb)')
    parser.add_argument('--bi-only', action='store_true', default=False,
                        help='execpt unary materials during subsitution')
    parser.add_argument('--get-bi', action='store_true', default=False,
                        help='extract binary materials in original fingerprint as .csv')
    args = parser.parse_args()
    
    return args

def subs(args):
    
    el_set_A =['Ag','Al','As','Au','Co','Cr','Cu','Fe','Ga','Ge','In','Ir','Mn','Mo',
                 'Ni','Os','Pb','Pd','Pt','Re','Rh','Ru','Sb','Se','Si','Sn','Ti','V',
                 'W','Zn']
    el_set_B = []
    
    data_path = args.data_path
    fp = pd.read_csv(data_path)
    
    fp_path = '/'.join(data_path.split('/')[:-1])
    fp_name = data_path.split('/')[-1].split('.')[0]
    
    if args.subs_type == 'comb':
        assert len(el_set_A) != 0, 'Need at least one element in atom set'
        el_set_B = el_set_A.copy()
        if args.bi_only:
            atom_set = [i for i in list(combination(el_set_A,el_set_B))]
        
        else:
            atom_set = [sorted([j for j in i]) for i in list(product(el_set_A,el_set_B))]
            atom_set = [literal_eval(j) for j in set(str(i) for i in atom_set)]
            
    elif args.subs_type == 'prod':
        assert len(atom_set_A) != 0, 'Need at least one element in atom set'
        
        atom_set = [i for i in list(product(el_set_A,el_set_B))]
    
    else:
        raise KeyError('subs_type should be comb or prod')
        
    aa = AtomAlter()
    binary_fp = aa.get_binary(fp)
    
    if args.get_bi:
        
        print(f'Save Binary Materials Fingeprint at {fp_path} ...') 
        binary_fp.to_csv(fp_path + '/' + fp_name+'_binary.csv')
        print(f'Save Success')
        print('')
        
    general_fp = aa.generalizer(binary_fp)
    
    subs_path = args.subs_path
    createFolder(subs_path)
    
    print('Initiate Active Motif Subsititution')
    for el_set in tqdm.tqdm(atom_set):
        el_name = '_'.join([el_set[0],el_set[1]]) 
        subs_fp = aa.substitution(general_fp,el_set)
        subs_fp.to_csv(subs_path+'/'+el_name+'.csv',index=None)
    print('Successfully Generate New Active Motifs!!')
    
def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)            
    
def main():
    args = parse_args()
    print(args)
    print('')
    
    subs(args)

if __name__ == "__main__":
    main()
