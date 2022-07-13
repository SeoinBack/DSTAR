import argparse
import pandas as pd
import os
from datetime import datetime
import joblib
import tqdm 

from dstar.regression import Regressor, createFolder
from dstar.fingerprint import surfs_to_df, motifs_to_df

def parse_args():
    parser = argparse.ArgumentParser(description='DSTAR')
    
    parser.add_argument('--test', action='store_true', default=False,
                        help='Test with trained model, Need model path')
    parser.add_argument('--atom-path', type=str, default='./atoms/',
                        help='path with atom to train, default: ./atoms/')
    parser.add_argument('--data-path', type=str, default='./data/fp.csv',
                        help='path to save or load fingerprint data, (default: ./data/fp.csv)')
    parser.add_argument('--load-data', action="store_true", default=False,
                        help='Load already generated fingerprint data')
    parser.add_argument('--convert-only', action="store_true", default=False,
                        help='Just convert atoms to fingerprint, skip train and test')
    parser.add_argument('--test-ratio', type=float, default=0.2,
                        help='ratio of test dataset')
    parser.add_argument('--algo', type=str, default='total',
                        help='Algorithm to train and test,(default: use all regressor and choose best)')
    
    parser.add_argument('--model-path', type=str, default='./model/',
                        help='Model path for --test = True')
    parser.add_argument('--subs-path', type=str, default='./subs/fp/',
                        help='substituted fingerprint path, (defaul: ./subs/fp/{fp.csv})')
    parser.add_argument('--desire-target', type=float, default=-0.67,
                        help='Desire target binding energy for ideal activity, (default: -0.67 for CO2RR)')
    parser.add_argument('--desire-range', type=float, default=0.1,
                        help='Desire range to consider error, (default: +- 0.1)')
    args = parser.parse_args()

    return args

def train(args):
        
    if args.load_data:
        data_path = args.data_path
        assert os.path.exists(data_path), f'{data_path} does not exist!'
        
        motif = pd.read_csv(data_path)
        target = pd.DataFrame(columns = ['name','target'])
        target['name'] = motif['name']
        target['target'] = motif['target']
        
        n = data_path.split('/')[-1]
        print(f'Load Motif From {n}...')
        dataset = motifs_to_df(motif)
        
    if not args.load_data:
        atom_path = args.atom_path
        assert os.path.exists(atom_path), f'{atom_path} does not exist!'
        assert os.path.exists(atom_path+'/target.csv'), f'target.csv does not exist in {atom_path}!'
        
        dataset, motif = surfs_to_df(atom_path)
        target = pd.read_csv(atom_path+'/target.csv')
        
        dataset = dataset.sort_values('name')
        dataset.reset_index(drop=True,inplace=True)
        target = target.sort_values('name')
        target.reset_index(drop=True, inplace=True)
        
        motif['target'] = target['target']
                
        if args.data_path == './data/fp.csv':
            data_path = './data/'
        else:
            data_path = args.data_path
        assert os.path.exists(data_path), f'{data_path} does not exist!'
        
        motif.to_csv(data_path+'/fp.csv',index=None)
    
    if not args.convert_only:
        if args.algo == 'total':
            reg = Regressor(dataset, target, test_ratio = args.test_ratio)
            error, trained_df, tested_df = reg.performance_comparison()

        else:
            reg = Regressor(dataset, target, test_ratio = args.test_ratio)
            error, trained_df, tested_df, reg, scaler = reg.regression(args.algo)
            
            print('=====================================================')
            print(' ')
            print(f'Train MAE : {error[0]}')
            print(f'Train RMSE : {error[1]}')
            print(f'Test MAE : {error[2]}')
            print(f'Test RMSE : {error[3]}')
            print(' ')
            print('=====================================================')
            
            save_dir = '../model/' + str(datetime.today().strftime("%Y-%m-%d")) +'/'
            createFolder(save_dir)
            joblib.dump(reg, save_dir+'model.pkl')
            joblib.dump(scaler, save_dir+'scaler.pkl')
    
        trained_df.to_csv('Train_results.csv',index=None)
        tested_df.to_csv('Test_results.csv',index=None)
    
def test(args):

    data_path = args.subs_path
    model_path = args.model_path
    csv_lst = [i for i in os.listdir(data_path) if i.endswith('csv')]
    assert len(csv_lst) != 0, f'No csv file in {data_path}!!'
    
    model_name = model_path.split('/')[-1]
    if model_name == '':
        model_name = model_path.split('/')[-2]
        
    print(f'Load Model {model_name}')
    print('Initiate Screening...')
    dens_df = pd.DataFrame(columns=['elements','density'])
    elem, dens = [], []
    for csv in tqdm.tqdm(csv_lst):
        elem.append(csv.split('.')[0])
        
        ## motif to fingerprint
        motif = pd.read_csv(data_path+csv)
        target = pd.DataFrame(columns=['name','target'])
        target['name'] = motif['name']
        target['target'] = motif['target']
        
        dataset = motifs_to_df(motif,skip=True)
    
        reg = Regressor(dataset, target, test=True, model_path=model_path)
        tested_df = reg.regression()
        motif['pred'] = tested_df['pred']
        motif.to_csv(data_path+csv)
        
        ## Caculate Ideal Target Density
        ideal = args.desire_target
        err = args.desire_range
        assert err >= 0, 'Ideal Range must be positive'
        
        den = len(motif[(ideal-err <= motif['pred']) & (motif['pred'] <= ideal+err)])/len(motif)
        dens.append(den)
    print('Successfully Predict Ideal Surface Density!!')
    
    dens_df['elements'] = elem
    dens_df['density'] = dens
    dens_df = dens_df.sort_values('density')
    dens_df.reset_index(drop=True, inplace=True)
    dens_df.to_csv('dens.csv',index=None)
    

    
def main():
    args = parse_args()
    print(args)
    print('')
    
    if not args.test:
      train(args)
    if args.test:
      test(args)

if __name__ == "__main__":
    main()
