import os
import logging
from datetime import datetime
import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import ElasticNet
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel as C
from sklearn.ensemble import GradientBoostingRegressor, ExtraTreesRegressor
import xgboost

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

class Regressor():
    def __init__(self,
                 data: pd.DataFrame,  
                 target: pd.DataFrame,
                 test: bool = False,
                 model_path: str = None, 
                 test_ratio: float = 0.2,
                 algo = 'total',
                 loaded_model = None,
                 loaded_scaler = None
        ):
        """
        Train_data, test_data and target dataframe should have 'name' column
        Train_data and target dataframe should have same elements in the 'name' column 
        """
        self.data = data
        self.target = target
        self.test = test
        self.regressor = algo
        self.loaded_model = loaded_model
        self.loaded_scaler = loaded_scaler
        
        #Initialize load_model and load_sclar to False by default
        #Added on Dec. 4. 2024
        self.load_scaler = False
        self.load_model = False
        
        
        if loaded_scaler != None:
          self.load_scaler = True
        if loaded_model != None:
          self.load_model = True
        #if self.test:
        #    #print('Initiate Data Prediction')
        #    assert model_path != None, 'There is no Model path to test!!'
        #    
        #    self.load_model = True
        #    self.model_name = model_path.split('/')[-1] 
        #    
        #    #assert 'model.pkl' in os.listdir(model_path), 'Model does not exist in model path!!'
        #    if 'model.pkl' in os.listdir(model_path):
        #        self.loaded_model = joblib.load(model_path+'/model.pkl')
        #    elif 'model.json' in os.listdir(model_path):
        #        self.loaded_model = xgboost.XGBRegressor()
        #        self.loaded_model.load_model(model_path+'/model.json')
        #    #logger.info(f'Load Model {self.model_name}')

        #    if 'scaler.pkl' in os.listdir(model_path):
        #        #logger.info('Load Scaler')
        #        self.load_scaler = True
        #        self.loaded_scaler = joblib.load(model_path+'/scaler.pkl')
        #    else:
        #        self.load_scaler = False
        
        self.load_model 
        self.test_ratio = test_ratio
        self.save_dir = './model/' + str(datetime.today().strftime("%Y-%m-%d")) +'/'
    
    def data_processing(self, data, target):
        
        data['name'] = [str(i) for i in data['name']]
        target['name'] = [str(i) for i in target['name']]
        
        data = data.sort_values('name')
        target = target.sort_values('name')
        data.reset_index(drop=True, inplace=True)
        target.reset_index(drop=True, inplace=True)
        
        name_x = list(data.iloc[:,0])
        name_y = list(target.loc[:,'name'])
        assert name_x == name_y, 'Name column of Target dataframe and name of atoms do not match'
        
        X = data.iloc[:,1:]
        y = target.loc[:,'target']
        
        if self.test:
            if self.load_scaler:
                scaler = self.loaded_scaler
                X_test = scaler.transform(X)
            else:
                X_test = X
            
            X_train = X
            y_train = y
            y_test = y
            name_train = name_x
            name_test = name_x
            
        else:
            X_train, X_test, y_train, y_test, name_train, name_test = train_test_split(X, y, name_x, test_size=self.test_ratio, random_state=42)

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            
        return X_train, X_test, y_train, y_test, name_train, name_test, scaler
        
    def regression(self, regressor = None):
        if regressor == None:
            regressor = self.regressor
        if not self.test:
            #assert self.target != None, 'Need target.csv in atoms folder!!' 
            
            if regressor == 'GBR':
                logger.info('Start Gradient Boosting Regression')
                reg = GradientBoostingRegressor(n_estimators=3938, learning_rate=0.14777,max_depth=17,
                                             max_features='sqrt',min_samples_leaf=28, min_samples_split=24,
                                             loss='absolute_error',random_state=42)
            elif regressor == 'KRR':
                logger.info('Start Kernel Ridge Regression')
                reg = KernelRidge(kernel = 'rbf')
            elif regressor == 'ELN':
                logger.info('Start ElasticNet Regression')
                reg = ElasticNet(alpha = 0.01)
            elif regressor == 'SVR':
                logger.info('Start Support Vector Regression')
                reg = SVR(kernel = 'rbf')
            elif regressor == 'GPR':
                logger.info('Start Gaussian Process Regression')
                kernel = 1.0 * Matern(length_scale=1.0,nu=2.5)
                reg = GaussianProcessRegressor(kernel=kernel,n_restarts_optimizer=10)
            elif regressor == 'ETR':
                logger.info('Start Gaussian Process Regression')
                reg =ExtraTreesRegressor( bootstrap=False, max_features=0.7500000000000001, 
                                   min_samples_leaf=2, min_samples_split=2, n_estimators=100)
            elif regressor == 'XGB':
                logger.info('Start XGbooster Regression')
                reg = xgboost.XGBRegressor(n_estimators=1000, learning_rate=0.1, gamma=0, subsample=0.75,
                           colsample_bytree=1, max_depth=11)
                
            else:
                raise RuntimeError(f'{regressor} was not defined')
                
            X_train, X_test, y_train, y_test, name_train, name_test, scaler = self.data_processing(self.data, self.target)
            y_train = np.array(y_train)
            y_test = np.array(y_test)

            reg.fit(X_train,y_train)
            
            y_train_pred = reg.predict(X_train)
            y_pred = reg.predict(X_test)
            
            train_mae = mean_absolute_error(y_train, y_train_pred)
            train_rmse = mean_squared_error(y_train, y_train_pred, squared=False)
            test_mae = mean_absolute_error(y_test, y_pred)
            test_rmse = mean_squared_error(y_test, y_pred, squared=False)
            
            trained_df = pd.DataFrame(columns = ['name','target','pred'])
            trained_df['name'] = name_train
            trained_df['target'] = y_train
            trained_df['pred'] = y_train_pred
            
            tested_df = pd.DataFrame(columns = ['name','target','pred'])
            tested_df['name'] = name_test
            tested_df['target'] = y_test
            tested_df['pred'] = y_pred
            
            
            return [train_mae, train_rmse, test_mae, test_rmse], trained_df, tested_df, reg, scaler
            
        if self.test:
            _, X, _, y, _, name, _ = self.data_processing(self.data, self.target)
            
            reg = self.loaded_model
            y_pred = reg.predict(X)
            
            tested_df = pd.DataFrame(columns = ['name','target','pred'])
            tested_df['name'] = name
            tested_df['target'] = y
            tested_df['pred'] = y_pred
            
            return tested_df
    
    def performance_comparison(self):
        if not self.test:
            gbr_error, gbr_train, gbr_test, gbr, gbr_scaler = self.regression('GBR')
            krr_error, krr_train, krr_test, krr, krr_scaler = self.regression('KRR')
            eln_error, eln_train, eln_test, eln, eln_scaler = self.regression('ELN')
            svr_error, svr_train, svr_test, svr, svr_scaler = self.regression('SVR')
            gpr_error, gpr_train, gpr_test, gpr, gpr_scaler = self.regression('GPR')
            xgb_error, xgb_train, xgb_test, xgb, xgb_scaler = self.regression('XGB')
            
            perf_dict = {gbr_error[2] : ['Gradient Boosting Regressor', gbr_error, gbr_train, gbr_test, gbr, gbr_scaler],
                         krr_error[2] : ['Kernel Ridge Regressor', krr_error, krr_train, krr_test, krr, krr_scaler],
                         eln_error[2] : ['ElasticNet Regressor', eln_error, eln_train, eln_test, eln, eln_scaler],
                         svr_error[2] : ['Support Vector Regressor', svr_error, svr_train, svr_test, svr, svr_scaler],
                         gpr_error[2] : ['Gaussian Process Regressor', gpr_error, gpr_train, gpr_test, gpr, gpr_scaler],
                         xgb_error[2] : ['XGBooster Regressor', xgb_error, xgb_train, xgb_test, xgb, xgb_scaler]}
            
            best_model = perf_dict[min(gbr_error[2], krr_error[2], eln_error[2], svr_error[2], gpr_error[2], xgb_error[2])]
            
            print('=====================================================')
            print(' ')
            print(f'Best performance model : {best_model[0]}')
            print(f'MAE : {best_model[1][2]}')
            print(f'RMSE : {best_model[1][3]}')
            print(' ')
            print('=====================================================')
            
            createFolder(self.save_dir)
            
            if best_model[0] == 'XGBooster Regressor':
                best_model[4].save_model(save_dir+'model.json')
            else:
                joblib.dump(best_model[4], self.save_dir+'model.pkl')
            joblib.dump(best_model[5], self.save_dir+'scaler.pkl')
            
            
            
            return best_model[1], best_model[2], best_model[3]
            
            
def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)      
