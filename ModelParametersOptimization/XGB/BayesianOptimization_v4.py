
# coding: utf-8

# In[134]:


#https://www.kaggle.com/tilii7/bayesian-optimization-of-xgboost-parameters
import pandas as pd
import numpy as np
import warnings
import sys
import gc

# In[135]:


import time
timestr = time.strftime("%Y%m%d-%H%M%S")


# In[136]:




acq=sys.argv[1]
init_points=int(sys.argv[2])
n_iter=int(sys.argv[3])
xi=1e-4
kappa=int(sys.argv[4])
nrounds = 5000
folds = 10


# In[137]:


#log_file
log_file = open('/home/kate/logs/BayesianOptimization/full_log_%s_%s_%s_%s.log'%(acq,init_points,n_iter,kappa),  'w')
log_file_bestparam = open('/home/kate/logs/BayesianOptimization/bestparam_%s_%s_%s_%s.log'%(acq,init_points,n_iter,kappa),  'w')


# In[138]:


#log in csv file
csv_file='/home/kate/logs/BayesianOptimization/AllModel_%s_%s_%s_%s.csv'%(acq,init_points,n_iter,kappa)


# In[139]:


#data
dataset = pd.read_csv('/home/kate/data/ClaimPrediction/fdata_v1_encd.csv', index_col=None)


# In[140]:


#features
target_column = 'hasclaim'
featureset=[
'accidentpreventioncourseind_encd',
'carpoolind_encd',
'classcd_encd',
'driverage',
'drivernumber',
'driverstatuscd_encd',
'drivertrainingind_encd',
'estimatedannualdistance',
'gooddriverind_encd',
'maturedriverind_encd',
'mvrstatus_encd',
'mvrstatusage',
'ratingvalue',
'relationshiptoinsuredcd_encd',
'scholasticdiscountind_encd',
'vehbodytypecd_encd',
'vehicleage',
'vehnumber'
]


# In[141]:


#xgb library and parameters to tune later
import xgboost as xgb
xgb_params = {'eta': 0.02, 'max_depth': 4, 'subsample': 0.9, 'colsample_bytree': 0.9, 
          'objective': 'binary:logistic', 'eval_metric': 'auc', 'silent': True}


# In[142]:


#Random Undersampler to balance the dataset
from imblearn.under_sampling import RandomUnderSampler
rus = RandomUnderSampler(random_state=42)


# In[143]:


#splitting to train/test
from sklearn.model_selection import train_test_split
s=0.25
X_train, X_test, y_train, y_test = train_test_split(dataset.loc[:,featureset], dataset[target_column], test_size=s, random_state=42)
X_train = X_train.values
y_train = y_train.values
#balancing dataset
X_res, y_res = rus.fit_sample(X_train, y_train)
dtrain = xgb.DMatrix(X_res, y_res)
del X_test,  y_test
del X_train, y_train
gc.collect()

# In[144]:


#best metric variables
AUCbest = -1.
ITERbest = 0


# In[145]:


# cv fold for each parameters set
def xgb_evaluate(
                 #max_depth,
                 min_child_weight,
                 colsample_bytree,
                 #subsample,
                 gamma,
                 #colsample_bylevel,
                 #max_delta_step,
                 #eta,
                 reg_alpha
                 #,reg_lambda
         ):

    global AUCbest
    global ITERbest

    params={}
    params['booster'] = 'gbtree'
    params['min_child_weight'] = int(min_child_weight)
    params['cosample_bytree'] = max(min(colsample_bytree, 1), 0)
    params['max_depth'] = 5
    params['subsample'] = 1
    params['gamma'] = gamma
    #params['colsample_bylevel'] = max(min(colsample_bylevel, 1), 0)
    params['colsample_bylevel'] = 0.2
    params['max_delta_step']=8
    params['eta']=0.02
    params['reg_alpha'] = max(reg_alpha, 0)
    params['reg_lambda']=1
    params['eval_metric']='auc'
    params['silent']=True
    params['objective']='binary:logistic'
    params['seed'] =42

    
    

    print("\n Search parameters (%d-fold validation):\n %s" % (folds, params), file=log_file )
    log_file.flush()

    xgbc = xgb.cv(
                    params,
                    dtrain,
                    num_boost_round = nrounds,
                    stratified = True,
                    nfold = folds,
                    early_stopping_rounds = 100,
                    metrics = 'auc',
                    show_stdv = True
               )


    val_score = xgbc['test-auc-mean'].iloc[-1]
    train_score = xgbc['train-auc-mean'].iloc[-1]
    print(' Stopped after %d iterations with train-auc = %f val-auc = %f ( diff = %f ) train-gini = %f val-gini = %f' 
          % ( len(xgbc), train_score, val_score, (train_score - val_score), (train_score*2-1),
(val_score*2-1)) , file=log_file)
    if ( val_score > AUCbest ):
        AUCbest = val_score
        ITERbest = len(xgbc)
        print('\n\nBest Valid AUC changed to %f'%AUCbest, file=log_file)
        log_file.flush()
        #
        print("\n Best parameters (%d-fold validation):\n %s" % (folds, params), file=log_file_bestparam )
        print('\n Best Valid AUC changed to %f'%AUCbest, file=log_file_bestparam)
        print('\n Train AUC is %f'%train_score, file=log_file_bestparam)
        log_file_bestparam.flush()
        #
    del xgbc
    gc.collect()
    return (val_score*2) - 1


# In[146]:


#bayesian optimization
from bayes_opt import BayesianOptimization
#XGB_BO = BayesianOptimization(xgb_evaluate, {   'max_depth': (2, 12),
#                                                'min_child_weight': (0.1, 20),
#                                                'colsample_bytree': (0.2, 1.1),
#                                                'subsample': (0.1, 1.1),
#                                                'gamma': (0.001, 10),
#                                                'colsample_bylevel': (0.2, 1.1),
#                                               'max_delta_step':(0,10),
#                                                'eta':(0.01,1.1),
#                                                'reg_alpha': (0, 10),
#                                                'reg_lambda':(1,10)
#                                                })

#v2
#XGB_BO = BayesianOptimization(xgb_evaluate, {  
#'max_depth': (4, 7),
#'min_child_weight': (11, 16), #0.5
#'colsample_bytree': (0.6, 1.1),
#'subsample': (0.8, 1.1),
#'gamma': (0.11, 7),
#'colsample_bylevel': (0.2, 0.25),
#'max_delta_step':(0,9),
#'eta':(0.01,0.03),
#'reg_alpha': (0, 6),
#'reg_lambda':(1,9)
# })

#v3 
#XGB_BO = BayesianOptimization(xgb_evaluate, {  
#'max_depth': (5, 6),
#'min_child_weight': (11, 14),
#'colsample_bytree': (0.6, 1.03),
#'subsample': (1.08, 1.1),
#'gamma': (0.4,3.5),
##colsample_bylevel: 0.2,
#'max_delta_step':(8,9),
#'eta':(0.017,0.028),
#'reg_alpha': (1.6,3.1),
#'reg_lambda':(1,8.7)
# })
 
#v4
XGB_BO = BayesianOptimization(xgb_evaluate, {  
#'max_depth': (5, 6),
'min_child_weight': (12, 13),
'colsample_bytree': (0.647, 0.737),
#'subsample': (1.08, 1.1),
'gamma': (3.096,3.389),
##colsample_bylevel: 0.2,
#'max_delta_step':(8,9),
#'eta':(0.017,0.028),
'reg_alpha': (1.8,2),
#'reg_lambda':(1,8.7)
 })
 

 
# In[147]:


#run optimization
print('-'*130)
print('-'*130, file=log_file)
log_file.flush()

with warnings.catch_warnings():
    warnings.filterwarnings('ignore')
    if acq in ('ei','poi'):
        XGB_BO.maximize(init_points=init_points, n_iter=n_iter, acq=acq, xi=xi)
    else:
        XGB_BO.maximize(init_points=init_points, n_iter=n_iter, acq=acq, kappa=kappa)

# In[148]:


print('-'*130)
print('Final Results')
print('Maximum XGBOOST value: %f' % XGB_BO.res['max']['max_val'])
print('Best XGBOOST parameters: ', XGB_BO.res['max']['max_params'])
print('-'*130, file=log_file)
print('Final Result:', file=log_file)
print('Maximum XGBOOST value: %f' % XGB_BO.res['max']['max_val'], file=log_file)
print('Best XGBOOST parameters: ', XGB_BO.res['max']['max_params'], file=log_file)
log_file.flush()
log_file.close()


# In[149]:


history_df = pd.DataFrame(XGB_BO.res['all']['params'])
history_df2 = pd.DataFrame(XGB_BO.res['all']['values'])
history_df = pd.concat((history_df, history_df2), axis=1)
history_df.rename(columns = { 0 : 'gini'}, inplace=True)
history_df['AUC'] = ( history_df['gini'] + 1 ) / 2
history_df.to_csv(csv_file)

