
# coding: utf-8

# In[86]:


#performs frequency encoding
#converting _encd to _freq
#compare to a base model
#should use the same Model.csv files as in v2 
import pandas as pd
import numpy as np
import sys

# In[87]:


#data
dataset = pd.read_csv('/home/kate/data/ClaimPrediction/fdata_v1_encd.csv', index_col=None)
target_column = 'hasclaim'


# In[88]:


#models and analyzing results directory
ModelsDir=sys.argv[1]


# In[89]:


#comparing model metrics with t-test
#and save results
import scipy.stats as stats
def AnalyzeAndSaveModelsResults(result_df,result_lst,ModelName,filename):
    df=pd.DataFrame([result_lst])
    TestSizeColumns=['S0.45','S0.4','S0.35','S0.3','S0.25','S0.2','S0.15','S0.1']
    #TestSizeColumns=['S0.2','S0.15','S0.1']
    df.columns=['Model']+TestSizeColumns
    #mean
    df['Mean'] = df.drop('Model', axis=1).mean(axis=1)
    df['t-pvalue'] = 1
    df['t-statistic'] = 0
    df['Group'] = 1
    #t-test with base model
    if ModelName!='BaseModel':
        base_model=result_df[result_df['Model'] == 'BaseModel'].iloc[0]
        current_model=df.iloc[0]
        t=stats.ttest_ind(base_model[TestSizeColumns].tolist(),current_model[TestSizeColumns].tolist())
        line_to_save=[current_model['Model']]
        line_to_save.extend(current_model[TestSizeColumns].tolist())
        line_to_save.append(current_model[TestSizeColumns].mean())
        line_to_save.append(t.pvalue)
        line_to_save.append(t.statistic)
        if ((t.pvalue<=0.05) and (base_model['Mean']<current_model['Mean'])):
            line_to_save.append(2)
        elif ((t.pvalue<=0.05) and (base_model['Mean']>current_model['Mean'])):
            line_to_save.append(3)    
        else:
            line_to_save.append(1)                  
        df_to_save=pd.DataFrame([line_to_save])
        df_to_save.columns=['Model']+TestSizeColumns+['Mean','t-pvalue','t-statistic','Group']
        result_df=result_df.append(df_to_save, ignore_index=True)
    else:
        result_df=result_df.append(df, ignore_index=True)
    result_df.to_csv(ModelsDir+filename, index = False)
    return result_df


# In[90]:


# This function performs frequency encoding
# and return encoded column for train dataset
def freq_encoding(col_name, train_col):
    col_name_freq=col_name.replace('_encd','')+'_freq'
    freq=train_col.value_counts()
    freq=pd.DataFrame(freq)
    freq.reset_index(inplace=True)
    freq.columns=[[col_name,col_name_freq]]
    
    return freq


# In[91]:


#xgb library and parameters to tune later
import xgboost as xgb
#Adjusted Optimized

xgb_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'silent': True,
        'booster': 'gbtree',
        'seed': 42,
        'scale_pos_weight':0.3,
        'colsample_bylevel': 0.232094506,
        'colsample_bytree': 0.978684648,
        'eta': 0.01208041,
        'max_depth': 4}


# In[92]:


#Evaluation metric to be used in tuning
from sklearn.metrics import roc_auc_score
def gini(y, pred):
    g = np.asarray(np.c_[y, pred, np.arange(len(y)) ], dtype=np.float)
    g = g[np.lexsort((g[:,2], -1*g[:,1]))]
    gs = g[:,0].cumsum().sum() / g[:,0].sum()
    gs -= (len(y) + 1) / 2.
    return gs / len(y)
def gini_xgb(pred, y):
    y = y.get_label()
    return 'gini', gini(y, pred) / gini(y, y)


# In[93]:





# In[94]:


#StratifiedKFold
from sklearn.model_selection import StratifiedKFold
nrounds=5000 # need to change to 2000
kfold = 10  # need to change to 10
skf = StratifiedKFold(n_splits=kfold, random_state=42)


# In[95]:


#splitting to train/test in the loop below
from sklearn.model_selection import train_test_split


# In[96]:


#each line in the file contains the model name and set of features to analize
models = pd.read_csv(ModelsDir+'Models.csv', index_col=None)


# In[97]:


#summary for test and train metrics for each model to test overfitting
models_test_gini_df=pd.DataFrame()
models_test_roc_auc_df=pd.DataFrame()
#
models_train_gini_df=pd.DataFrame()
models_train_roc_auc_df=pd.DataFrame()


# In[98]:


for index, row in models.iterrows():
    #for test and train metrics for each model to test overfitting
    gini_test_lst=[]
    roc_auc_test_lst=[]
    gini_train_lst=[]
    roc_auc_train_lst=[]
    #Starting analyzing metric
    print (index, ': Analyzing model %s'%row['Model'])
    gini_test_lst.append(row['Model'])
    roc_auc_test_lst.append(row['Model'])
    gini_train_lst.append(row['Model'])
    roc_auc_train_lst.append(row['Model'])
    featureset=row[1:51].tolist()
    featureset=[x for x in featureset if str(x) != 'nan']
    #calculating metrics for the current featureset and 
    #several data sizes
    for s in (0.45,0.4,0.35,0.3,0.25,0.2,0.15,0.1):
    #for s in (0.2,0.15,0.1):
        print ('Test size %s'%s)
        X, X_test, y, y_test = train_test_split(dataset.loc[:,featureset], dataset[target_column], test_size=s, random_state=42)
        #prediction dataframes
        y_pred_test=pd.DataFrame(index=y_test.index)
        y_pred_test[target_column]=0
        #
        X_test_origin=X_test.copy(deep=True)
        #Stratified Fold
        for i, (train_index, test_index) in enumerate(skf.split(X, y)):
            print(' xgb kfold: {}  of  {} : '.format(i+1, kfold))
            #getting fold data
            X_train, X_valid = X.iloc[train_index,:].copy(), X.iloc[test_index,:].copy()
            #adding frequency encoding to train set
            #for each train, test and valid part _encd columns
            X_test=X_test_origin.copy(deep=True)
            for c in X_train.columns[X_train.columns.to_series().str.contains('_encd')]:
                freq_df=freq_encoding(c, X_train[c])
                X_train=pd.merge(X_train, freq_df, how='left', on=c)
                X_train.drop(c, axis=1, inplace=True)
                #valid
                X_valid=pd.merge(X_valid, freq_df, how='left', on=c)
                # if certain levels in the valid dataset is not observed in the train dataset, 
                # we assign frequency of zero to them
                X_valid.fillna(0, inplace=True)
                X_valid[c.replace('_encd','')+'_freq']=X_valid[c.replace('_encd','')+'_freq'].astype(np.int32)
                X_valid.drop(c, axis=1, inplace=True)
                #test
                X_test=pd.merge(X_test, freq_df, how='left', on=c)
                # if certain levels in the test dataset is not observed in the train dataset, 
                # we assign frequency of zero to them
                X_test.fillna(0, inplace=True)
                X_test[c.replace('_encd','')+'_freq']=X_test[c.replace('_encd','')+'_freq'].astype(np.int32)
                X_test.drop(c, axis=1, inplace=True)
            #
            X_train = X_train.values
            X_valid = X_valid.values
            #
            y_train, y_valid = y.iloc[train_index], y.iloc[test_index]
            #
            y_pred_train=pd.DataFrame(index=y_train.index)
            y_pred_train[target_column]=0
            #
            y_train = y_train.values
            y_valid = y_valid.values

            #applying XGB
            d_train = xgb.DMatrix(X_train, y_train) 
            d_valid = xgb.DMatrix(X_valid, y_valid) 
            watchlist = [(d_train, 'train'), (d_valid, 'valid')]
            xgb_model = xgb.train(xgb_params, d_train, nrounds, watchlist, early_stopping_rounds=100, 
                          feval=gini_xgb, maximize=True, verbose_eval=1000)
            y_pred_test[target_column] +=  xgb_model.predict(xgb.DMatrix(X_test.values), ntree_limit=xgb_model.best_ntree_limit+50) / (kfold)
            y_pred_train[target_column] += xgb_model.predict(xgb.DMatrix(X_train), ntree_limit=xgb_model.best_ntree_limit+50) / (kfold)
        #Prediction results
        #test
        g=gini(y_test,y_pred_test)/gini(y_test,y_test)
        print('Test Gini - %f'%g)
        gini_test_lst.append(g)
        ROC_AUC=roc_auc_score(y_test, y_pred_test)
        print('Test ROC_AUC - %f'%ROC_AUC)
        roc_auc_test_lst.append(ROC_AUC)
        #train
        g=gini(y_train,y_pred_train)/gini(y_train,y_train)
        print('Train Gini - %f'%g)
        gini_train_lst.append(g)
        ROC_AUC=roc_auc_score(y_train, y_pred_train)
        print('Train ROC_AUC - %f'%ROC_AUC)
        roc_auc_train_lst.append(ROC_AUC)
    #save model analysis results
    models_test_gini_df=AnalyzeAndSaveModelsResults(models_test_gini_df,gini_test_lst,row['Model'],'models_test_gini.csv')
    models_test_roc_auc_df=AnalyzeAndSaveModelsResults(models_test_roc_auc_df,roc_auc_test_lst,row['Model'],'models_test_roc_auc.csv')
    models_train_gini_df=AnalyzeAndSaveModelsResults(models_train_gini_df,gini_train_lst,row['Model'],'models_train_gini.csv')
    models_train_roc_auc_df=AnalyzeAndSaveModelsResults(models_train_roc_auc_df,roc_auc_train_lst,row['Model'],'models_train_roc_auc.csv')



