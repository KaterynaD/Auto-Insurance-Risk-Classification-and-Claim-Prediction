
# coding: utf-8

# In[16]:


#compare to a base model
import pandas as pd
import numpy as np
import sys

#models and analyzing results directory
ModelsDir=sys.argv[1]

# In[17]:


#data
dataset = pd.read_csv('/home/kate/data/ClaimPrediction/fdata_v1_encd.csv', index_col=None)
target_column = 'hasclaim'


# In[18]:





# In[19]:


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


# In[20]:


def add_noise(series, noise_level):
    return series * (1 + noise_level * np.random.randn(len(series)))
def target_encode(trn_series=None,
                  tst_series=None,
                  target=None,
                  fmin_samples_leaf=1.0,
                  smoothing=1,
                  noise_level=0):
    """
    Smoothing is computed like in the following paper by Daniele Micci-Barreca
    https://kaggle2.blob.core.windows.net/forum-message-attachments/225952/7441/high%20cardinality%20categoricals.pdf
    trn_series : training categorical feature as a pd.Series
    tst_series : test categorical feature as a pd.Series
    target : target data as a pd.Series
    fmin_samples_leaf (float) : minimum samples to take category average into account as a fraction of count
    KD: original min_samples_leaf = level count * fmin_samples_leaf
    smoothing (int) : smoothing effect to balance categorical average vs prior
    """
    temp = pd.concat([trn_series, target], axis=1)
    # Compute target mean
    averages = temp.groupby(by=trn_series.name)[target.name].agg(["mean", "count"])
    # Compute smoothing
    # 
    smoothing = 1 / (1 + np.exp(-(averages["count"] - averages["count"]*fmin_samples_leaf) / smoothing))
    # Apply average function to all target data
    prior = target.mean()
    # The bigger the count the less full_avg is taken into account
    averages[target.name] = prior * (1 - smoothing) + averages["mean"] * smoothing
    averages.drop(["mean", "count"], axis=1, inplace=True)
    # Apply averages to trn and tst series
    ft_trn_series = pd.merge(
        trn_series.to_frame(trn_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=trn_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    # pd.merge does not keep the index so restore it
    ft_trn_series.index = trn_series.index
    ft_tst_series = pd.merge(
        tst_series.to_frame(tst_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=tst_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    # pd.merge does not keep the index so restore it
    ft_tst_series.index = tst_series.index
    return add_noise(ft_trn_series, noise_level), add_noise(ft_tst_series, noise_level)


# In[21]:


#xgb library and parameters to tune later
import xgboost as xgb
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


# In[22]:


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


# In[23]:


#StratifiedKFold
from sklearn.model_selection import StratifiedKFold
nrounds=5000 # need to change to 2000
kfold = 10  # need to change to 10
skf = StratifiedKFold(n_splits=kfold, random_state=42)


# In[24]:


#splitting to train/test in the loop below
from sklearn.model_selection import train_test_split


# In[25]:


#each line in the file contains the model name and set of features to analize
models = pd.read_csv(ModelsDir+'Models.csv', index_col=None)


# In[26]:


#names of labels in new bin columns created
group_names = [1,2,3,4,5,6,7,8,9,10]


# In[27]:


#summary for test and train metrics for each model to test overfitting
models_test_gini_df=pd.DataFrame()
models_test_roc_auc_df=pd.DataFrame()
#
models_train_gini_df=pd.DataFrame()
models_train_roc_auc_df=pd.DataFrame()
#
base_model_df=pd.DataFrame()


# In[28]:


for index, row in models.iterrows():
    #for test and train metrics for each model to test overfitting
    gini_test_lst=[]
    roc_auc_test_lst=[]
    gini_train_lst=[]
    roc_auc_train_lst=[]
    #Starting analyzing metric
    print (index, ': Analyzing model %s'%row['Model'])
    #add model name to metric storage
    gini_test_lst.append(row['Model'])
    roc_auc_test_lst.append(row['Model'])
    gini_train_lst.append(row['Model'])
    roc_auc_train_lst.append(row['Model'])
    #getting model parameters
    #first element names of rows in a creating current row dataframe
    featureset=['feature']
    fmin_samples_leaf_set=['fmin_samples_leaf']
    smoothing_set=['smoothing']
    noise_level_set=['noise_level']
    for c in row.index:
        if 'fmin_samples_leaf' in c:
            fmin_samples_leaf_set.append(row[c])
        elif 'smoothing' in c:
            smoothing_set.append(row[c])
        elif 'noise_level' in c:
            noise_level_set.append(row[c])
        elif 'F' in c:
            featureset.append(row[c])
    featureset=[x for x in featureset if str(x) != 'nan']
    fmin_samples_leaf_set=[x for x in fmin_samples_leaf_set if str(x) != 'nan']
    smoothing_set=[x for x in smoothing_set if str(x) != 'nan']
    noise_level_set=[x for x in noise_level_set if str(x) != 'nan']
    #into a dataframe with index as names of rows: fmin_samples_leaf, smoothing, noise_level
    #and columns as feature names
    analyzed_model=pd.DataFrame([fmin_samples_leaf_set,smoothing_set,noise_level_set])
    analyzed_model.columns=featureset #first column is now names of rows
    analyzed_model.set_index('feature', inplace=True)
    #calculating metrics for the current featureset and other parameters and 
    #several data sizes
    for s in (0.45,0.4,0.35,0.3,0.25,0.2,0.15,0.1):
        print ('Test size %s'%s)
        X, X_test, y, y_test = train_test_split(dataset.loc[:,analyzed_model.columns], dataset[target_column], test_size=s, random_state=42)
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
            y_train, y_valid = y.iloc[train_index], y.iloc[test_index]
            #adding targeting encoding 
            #for each train, test and valid part currently analyzing model columns
            #if all parameters are not -1
            X_test=X_test_origin.copy(deep=True)
            for c in analyzed_model.columns:
                fmin_samples_leaf=analyzed_model[c]['fmin_samples_leaf']
                smoothing=analyzed_model[c]['smoothing']
                noise_level=analyzed_model[c]['noise_level']
                if (fmin_samples_leaf>-1.0 and smoothing>-1.0 and noise_level>-1.0):
                    print ('--------------TARGETING ENCODING---------------------------------------')
                    X_train[c.replace('_encd','')+ "_trgenc"], X_test[c.replace('_encd','')+"_trgenc"] = target_encode(
                                         trn_series=X_train[c],
                                         tst_series=X_test[c],
                                         target=y_train,
                                         fmin_samples_leaf=fmin_samples_leaf,
                                         smoothing=smoothing,
                                         noise_level=noise_level)
                    X_train[c.replace('_encd','')+ "_trgenc"], X_valid[c.replace('_encd','')+ "_trgenc"] = target_encode(
                                         trn_series=X_train[c],
                                         tst_series=X_valid[c],
                                         target=y_train,
                                         fmin_samples_leaf=fmin_samples_leaf,
                                         smoothing=smoothing,
                                         noise_level=noise_level)
                    X_train.drop(c, axis=1, inplace=True)
                    X_valid.drop(c, axis=1, inplace=True)
                    X_test.drop(c, axis=1, inplace=True)
                    #binning if there are more then N levels
                    BinsNum=len(group_names)
                    if len(X_train[c.replace('_encd','')+ "_trgenc"].unique())>BinsNum:
                        print('--------------------------------BINNING------------------------------')
                        trgencd_column_name=c.replace('_encd','')+ '_trgenc'
                        #binning data to N levels
                        max_val=max([X_train[trgencd_column_name].max(),X_test[trgencd_column_name].max(),X_valid[trgencd_column_name].max()])
                        min_val=min([X_train[trgencd_column_name].min(),X_test[trgencd_column_name].min(),X_valid[trgencd_column_name].min()])
                        d=(max_val-min_val)/(BinsNum+1)
                        bins=np.linspace(min_val-d, max_val+d, BinsNum+1)
                        bin_column_name=c.replace('_encd','')+ '_trgencbin'
                        X_train[bin_column_name] = pd.cut(X_train[trgencd_column_name], bins, labels=group_names)
                        X_valid[bin_column_name] = pd.cut(X_valid[trgencd_column_name], bins, labels=group_names)
                        X_test[bin_column_name] = pd.cut(X_test[trgencd_column_name], bins, labels=group_names)
                        #deleting target encoded column
                        X_train.drop(trgencd_column_name, axis=1, inplace=True)
                        X_valid.drop(trgencd_column_name, axis=1, inplace=True)
                        X_test.drop(trgencd_column_name, axis=1, inplace=True)
            #
            X_train = X_train.values
            X_valid = X_valid.values
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


