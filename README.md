# Auto-Insurance-Risk-Classification-and-Claim-Prediction
This is a multistage project to investigate data, build a machine learning model, incorporate the results in Data Warehouse and create ETLs and dashboards.

For now there are data research and XGB model code and results.


1. fdataset_v1_default_values_analysis.xlsx - numbers of default values per feature and Claim/No Claim performed from first, raw SQL result. This is the first base 
what to include in the next step analysis.
2. Features Summary.xlsx - features complete description in a table format
3. Models Workbook.xlsx - feature selection deep analysis. It consists from the models features and corresponding models metrics (gini) for different train and test sizes.
There are simple (native and external) feature importance comparison as well as simple calculated and frequency encoded features analysis.
4. Targeting Encoded Models Workbook.xlsx - targeting encoded features analysis
5. Parameters Workbook.xlsx - comparing models with different parameters
6. Improving Models Workbook.xlsx - continue of the work started in Parameters Workbook.xlsx This is a final workbook with a base feature set and recommended parameters.
7. Scale Pos Weight  Workbook.xlsx - results of the run with different scale pos weight XGB parameter for unbalanced data set.
8. Feature Engineering Workbook.xlsx - analysis of adding external features, simple calculated features, frequency and targeting encoding in the new selected BaseModel
with new parameters and without balancing.
9. Final Models Feature Importance and Partial Dependency v3.xlsx - final models: XGB parameters, feature importance and partial dependency

- Data Preparation

1.Encoding_fdata_v1.ipynb -  coding categorical, test features to numerical using pd.Categorical(...).codes
2.TargetEncoding_fdata_v1.ipynb - it's recommended to have no more then 5 levels for the statistical research dependency categorical features and HasClaim.
Target encoding is done according to the paper by Daniele Micci-Barreca where we take into account number of claims by category levels. 
The results will be used only in the statistical research (R) so overfitting and noise level is not taken into account.

- R


1. EDAHasClaimDependentFeatures.rmd (html) - statistical research claims dependencies from features.
2. EDACorrelation.rmd (html) - features correlation
3. EDAInformationValue.rmd (html) - information values


- EDA Feature Importance

1. EDA XGB Features Importance v1.ipynb analyzes XGB feature importance with default parameters. 
2. EDA Boruta Features Selection v1.ipynb uses Boruta and RandomForest to rank features. 
3. PCA v1.ipynb principal component analysis, dataset visualization.


- Output

1. XGB_Native_Features_Importance.csv native features importance from EDA XGB Features Importance v1.ipynb
2. XGB_External_Features_Importance.csv external features importance from EDA XGB Features Importance v1.ipynb 
3. Boruta_Native_Features_ranking.csv from EDA Boruta Features Selection v1.ipynb
4. Boruta_External_Features_ranking.csv from EDA Boruta Features Selection v1.ipynb

from R

1. categorical_features_stats.csv - consolidated table with ChiSq test to test significance of the relation and Cramer V test to measure the association from EDAHasClaimDependentFeatures.rmd
2. continuous_features_stats.csv  - consolidated table with Shapiro test to test normality (optinal because the further analysis do not depend on the feature normality) from EDAHasClaimDependentFeatures.rmd
                                    and Wilcox test to test dependence with the target from EDAHasClaimDependentFeatures.rmd
3. fdata_v1_status.csv            - type, number of 0, null, infinity and unique values from EDAHasClaimDependentFeatures.rmd
4. fdata_v1_structure.csv         - type and samples of values from EDAHasClaimDependentFeatures.rmd
5. fdata_v1_summary.csv           - type and Median,Mean, 3rd Qu.,Max. for numerical or levels statistics for categorical from EDAHasClaimDependentFeatures.rmd
6. information_value.csv          - information values from EDAInformationValue.rmd

- Finding Best Model\XGB

There are ipython notebooks and programs to analyze individual features impact on the final model metric.
All of them use:
1. XGB with this working parameters: {'eta': 0.02, 'max_depth': 4, 'subsample': 0.9, 'colsample_bytree': 0.9, 
          'objective': 'binary:logistic', 'eval_metric': 'auc', 'silent': True} which were adjusted later
2. Random Undersampling with default parameters to balance the dataset and changed later to scale_pos_weight=0.3 XGB parameter
3. Stratified 10 folds and 5000 rounds
4. Each tested model (consisted from different features) is run for 8 train/test sizes: (0.45,0.4,0.35,0.3,0.25,0.2,0.15,0.1), then gini and  ROC-AUC are calculated for train 
and test sets. 
5. At the next step, mean of the metrics is calculated and compared to a pre-selected based model with t-test (assuming the metrics are normally distributed).

In this way I can distiguish if the difference between models with a feature (or set of features) is statistically significant.
Train metrics are needed for overfitting control

Noteboks (except "monitor") were used to develop and debug the code. Then they were converted to programs and run in background mode for hours.
Monitor notebooks were used to monitor progress of the background programs displaying models metric comparison in table and graph mode.

Models were described in csv files. Each row is a model.Each column in the row is a feature name, calcualtion between features or some parameters to apply for encoding.

1. Comparing To BaseModel.ipynb and CompareToBaseModel.py - run models with features "as is"
2. Simple calculated feature.ipynb and SimpleCalculatedFeature.py - perform a simple calculation (summary or multiplication of features) coded in models description.
3. Frequency Encoding.ipynb and FrequencyEncoding.py - perform frequency encoding before running models
4. Targeting Encoding.ipynb and TargetingEncoding.py - perform targeting encoding
5. Targeting Encoding with Binning.ipynb - performs targeting encodding and creates bins from targeting encodded columns for high cardinality columns
6. Monitor.ipynb - simple monitors of models metrics

7. Monitor Targeting Encoding-Get Specific Model Results.ipynb and Monitor Targeting Encoding.ipynb - not just monitors metrics but also displays charts to control overfitting
8. XGB with Transformations.ipynb and XGBwithTransformations.py -  read models from different format: 
Model,Feature,Encoding,Noise_Level
BaseModel,DriverAge,,
...
FequencyEncoded,manufacturer_encd,Frequency,
...
TargetEncoded,manufacturer_encd,Target,0.05
each line describes a feature in a model
first column - ModelName, the same for each feature in the model
compare to a base model
apply frequency or target encoding
In general, the format is more flexible but not visible how models are different from each other. I used this code at the end to compare models with differently 
engineered features


9. Models Results Visualization.ipynb reads output of the model research (Gini test and train) and builds charts to compare train vs test results in each model plus 
difference between train and test in BaseModel and each other to compare overfitting.

The results of the programs as well as analyzed models are in Models Workbook.xlsx, Targeting Encoded Models Workbook.xlsx, etc


- ModelParametersOptimization\XGB

BayesianOptimization.ipynb and BayesianOptimization.py (all versions) were used to find best XGB models parameters (brute force approach).
There were several runs with different ranges of parameters. There are several parameters sets with almost identical metric value.

Finally few sets were selected with a close results and compared using Features Importance XGB Models Analysis Comparing To a BaseModel-NewParams.ipynb
Model Results Visualization.ipynb was used to clarify overfitting

Improving Models
It was discovered the new parameters set reveals that some features decrease the metric. A new BaseModel was created from all features models by removing "bad" features.
2 ShortBestModel were selected from  the new BaseModel with statistically the same result. However they are are only 2 features less then the BaseModel. (22 vs 24)
The final parameters set was choosen with slightly less result but also less overfitting.



- Balancing
Features Importance XGB Models Analysis RandomUnderSampler Ratio.ipynb was used to the BaseModel with the best parameters with different undersampler ratios. 
It reveals it is not needed. The maximum metric is achived with ratio=0.93 which is almost the same as the original results without balancing.
The results without balancing (XGBparametersModelsAnalysisComparingToaBaseModelNoBalancing.py) or with scale_pos_weight=0.3
(XGB parameters Models Analysis Comparing To a BaseModel Scale_Pos_Weight.ipynb)
are almost identicall. 




Feature Engineering

Because now I have the new BaseModel, new parameters set and do not need random undersampler balancing I decided to repeat the work I did in Models Workbook 
to analyze impact external features, simple calculated features, frequency and targeting encoding.
The same code developed earlier was used with the new parameters.

Vehicle Length and Width together from external features improves the result.

- Html

Copy of all Python and R notebooks in html format to easy review



- XGBFinalModel

When a final model was ready and external features inportance were clear the next step is to train the model, save files and run to visualize the results (feature importance, partial dependencies).

I removed all low importance features plus discovered several features depend on the method how they were collected. The same probability of a claim for Yes/No and different
only for Unknown values. Most likely it was populated after a claim.

1. XGB Models Prepare Base Model.ipynb train a model with the base feature set and save files.
2. XGB Models Prepare Extended Model.ipynb train a model with the base feature set and 2 external features and save files.
3. XGB Models Run.ipynb reads files with a trained model and predict
4. XGB Models Visualization with Base Model.ipynb reads file, builds feature importance and partial dependency data sets and save them in csv files with addition of distribution train set values
5. XGB Models Visualization Extended Model.ipynb the same as above
6. Utility - Joining_Tbl_PD Base Model.ipynb joins output of feature importanc and partial dependencies files in a way more usuable to build charts in Excel.
7. Utility - Joining_Tbl_PD Extended Model.ipynb the same as above
8. XGB Models Run 2 Models and ROC Curve.ipynb builds ROC Curves for 2 models

- ResearchReport

html and other file types used to describe the results
