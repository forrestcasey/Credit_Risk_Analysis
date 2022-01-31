# Credit_Risk_Analysis

## Purpose
Applying machine learning to solve credit card risk challenge. 
In this challenge we use imbalanced-learn and scikit-learn libraries to build and evaluate models using resampling.
Using the credit card dataset from LendingClub, a peer-to-peer lending services company, the data will be oversampled using the RandomOverSampler and SMOTE algorithms, and undersample the data using the ClusterCentroids algorithm. Then, using a combinatorial approach of over- and undersampling using the SMOTEENN algorithm. Then comparing two new machine learning models that reduce bias, BalancedRandomForestClassifier and EasyEnsembleClassifier, to predict credit risk. Lastly an evaluation on the performance of these models and a written recommendation on whether they should be used to predict credit risk is completed.

## Results
### Naive Random Oversampling
![naive_random_oversampling.png](https://github.com/forrestcasey/Credit_Risk_Analysis/blob/main/repo_images/resampling/naive_random_oversampling.png)

- The balanced accuracy score was 0.64, the model predicted the credit risk accurately 64% of the time.
- High_risk precision = 1%, with sensitivity = 66% ,and F1 = 2% . 
- Low_risk precision = 100%, with sensitivity = 62% and F1 = 76%

### SMOTE Oversampling
![smote_oversampling](https://github.com/forrestcasey/Credit_Risk_Analysis/blob/main/repo_images/resampling/smote_oversampling.png)

- The balanced accuracy score was 0.64
- High_risk precision = 1%, with sensitivity = 66% ,and F1 = 2% . 
- Low_risk precision = 100%, with sensitivity = 62% and F1 = 76%


### Undersampling
![undersampling.png](https://github.com/forrestcasey/Credit_Risk_Analysis/blob/main/repo_images/resampling/undersampling.png)

- The balanced accuracy score was 0.64
- High_risk precision = 1%, with sensitivity = 66% ,and F1 = 2% . 
- Low_risk precision = 100%, with sensitivity = 62% and F1 = 76%

### Combo Sampling (SMOTEENN)
![combo_sampling.png](https://github.com/forrestcasey/Credit_Risk_Analysis/blob/main/repo_images/resampling/combo_sampling.png)

- The balanced accuracy score was 0.64
- High_risk precision = 1%, with sensitivity = 66% ,and F1 = 2% . 
- Low_risk precision = 100%, with sensitivity = 62% and F1 = 76%


### Balanced Random Forest Classifier
![bal_random_forest.png](https://github.com/forrestcasey/Credit_Risk_Analysis/blob/main/repo_images/ensemble/bal_random_forest.png)

- The balanced accuracy score was 0.79(rounded)
- High_risk precision = 4%, with sensitivity = 67% ,and F1 = 7% . 
- Low_risk precision = 100%, with sensitivity = 91% and F1 = 95%

### Easy Ensemble AdaBoost Classifier
![easy_ADA_boost.png](https://github.com/forrestcasey/Credit_Risk_Analysis/blob/main/repo_images/ensemble/easy_ADA_boost.png)

- The balanced accuracy score was 0.93
- High_risk precision = 7%, with sensitivity = 91% ,and F1 = 14% . 
- Low_risk precision = 100%, with sensitivity = 94% and F1 = 97%


## Summary
All the models had poor precision scores for the high risk loans.
The Easy Ensemble Classifier & Random Forest Classifier seemed to do better compared to the other moethods. Of the models created, the Easy Ensemble Classifier would be the best model to use to predict credit risk due to the high recall scores for both high and low risk loans, as well as an accuracy score of 93%. The precision for this model is still very off, and so this model could be much improved before being put into use.
