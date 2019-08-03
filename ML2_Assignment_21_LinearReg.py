''' import all necessary modules '''
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston
from sklearn import cross_validation
from sklearn.model_selection import cross_val_score
import seaborn as sns

# Load Dataset and create DataFrame
boston = load_boston()
bos_df = pd.DataFrame(boston.data)
bos_df.columns = boston.feature_names

#Check if any null value present in any column
bos_df.columns[bos_df.isnull().any()].tolist() 
#As its returning empty result, hence we can say there is no null value.

#adding price column to dataframe as target column
bos_df['PRICE']=boston.target

bos_df.columns

'''
Index(['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX',
       'PTRATIO', 'B', 'LSTAT', 'PRICE'],
      dtype='object')
'''

#Now create Corelation Matrix and plot heatmap 
sns.set(style="whitegrid")
# Compute the correlation matrix
corr = bos_df.dropna().corr()
# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(30, 10))
# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, square=True, linewidths=.5, ax=ax)

#### From the Hitmap , RAD and TAX are highly co-related with Price

sns.regplot(y="PRICE", x="CRIM", data=boston_df, fit_reg = True)
sns.regplot(y="PRICE", x="ZN", data=boston_df, fit_reg = True)
sns.regplot(y="PRICE", x="INDUS", data=boston_df, fit_reg = True)
sns.regplot(y="PRICE", x="CHAS", data=boston_df, fit_reg = True)
sns.regplot(y="PRICE", x="NOX", data=boston_df, fit_reg = True)
sns.regplot(y="PRICE", x="RM", data=boston_df, fit_reg = True)
sns.regplot(y="PRICE", x="AGE", data=boston_df, fit_reg = True)
sns.regplot(y="PRICE", x="DIS", data=boston_df, fit_reg = True)
sns.regplot(y="PRICE", x="RAD", data=boston_df, fit_reg = True)
sns.regplot(y="PRICE", x="TAX", data=boston_df, fit_reg = True)
sns.regplot(y="PRICE", x="PTRATIO", data=boston_df, fit_reg = True)
sns.regplot(y="PRICE", x="B", data=boston_df, fit_reg = True)
sns.regplot(y="PRICE", x="LSTAT", data=boston_df, fit_reg = True)
plt.show()

fig, axs = plt.subplots(1, 3, sharey=True)
bos.plot(kind='scatter', x='PTRATIO', y='Price', ax=axs[0], figsize=(16, 8))
bos.plot(kind='scatter', x='LSTAT', y='Price', ax=axs[1])
bos.plot(kind='scatter', x='RM', y='Price', ax=axs[2])

### From the scatter plot, RM, PTRATIO and LSTAT are hightly co-related with Price

# Split data into training and test datasets
X=bos_df.drop('PRICE',axis=1)
y=bos_df['PRICE']

## Create test and train data from datasets with 33% and 77% respectively
x_train, x_test, y_train, y_test = cross_validation.train_test_split(
X,y, test_size=0.33, random_state=5)

##Create Linear regression model using train data
lm = LinearRegression()
lm.fit(x_train, y_train)

## Calculate Slope(Intercept) and Constant(Coefficient)
print ('Estimated intercept :', lm.intercept_)
print ('Coefficients :', lm.coef_)
print ('Number of coefficients:', len(lm.coef_))
#variance score: 1 is perfect prediction
print('Variance score: %.2f' % lm.score(x_test, y_test))

'''
Estimated intercept : 32.858932634086806
Coefficients : [-1.56381297e-01  3.85490972e-02 -2.50629921e-02  7.86439684e-01
 -1.29469121e+01  4.00268857e+00 -1.16023395e-02 -1.36828811e+00
  3.41756915e-01 -1.35148823e-02 -9.88866034e-01  1.20588215e-02
 -4.72644280e-01]
Number of coefficients: 13
Variance score: 0.70
'''

##Predict the value using test data
pred_test = lm.predict(x_test)
# The mean squared error using test data
print("Mean squared error with test data: %.2f"
      % np.mean((pred_test - y_test) ** 2))


##Predict the value using train data
pred_train = lm.predict(x_train)
# The mean squared error using train data
print("Mean squared error with train data: %.2f"
      % np.mean((pred_train - y_train) ** 2))

### As both MSE are almost in a same range , 
### hence we can say that model is fitted with both train data and test data

'''
Mean squared error with test data: 28.54
Mean squared error with train data: 19.55
'''

#Residual plots

plt.scatter(pred_train, pred_train - y_train, c='b', s=40, alpha=0.5)
plt.scatter(pred_test, pred_test - y_test, c='g', s=40)
plt.hlines(y = 0, xmin=0, xmax = 50)
plt.title('Residual Plot using training (blue) and test (green) data')
plt.ylabel('Residuals')
plt.show()

#cross validation using k=4 to check performance of the model
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
#X=boston_df.drop('PRICE',axis=1)
X1=X.values
y1=y.values
# We give cross_val_score a model, the entire data set and its "real" values, and the number of folds:
scores = cross_val_score(lm, X1,y1, cv=10)
# Print the accuracy for each fold:
print ('Accuracies of 10 folds: ' , scores)

# And the mean accuracy of all 10 folds:
print ('Mean accuracy of 10 folds with all columns: ' , scores.mean())

#Now using "RM" "PTRATIO" and "LSTAT" feature(since these features are highly related with price)

lm.fit(X[['PTRATIO','LSTAT','RM']], bos_df.PRICE)
msePTRATIO = np.mean((bos_df.PRICE - lm.predict(X[['PTRATIO','LSTAT','RM']])) ** 2)
print ("Mean squared error with 'PTRATIO','LSTAT','RM': %.2f" %msePTRATIO)

print('Score using these 3 columns:',lm.score(X[['PTRATIO','LSTAT','RM']], bos_df.PRICE))

'''
Accuracies of 10 folds:  [ 0.73334917  0.47229799 -1.01097697  0.64126348  0.54709821  0.73610181
  0.37761817 -0.13026905 -0.78372253  0.41861839]
Mean accuracy of 10 folds with all columns:  0.200137867354187
Mean squared error with 'PTRATIO','LSTAT','RM': 27.13
Score using these 3 columns: 0.6786241601613112
'''

#using cross validation we get

X1=X[['PTRATIO','LSTAT','RM']]
X_val=X1.values
y1=bos_df.PRICE.values

x_train1, x_test1, y_train1, y_test1 = cross_validation.train_test_split(
X1,bos_df.PRICE, test_size=0.33, random_state=5)
lm2 = linear_model.LinearRegression()
# Train the model using the training sets
regrfit= lm2.fit(x_train1, y_train1)
# We give cross_val_score a model, the entire data set and its "real" values, and the number of folds:
scores = cross_val_score(regrfit, X_val,y1, cv=10)
# Print the accuracy for each fold:
print ('Accuracies of 10 folds: ' , scores)

# And the mean accuracy of all 10 folds:
print ('Mean accuracy of 10 folds: ' , scores.mean())
print('Score for these features: ',lm2.score(X1, bos_df.PRICE))

'''
Accuracies of 10 folds:  [ 0.76383016  0.64457435 -0.59161662  0.5585493   0.60444316  0.65850873
  0.05756559  0.05661888 -1.12985209  0.47621583]
Mean accuracy of 10 folds:  0.20988372914637457
Score for these features:  0.6774577145583722
'''
## Now we can see that Mean Accuracy of 10 folds are almost same if we consider full column
## and only ['PTRATIO','LSTAT','RM'] i.e 0.200137867354187 and 0.20988372914637457 respectively

import statsmodels.api as sm
from statsmodels.sandbox.regression.predstd import wls_prediction_std

model1=sm.OLS(y_train,x_train)
result=model1.fit()
result.summary()

# From stats model, we will now consider feature columns where p_values < .05

X_new=bos_df.drop(['PRICE','INDUS','CHAS','NOX','AGE'],axis=1)

#linear fit

lm.fit(X_new, bos_df.PRICE)
msePTRATIO = np.mean((bos_df.PRICE - lm.predict(X_new)) ** 2)
print ("Mean squared error with X_new feature: ",msePTRATIO)
print('Score for these features: ',lm.score(X_new, bos_df.PRICE))

#Mean squared error with X_new feature:  23.32331293516569
#Score for these features:  0.7237214456325678

x_train_2, x_test_2, y_train_2, y_test_2 = cross_validation.train_test_split(
X_new,bos_df.PRICE, test_size=0.33, random_state=5)
regr2 = linear_model.LinearRegression()
# Train the model using the training sets
regrfit2= regr2.fit(x_train_2, y_train_2)
scores2 = cross_val_score(regrfit2, X_new.values,y1, cv=10)
# Print the accuracy for each fold:
print ('Accuracies of 10 folds: ' , scores2)
print ('Mean accuracy of 10 folds: ' , scores2.mean())

'''
Accuracies of 10 folds:  [ 0.73541074  0.57193453 -0.18296454  0.61193365  0.62138919  0.76456959
  0.39308275 -0.06722205 -0.95810462  0.27925426]
Mean accuracy of 10 folds:  0.2769283490414015

'''


## Conclusion

## Here we can see that Mean Accuracy for X_new is higher(.2769) than other feature set. 
## Also score of new features is 0.7237.
## So final model is created with X['CRIM','ZN','RM','DIS','RAD','TAX','PTRATIO','B','LSTAT']
