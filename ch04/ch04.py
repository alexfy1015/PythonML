# -*- coding: utf-8 -*-
"""
Created on Sat Jun 10 22:36:05 2023

@author: alexfy1015
"""

# *Python Machine Learning 2nd Edition* by [Sebastian Raschka](https://sebastianraschka.com), Packt Publishing Ltd. 2017
# 
# Code Repository: https://github.com/rasbt/python-machine-learning-book-2nd-edition
# 
# Code License: [MIT License](https://github.com/rasbt/python-machine-learning-book-2nd-edition/blob/master/LICENSE.txt)

# # Python Machine Learning - Code Examples

# # Chapter 4 - Building Good Training Sets – Data Preprocessing

# Note that the optional watermark extension is a small IPython notebook plugin that I developed to make the code reproducible. You can just skip the following line(s).





# *The use of `watermark` is optional. You can install this IPython extension via "`pip install watermark`". For more information, please see: https://github.com/rasbt/watermark.*


# ### Overview

# - [Dealing with missing data](#Dealing-with-missing-data)
#   - [Identifying missing values in tabular data](#Identifying-missing-values-in-tabular-data)
#   - [Eliminating samples or features with missing values](#Eliminating-samples-or-features-with-missing-values)
#   - [Imputing missing values](#Imputing-missing-values)
#   - [Understanding the scikit-learn estimator API](#Understanding-the-scikit-learn-estimator-API)
# - [Handling categorical data](#Handling-categorical-data)
#   - [Nominal and ordinal features](#Nominal-and-ordinal-features)
#   - [Mapping ordinal features](#Mapping-ordinal-features)
#   - [Encoding class labels](#Encoding-class-labels)
#   - [Performing one-hot encoding on nominal features](#Performing-one-hot-encoding-on-nominal-features)
# - [Partitioning a dataset into a separate training and test set](#Partitioning-a-dataset-into-seperate-training-and-test-sets)
# - [Bringing features onto the same scale](#Bringing-features-onto-the-same-scale)
# - [Selecting meaningful features](#Selecting-meaningful-features)
#   - [L1 and L2 regularization as penalties against model complexity](#L1-and-L2-regularization-as-penalties-against-model-omplexity)
#   - [A geometric interpretation of L2 regularization](#A-geometric-interpretation-of-L2-regularization)
#   - [Sparse solutions with L1 regularization](#Sparse-solutions-with-L1-regularization)
#   - [Sequential feature selection algorithms](#Sequential-feature-selection-algorithms)
# - [Assessing feature importance with Random Forests](#Assessing-feature-importance-with-Random-Forests)
# - [Summary](#Summary)

#%%
# # Dealing with missing data

# ## Identifying missing values in tabular data

import pandas as pd
from io import StringIO

csv_data = \
    '''A,B,C,D
    1.0,2.0,3.0,4.0
    5.0,6.0,,8.0
    10.0,11.0,12.0,'''
df = pd.read_csv(StringIO(csv_data))
df
#%%
df.isnull().sum()

# access the underlying NumPy array
# via the `values` attribute
df.values

# ## Eliminating samples or features with missing values



# remove rows that contain missing values
#drop all rows with NaN
df.dropna(axis=0)

# remove columns that contain missing values
#drop all columns with NaN
df.dropna(axis=1)

# only drop rows where all columns are NaN
# (returns the whole array here since we don't
# have a row with where all values are NaN)
df.dropna(how='all')

# drop rows that have less than 4 real values
df.dropna(thresh=4)

# only drop rows where NaN appear in specific columns (here: 'C')
df.dropna(subset=['C'])

#%%
# impute missing values via the column mean

from sklearn.impute import SimpleImputer
import numpy as np
imr = SimpleImputer(missing_values=np.nan, strategy='mean')
imr = imr.fit(df.values)
imputed_data = imr.transform(df.values)
imputed_data

#%%
# ## Understanding the scikit-learn estimator API










# # Handling categorical data

# ## Nominal and ordinal features

import pandas as pd
df = pd.DataFrame([
            ['green', 'M', 10.1, 'class1'],
            ['red', 'L', 13.5, 'class2'],
            ['blue', 'XL', 15.3, 'class1']])
df.columns = ['color', 'size', 'price', 'classlabel']
df
#%%
# ## Mapping ordinal features

size_mapping = {
                'XL': 3,
                'L': 2,
                'M': 1}
df['size'] = df['size'].map(size_mapping)
df
#%%
inv_size_mapping = {v: k for k, v in size_mapping.items()}
df['size'].map(inv_size_mapping)
#%%

# ## Encoding class labels




# create a mapping dict
# to convert class labels from strings to integers

import numpy as np
class_mapping = {label:idx for idx,label in
                 enumerate(np.unique(df['classlabel']))}
class_mapping
#%%
# to convert class labels from strings to integers

df['classlabel'] = df['classlabel'].map(class_mapping)
df
#%%
# reverse the class label mapping

inv_class_mapping = {v: k for k, v in class_mapping.items()}
df['classlabel'] = df['classlabel'].map(inv_class_mapping)
df
#%%
# Label encoding with sklearn's LabelEncoder

from sklearn.preprocessing import LabelEncoder
class_le = LabelEncoder()
y = class_le.fit_transform(df['classlabel'].values)
y

# reverse mapping
class_le.inverse_transform(y)

#%%

# ## Performing one-hot encoding on nominal features

X = df[['color', 'size', 'price']].values
color_le = LabelEncoder()
X[:, 0] = color_le.fit_transform(X[:, 0])
X

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

ohe = ColumnTransformer([("one_hot_encoder", OneHotEncoder(),[0])], remainder="passthrough") # The last arg ([0]) is the list of columns you want to transform in this step
#OneHotEncoder(categories=[0])
ohe.fit_transform(X)

#%%
# one-hot encoding via pandas
pd.get_dummies(df[['price', 'color', 'size']])

# multicollinearity guard in get_dummies
pd.get_dummies(df[['price', 'color', 'size']],
               drop_first=True) #The drop_first help remove the multicollinearity introduced by onehotencoding

# multicollinearity guard for the OneHotEncoder
ohe.fit_transform(X)[:,1:] #This call can drop the first column in the previous ohe, and hence remove multicollinearity

#%%
# # Partitioning a dataset into a seperate training and test set

df_wine = pd.read_csv('https://archive.ics.uci.edu/'
                      'ml/machine-learning-databases/'
                      'wine/wine.data', header=None)

# if the Wine dataset is temporarily unavailable from the
# UCI machine learning repository, un-comment the following line
# of code to load the dataset from a local path:

# df_wine = pd.read_csv('wine.data', header=None)

df_wine.columns=['Class label', 'Alcohol',
                 'Malic acid', 'Ash',
                 'Alcalinity of ash', 'Magnesium',
                 'Total phenols', 'Flavanoids',
                 'Nonflavanoid phenols',
                 'Proanthocyanins',
                 'Color intensity', 'Hue',
                 'OD280/OD315 of diluted wines',
                 'Proline']
print('Class labels', np.unique(df_wine['Class label']))
df_wine.head()

#%%
from sklearn.model_selection import train_test_split
X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
X_train, X_test, y_train, y_test =\
    train_test_split(X, y,
                     test_size=0.3,
                     random_state=0,
                     stratify=y)

#%%
# # Bringing features onto the same scale

from sklearn.preprocessing import MinMaxScaler
mns = MinMaxScaler()
X_train_norm = mns.fit_transform(X_train)
X_test_norm = mns.transform(X_test)

#%%
# A visual example:
ex = np.array([0, 1, 2, 3, 4, 5])
print('standardized:', (ex - ex.mean()) / ex.std())

# Please note that pandas uses ddof=1 (sample standard deviation) 
# by default, whereas NumPy's std method and the StandardScaler
# uses ddof=0 (population standard deviation)

print('normalized:', (ex - ex.min()) / (ex.max() - ex.min()))

#%%
from sklearn.preprocessing import StandardScaler
stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.transform(X_test)

#%%

# # Selecting meaningful features

# ...

# ## L1 and L2 regularization as penalties against model complexity

# ## A geometric interpretation of L2 regularization









# ## Sparse solutions with L1-regularization





# For regularized models in scikit-learn that support L1 regularization, we can simply set the `penalty` parameter to `'l1'` to obtain a sparse solution:


from sklearn.linear_model import LogisticRegression
LogisticRegression(penalty='l1')

# Applied to the standardized Wine data ...

lr = LogisticRegression(penalty='l1', C=1.0, solver='liblinear')
# Note that C=1.0 is the default. You can increase
# or decrease it to make the regularization effect
# stronger or weaker, respectively.
lr.fit(X_train_std, y_train)
print('Training accuracy:', lr.score(X_train_std, y_train))
print('Test accuracy:', lr.score(X_test_std, y_test))

lr.intercept_

np.set_printoptions(8)

lr.coef_[lr.coef_!=0].shape

lr.coef_

#%%
import matplotlib.pyplot as plt

fig = plt.figure()
ax = plt.subplot(111)

colors = ['blue', 'green', 'red', 'cyan',
          'magenta', 'yellow', 'black',
          'pink', 'lightgreen', 'lightblue',
          'gray', 'indigo', 'orange']
weights, params = [], []
for c in np.arange(-4., 6.):
    lr = LogisticRegression(penalty='l1',
                            C=10.**c,
                            solver='liblinear',
                            random_state=0)
    lr.fit(X_train_std, y_train)
    weights.append(lr.coef_[1])
    params.append(10**c)

weights = np.array(weights)

for column, color in zip(range(weights.shape[1]), colors):
    plt.plot(params, weights[:, column],
             label=df_wine.columns[column + 1],
             color=color)
plt.axhline(0, color='black', linestyle='--', linewidth=3)
plt.xlim([10**(-5), 10**5])
plt.ylabel('weight coefficient')
plt.xlabel('C')
plt.xscale('log')
plt.legend(loc='upper left')
ax.legend(loc='upper center',
          bbox_to_anchor=(1.38, 1.03),
          ncol=1, fancybox=True)
plt.show()

#%%

# ## Sequential feature selection algorithms

from sklearn.base import clone
from itertools import combinations
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

class SBS(): # Sequential Backward Selection
    def __init__(self, estimator, k_features,
                 scoring=accuracy_score,
                 test_size=0.25, random_state=1):
        self.scoring = scoring
        self.estimator = estimator
        self.k_features = k_features
        self.test_size = test_size
        self.random_state = random_state
    
    def fit(self, X, y):
        
        X_train, X_test, y_train, y_test =\
            train_test_split(X, y, test_size=self.test_size,
                             random_state=self.random_state)
        
        dim = X_train.shape[1]
        self.indices_ = tuple(range(dim))
        self.subsets_ = [self.indices_]
        score = self._calc_score(X_train, y_train, 
                                 X_test, y_test, self.indices_)
        self.scores_ = [score]
        
        while dim > self.k_features:
            scores = []
            subsets = []
            
            for p in combinations(self.indices_, r=dim - 1):
                score = self._calc_score(X_train, y_train,
                                         X_test, y_test, p)
                scores.append(score)
                subsets.append(p)
            
            best = np.argmax(scores)
            self.indices_ = subsets[best]
            self.subsets_.append(self.indices_)
            dim -= 1
            
            self.scores_.append(scores[best])
        self.k_score_ = self.scores_[-1]
        
        return self
    
    def transform(self, X):
        return X[:, self.indices_]
    
    def _calc_score(self, X_train, y_train, X_test, y_test, 
                    indices):
        self.estimator.fit(X_train[:, indices], y_train)
        y_pred = self.estimator.predict(X_test[:, indices])
        score = self.scoring(y_test, y_pred)
        return score

#%%

import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5)

# selecting features
sbs = SBS(knn, k_features=1)
sbs.fit(X_train_std, y_train)

#%%
# plotting performance of feature subsets
k_feat = [len(k) for k in sbs.subsets_]

plt.plot(k_feat, sbs.scores_, marker='o')
plt.ylim([0.7, 1.02])
plt.ylabel('Accuracy')
plt.xlabel('Number of features')
plt.grid()
plt.show()

#%%
k3 = list(sbs.subsets_[10])
print(df_wine.columns[1:][k3]) # [1:] is needed to exclude the column of y, 'Class label'

#%%
knn.fit(X_train_std, y_train)
print('Training accuracy:', knn.score(X_train_std, y_train))
print('Test accuracy:', knn.score(X_test_std, y_test))

#%%
knn.fit(X_train_std[:, k3], y_train)
print('Training accuracy:',
      knn.score(X_train_std[:, k3], y_train))
print('Testing accuracy:',
      knn.score(X_test_std[:, k3], y_test))

#%%
# # Assessing feature importance with Random Forests

from sklearn.ensemble import RandomForestClassifier

feat_labels = df_wine.columns[1:]

forest = RandomForestClassifier(n_estimators=500,
                                random_state=1)
forest.fit(X_train, y_train)
importances = forest.feature_importances_

indices = np.argsort(importances)[::-1]

for f in range(X_train.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30,
                            feat_labels[indices[f]],
                            importances[indices[f]]))
plt.title('Feature Importance')
plt.bar(range(X_train.shape[1]),
        importances[indices],
        align='center')

plt.xticks(range(X_train.shape[1]),
           feat_labels[indices], rotation=90)
plt.xlim([-1, X_train.shape[1]])
plt.tight_layout()
plt.show()

#%%

from sklearn.feature_selection import SelectFromModel

sfm = SelectFromModel(forest, threshold=0.1, prefit=True)
X_selected = sfm.transform(X_train)
print('Number of features that meet this threshold criterion:',
      X_selected.shape[1])

# Now, let's print the 5 features that met the threshold criterion for feature selection that we set earlier (note that this code snippet does not appear in the actual book but was added to this notebook later for illustrative purposes):


for f in range(X_selected.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30,
                            feat_labels[indices[f]],
                            importances[indices[f]]))
