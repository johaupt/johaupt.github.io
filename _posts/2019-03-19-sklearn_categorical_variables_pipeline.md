---
layout: post
title:  "Categorical Variables and ColumnTransformer in scikit-learn"
date:   2019-03-10
categories:
  - scikit-learn
  - tutorial
  - python
  - data processing
  - ML pipeline
---


# Dealing with Categorical Variables in Scikit-learn


```python
import numpy as np
import scipy.stats as sp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import minmax_scale, scale, MinMaxScaler, KBinsDiscretizer
from sklearn.compose import ColumnTransformer

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
```


```python
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, brier_score_loss
```

## Data with mixed data types

Create some data. To create categorical variables, I bin them into an arbitrary number of bins.


```python
no_cont = 5
no_cat = 5
no_vars = no_cont + no_cat
N= 50000

# Create single dataset to avoid random effects
# Only works for all informative features
X,y = make_classification(n_samples=N, weights=[0.9,0.1], n_clusters_per_class=5,
                              n_features=no_vars, 
                              n_informative=no_vars, 
                              n_redundant=0, n_repeated=0,
                             random_state=123)
```


```python
binner = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy="quantile", )
X[:,no_cont:] = binner.fit_transform(X[:,no_cont:])

```


```python
X = pd.DataFrame(X, columns=["X"+str(i) for i in [0,2,4,6,8,1,3,5,7,9]])
```


```python
X[0:5]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>X0</th>
      <th>X2</th>
      <th>X4</th>
      <th>X6</th>
      <th>X8</th>
      <th>X1</th>
      <th>X3</th>
      <th>X5</th>
      <th>X7</th>
      <th>X9</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.921406</td>
      <td>4.480504</td>
      <td>-1.231670</td>
      <td>-1.814375</td>
      <td>4.187405</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.544821</td>
      <td>0.948336</td>
      <td>0.472346</td>
      <td>-1.126138</td>
      <td>2.157616</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.874012</td>
      <td>0.131283</td>
      <td>-3.637079</td>
      <td>0.447905</td>
      <td>-1.041823</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-1.737486</td>
      <td>-1.664507</td>
      <td>-0.084009</td>
      <td>1.294248</td>
      <td>0.492214</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.292494</td>
      <td>0.992953</td>
      <td>1.559877</td>
      <td>-1.070859</td>
      <td>1.391606</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



The raw categorical variables are often not ordinally encoded but contain an ID or hash for each level. 


```python
import string
# Efficiently map values to another value with .map(dict)
X.iloc[:,no_cont:] = X.iloc[:,no_cont:].apply(
    lambda x: x.map({i:letter for i,letter in enumerate(string.ascii_uppercase)})
)

```

The raw data typically also mixes the order of categorical and numeric variables


```python
X.sort_index(axis=1, inplace=True)
```

So this is how the raw data looks like when we receive it from the client!


```python
X[0:5]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>X0</th>
      <th>X1</th>
      <th>X2</th>
      <th>X3</th>
      <th>X4</th>
      <th>X5</th>
      <th>X6</th>
      <th>X7</th>
      <th>X8</th>
      <th>X9</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.921406</td>
      <td>E</td>
      <td>4.480504</td>
      <td>E</td>
      <td>-1.231670</td>
      <td>C</td>
      <td>-1.814375</td>
      <td>B</td>
      <td>4.187405</td>
      <td>B</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.544821</td>
      <td>C</td>
      <td>0.948336</td>
      <td>D</td>
      <td>0.472346</td>
      <td>C</td>
      <td>-1.126138</td>
      <td>C</td>
      <td>2.157616</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.874012</td>
      <td>E</td>
      <td>0.131283</td>
      <td>E</td>
      <td>-3.637079</td>
      <td>D</td>
      <td>0.447905</td>
      <td>D</td>
      <td>-1.041823</td>
      <td>E</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-1.737486</td>
      <td>C</td>
      <td>-1.664507</td>
      <td>B</td>
      <td>-0.084009</td>
      <td>D</td>
      <td>1.294248</td>
      <td>A</td>
      <td>0.492214</td>
      <td>E</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.292494</td>
      <td>D</td>
      <td>0.992953</td>
      <td>A</td>
      <td>1.559877</td>
      <td>D</td>
      <td>-1.070859</td>
      <td>D</td>
      <td>1.391606</td>
      <td>A</td>
    </tr>
  </tbody>
</table>
</div>



## Categorical variables (and pandas)

In pandas, the new way to handle categorical variables as to define their type as 'category' similar to R's factor type.


```python
cat_columns = [1,3,5,7,9]
```


```python
X.iloc[:,cat_columns] = X.iloc[:,cat_columns].astype("category")
```


```python
X.dtypes
```




    X0     float64
    X1    category
    X2     float64
    X3    category
    X4     float64
    X5    category
    X6     float64
    X7    category
    X8     float64
    X9    category
    dtype: object




```python
X[0:5]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>X0</th>
      <th>X1</th>
      <th>X2</th>
      <th>X3</th>
      <th>X4</th>
      <th>X5</th>
      <th>X6</th>
      <th>X7</th>
      <th>X8</th>
      <th>X9</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.921406</td>
      <td>E</td>
      <td>4.480504</td>
      <td>E</td>
      <td>-1.231670</td>
      <td>C</td>
      <td>-1.814375</td>
      <td>B</td>
      <td>4.187405</td>
      <td>B</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.544821</td>
      <td>C</td>
      <td>0.948336</td>
      <td>D</td>
      <td>0.472346</td>
      <td>C</td>
      <td>-1.126138</td>
      <td>C</td>
      <td>2.157616</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.874012</td>
      <td>E</td>
      <td>0.131283</td>
      <td>E</td>
      <td>-3.637079</td>
      <td>D</td>
      <td>0.447905</td>
      <td>D</td>
      <td>-1.041823</td>
      <td>E</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-1.737486</td>
      <td>C</td>
      <td>-1.664507</td>
      <td>B</td>
      <td>-0.084009</td>
      <td>D</td>
      <td>1.294248</td>
      <td>A</td>
      <td>0.492214</td>
      <td>E</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.292494</td>
      <td>D</td>
      <td>0.992953</td>
      <td>A</td>
      <td>1.559877</td>
      <td>D</td>
      <td>-1.070859</td>
      <td>D</td>
      <td>1.391606</td>
      <td>A</td>
    </tr>
  </tbody>
</table>
</div>



Like in R, the levels are now saved as integer values and the mapping from integer value to original level is saved.


```python
X.X9.cat.categories
```




    Index(['A', 'B', 'C', 'D', 'E'], dtype='object')




```python
X.X9.cat.codes[0:5]
```




    0    1
    1    2
    2    4
    3    4
    4    0
    dtype: int8



Sadly, there's no describe() method in place for category variables that could be similar to R's summary() for factor  variables and give, e.g. the counts. 


```python
X.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>X0</th>
      <th>X2</th>
      <th>X4</th>
      <th>X6</th>
      <th>X8</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>50000.000000</td>
      <td>50000.000000</td>
      <td>50000.000000</td>
      <td>50000.000000</td>
      <td>50000.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.070105</td>
      <td>0.077555</td>
      <td>0.492110</td>
      <td>-0.555985</td>
      <td>0.157141</td>
    </tr>
    <tr>
      <th>std</th>
      <td>2.172805</td>
      <td>2.211541</td>
      <td>2.065442</td>
      <td>1.863217</td>
      <td>2.068314</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-9.274609</td>
      <td>-10.270653</td>
      <td>-8.321548</td>
      <td>-8.591301</td>
      <td>-7.706999</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-1.370276</td>
      <td>-1.419127</td>
      <td>-0.894180</td>
      <td>-1.789640</td>
      <td>-1.282510</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.138365</td>
      <td>0.099075</td>
      <td>0.514206</td>
      <td>-0.675296</td>
      <td>0.123451</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1.577229</td>
      <td>1.594109</td>
      <td>1.886729</td>
      <td>0.548743</td>
      <td>1.551513</td>
    </tr>
    <tr>
      <th>max</th>
      <td>8.803489</td>
      <td>9.373858</td>
      <td>9.656678</td>
      <td>9.396778</td>
      <td>9.688887</td>
    </tr>
  </tbody>
</table>
</div>



I'll also split the data as we often do before building models.


```python
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, 
                                                    test_size=0.5, random_state=123)

```

## The column transformer in scikit-learn

Why do we need a column transposer? It applies different transformations to subsets of the data columns. Why is that useful? We like to build a pipeline that does preprocessing (and training) and predicting in one go. When new data comes in, we don't need to look for the values to standardize it, we can just apply the full pipeline.

But what if we have categorical variables in the raw data? We could first one hot encode them, but then we don't want to apply the standardizer to the one hot encoded values! We want to tell the preprocessor to standardize the numeric variables and one hot encode the categorical variables. That's what the ColumnTransformer does.


```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
```

The ColumnTransformer looks like a sklearn pipepline with an additional argument to select the columns for each transformation. 

**Take care here when collecting the column indices automatically. They need to be basic int not numpy types or the sklearn checks within the ColumnTransformer will fail.**


```python
num_columns = [idx for idx in range(X.shape[1]) if idx not in cat_columns]
print(cat_columns, num_columns)
np.all([isinstance(x,int) for x in cat_columns])
```

    [1, 3, 5, 7, 9] [0, 2, 4, 6, 8]





    True



I would like to impute potential missing values in the numeric variables and scale them, so I'll build a pipe to do these transformations. I'll then integrate the pipe into the ColumnTransformer to see how that works.


```python
num_preproc = Pipeline([
    ('cat_imputer', SimpleImputer(strategy='most_frequent')),
    ("scaler", StandardScaler())
])
```


```python
ct = ColumnTransformer([
    # (name, transformer, columns)
    # Transformer for categorical variables
    ("onehot", 
         OneHotEncoder(categories='auto', handle_unknown='ignore', ),  
         cat_columns),
    # Transformer for numeric variables
    ("num_preproc",num_preproc, num_columns)
    ], 
    # what to do with variables not specified in the indices?
    remainder="drop")
```


```python
X_train_transformed = ct.fit_transform(X_train, y)
```

Let's see how that looks!


```python
pd.DataFrame(X_train_transformed)[0:5]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>20</th>
      <th>21</th>
      <th>22</th>
      <th>23</th>
      <th>24</th>
      <th>25</th>
      <th>26</th>
      <th>27</th>
      <th>28</th>
      <th>29</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>-0.349437</td>
      <td>-0.268779</td>
      <td>0.002266</td>
      <td>-1.042649</td>
      <td>-0.827631</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>2.128186</td>
      <td>-0.382543</td>
      <td>-0.827437</td>
      <td>2.914705</td>
      <td>-0.048655</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-0.488686</td>
      <td>-1.275460</td>
      <td>0.335586</td>
      <td>0.154043</td>
      <td>-0.325587</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>-0.069370</td>
      <td>1.443652</td>
      <td>1.264377</td>
      <td>-0.265769</td>
      <td>1.344830</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.736065</td>
      <td>0.806747</td>
      <td>0.381054</td>
      <td>-0.793548</td>
      <td>1.775778</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 30 columns</p>
</div>



For the numeric variables, the column means should be (close to) 0.


```python
X_train_transformed[:,25:].mean(axis=0)
```




    array([-1.09756648e-17, -1.09601217e-17, -2.52770027e-17, -2.64499533e-17,
            1.84696702e-17])



See how the column transformer takes the data apart and pipes it through each transformer separately? **Be careful that the output of the combined transformer is a concatenation of the different transformers. The order of variables may change and the indices that we defined above are no longer correct**. If I'd pass the indices to the classifier or data balancer in the next step, I'd have a bad time. 

Next, I'll integrate this in a modeling pipeline.


```python
from sklearn.linear_model import LogisticRegression
```


```python
pipe = Pipeline([
    ("preprocessing", ct),
    ("classifier", LogisticRegression(C=1, solver='lbfgs'))
])
```


```python
pipe.fit(X_train, y_train)
```




    Pipeline(memory=None,
         steps=[('preprocessing', ColumnTransformer(n_jobs=None, remainder='drop', sparse_threshold=0.3,
             transformer_weights=None,
             transformers=[('onehot', OneHotEncoder(categorical_features=None, categories='auto',
           dtype=<class 'numpy.float64'>, handle_unknown='ignore',
           n_val...enalty='l2', random_state=None, solver='lbfgs',
              tol=0.0001, verbose=0, warm_start=False))])



We can dive into the pipeline to extract the model coefficients. We can see that the 10 raw variables where extended to 30 variables after the 5 categorical variables are one-hot encoded with 5 binary variables each. 


```python
pipe.named_steps.classifier.coef_
```




    array([[ 0.95781925,  0.42040793, -0.03151696, -0.48188476, -0.86403698,
             0.66353437, -0.03752612, -0.11350985, -0.32103785, -0.19067207,
             0.151134  ,  0.13419442,  0.07671957, -0.13323205, -0.22802746,
             0.35972866,  0.00602037, -0.22095139, -0.11843307, -0.02557609,
             0.0825338 ,  0.17822471,  0.19712107, -0.0180598 , -0.4390313 ,
            -0.52420744, -0.53297249, -0.53005165,  0.27383866,  0.10902432]])



The cool part is this: When we get new data, we don't need to worry about the cleaning steps. As long as they are included in the pipeline, they are applied during prediction time.


```python
from sklearn.metrics import roc_auc_score
```


```python
X_test[0:5]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>X0</th>
      <th>X1</th>
      <th>X2</th>
      <th>X3</th>
      <th>X4</th>
      <th>X5</th>
      <th>X6</th>
      <th>X7</th>
      <th>X8</th>
      <th>X9</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>8248</th>
      <td>-1.160837</td>
      <td>D</td>
      <td>2.131385</td>
      <td>C</td>
      <td>0.686961</td>
      <td>D</td>
      <td>-0.880318</td>
      <td>E</td>
      <td>3.303355</td>
      <td>E</td>
    </tr>
    <tr>
      <th>2404</th>
      <td>2.530105</td>
      <td>B</td>
      <td>1.366410</td>
      <td>D</td>
      <td>-2.676441</td>
      <td>A</td>
      <td>-0.179998</td>
      <td>C</td>
      <td>-1.467805</td>
      <td>E</td>
    </tr>
    <tr>
      <th>19796</th>
      <td>-2.832175</td>
      <td>A</td>
      <td>0.291876</td>
      <td>E</td>
      <td>0.232748</td>
      <td>A</td>
      <td>-0.700043</td>
      <td>C</td>
      <td>-1.170993</td>
      <td>C</td>
    </tr>
    <tr>
      <th>4970</th>
      <td>-2.146905</td>
      <td>C</td>
      <td>-1.138431</td>
      <td>B</td>
      <td>1.503194</td>
      <td>C</td>
      <td>-3.747254</td>
      <td>D</td>
      <td>2.213337</td>
      <td>D</td>
    </tr>
    <tr>
      <th>38743</th>
      <td>2.613218</td>
      <td>B</td>
      <td>-3.237024</td>
      <td>E</td>
      <td>2.826875</td>
      <td>D</td>
      <td>-1.015023</td>
      <td>E</td>
      <td>2.429915</td>
      <td>A</td>
    </tr>
  </tbody>
</table>
</div>




```python
pred = pipe.predict_proba(X_test)[:,1]

roc_auc_score(y_test, pred)
```




    0.8065415723608013


