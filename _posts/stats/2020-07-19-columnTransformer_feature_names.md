---
layout: post
category: blog
title:  "Extracting Column Names from the ColumnTransformer"
date:   2020-07-19
tags:
  - scikit-learn
  - tutorial
  - python
  - data processing
  - ML pipeline
  - model interpretation
---

*scikit-learn's `ColumnTransformer` is a great tool for data preprocessing but returns a numpy array without column names. Its method `get_feature_names()` fails if at least one transformer does not create new columns. Here's a quick solution to return column names that works for all transformers and pipelines*

## Extracting Column Names from the ColumnTransformer

The following quick and dirty helper function is built around the `get_feature_names()` method of the `ColumnTransformer`, which can be found here: https://github.com/scikit-learn/scikit-learn/blob/fd237278e895b42abe8d8d09105cbb82dc2cbba7/sklearn/compose/_column_transformer.py#L345

The function walks through the steps of the `ColumnTransformer` and returns the input column names when the transformer does not provide a `get_feature_names()` method. For pipelines, it walks through the pipeline steps and will return either the output columns of the pipeline or the input columns, if the pipeline creates no new columns. 


```python
import warnings
import sklearn
import pandas as pd
```


```python
def get_feature_names(column_transformer):
    """Get feature names from all transformers.
    Returns
    -------
    feature_names : list of strings
        Names of the features produced by transform.
    """
    # Remove the internal helper function
    #check_is_fitted(column_transformer)
    
    # Turn loopkup into function for better handling with pipeline later
    def get_names(trans):
        # >> Original get_feature_names() method
        if trans == 'drop' or (
                hasattr(column, '__len__') and not len(column)):
            return []
        if trans == 'passthrough':
            if hasattr(column_transformer, '_df_columns'):
                if ((not isinstance(column, slice))
                        and all(isinstance(col, str) for col in column)):
                    return column
                else:
                    return column_transformer._df_columns[column]
            else:
                indices = np.arange(column_transformer._n_features)
                return ['x%d' % i for i in indices[column]]
        if not hasattr(trans, 'get_feature_names'):
        # >>> Change: Return input column names if no method avaiable
            # Turn error into a warning
            warnings.warn("Transformer %s (type %s) does not "
                                 "provide get_feature_names. "
                                 "Will return input column names if available"
                                 % (str(name), type(trans).__name__))
            # For transformers without a get_features_names method, use the input
            # names to the column transformer
            if column is None:
                return []
            else:
                return [name + "__" + f for f in column]

        return [name + "__" + f for f in trans.get_feature_names()]
    
    ### Start of processing
    feature_names = []
    
    # Allow transformers to be pipelines. Pipeline steps are named differently, so preprocessing is needed
    if type(column_transformer) == sklearn.pipeline.Pipeline:
        l_transformers = [(name, trans, None, None) for step, name, trans in column_transformer._iter()]
    else:
        # For column transformers, follow the original method
        l_transformers = list(column_transformer._iter(fitted=True))
    
    
    for name, trans, column, _ in l_transformers: 
        if type(trans) == sklearn.pipeline.Pipeline:
            # Recursive call on pipeline
            _names = get_feature_names(trans)
            # if pipeline has no transformer that returns names
            if len(_names)==0:
                _names = [name + "__" + f for f in column]
            feature_names.extend(_names)
        else:
            feature_names.extend(get_names(trans))
    
    return feature_names
```

### Example

This example is based on the sklearn tutorial on the `ColumnTransformer` with different data types: https://scikit-learn.org/stable/auto_examples/compose/plot_column_transformer_mixed_types.html#sphx-glr-auto-examples-compose-plot-column-transformer-mixed-types-py

We would like to structure preprocessing efficiently in a `ColumnTransformer` for different data types and using pipelines of transformers for each type. To interprete the model results, we would like to preserve the variable names in the transformed data. 


```python
from sklearn.datasets import fetch_openml

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
```


```python
X, y = fetch_openml("titanic", version=1, as_frame=True, return_X_y=True)
```

Create a complex data preprocessing pipeline using `ColumnTransformer` and pipelines of transformers and dropping some variables. 


```python
numeric_features = ['age', 'fare']
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

categorical_features = ['embarked', 'sex', 'pclass']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

```


```python
clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', LogisticRegression())])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf.fit(X_train, y_train)
print("model score: %.3f" % clf.score(X_test, y_test))
```

    model score: 0.775



```python
get_feature_names(preprocessor)
```

    <ipython-input-170-7a1be6e049c5>:27: UserWarning: Transformer imputer (type SimpleImputer) does not provide get_feature_names. Will return input column names if available
      warnings.warn("Transformer %s (type %s) does not "
    <ipython-input-170-7a1be6e049c5>:27: UserWarning: Transformer scaler (type StandardScaler) does not provide get_feature_names. Will return input column names if available
      warnings.warn("Transformer %s (type %s) does not "





    ['num__age',
     'num__fare',
     'onehot__x0_C',
     'onehot__x0_Q',
     'onehot__x0_S',
     'onehot__x0_missing',
     'onehot__x1_female',
     'onehot__x1_male',
     'onehot__x2_1.0',
     'onehot__x2_2.0',
     'onehot__x2_3.0']



The unhelpful variable names in the one-hot encoded variables are an issue with the names returned by the `OneHotEncoder`.

We need the variable names to understand the model structure.


```python
clf.named_steps['classifier'].coef_
```




    array([[-0.48401448,  0.0064347 ,  0.23762479, -0.15954077, -0.34818517,
             0.27042239,  1.25211668, -1.25179543,  1.01259174,  0.05134565,
            -1.06361614]])



This is where the feature extractor function comes in handy:


```python
pd.DataFrame(clf.named_steps['classifier'].coef_.flatten(), index=get_feature_names(preprocessor))
```

    <ipython-input-170-7a1be6e049c5>:27: UserWarning: Transformer imputer (type SimpleImputer) does not provide get_feature_names. Will return input column names if available
      warnings.warn("Transformer %s (type %s) does not "
    <ipython-input-170-7a1be6e049c5>:27: UserWarning: Transformer scaler (type StandardScaler) does not provide get_feature_names. Will return input column names if available
      warnings.warn("Transformer %s (type %s) does not "





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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>num__age</th>
      <td>-0.484014</td>
    </tr>
    <tr>
      <th>num__fare</th>
      <td>0.006435</td>
    </tr>
    <tr>
      <th>onehot__x0_C</th>
      <td>0.237625</td>
    </tr>
    <tr>
      <th>onehot__x0_Q</th>
      <td>-0.159541</td>
    </tr>
    <tr>
      <th>onehot__x0_S</th>
      <td>-0.348185</td>
    </tr>
    <tr>
      <th>onehot__x0_missing</th>
      <td>0.270422</td>
    </tr>
    <tr>
      <th>onehot__x1_female</th>
      <td>1.252117</td>
    </tr>
    <tr>
      <th>onehot__x1_male</th>
      <td>-1.251795</td>
    </tr>
    <tr>
      <th>onehot__x2_1.0</th>
      <td>1.012592</td>
    </tr>
    <tr>
      <th>onehot__x2_2.0</th>
      <td>0.051346</td>
    </tr>
    <tr>
      <th>onehot__x2_3.0</th>
      <td>-1.063616</td>
    </tr>
  </tbody>
</table>
</div>




```python

```
