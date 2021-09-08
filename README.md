# FEMDA: Robust classification with Flexible Discriminant Analysis in heterogeneous data
Flexible EM-Inspired Discriminant Analysis is a robust supervised classification algorithm that performs well in noisy and contaminated datasets.

### Authors
Andrew Wang, University of Cambridge, Cambridge, UK
Pierre Houdouin, CentraleSupélec, Paris, France

## Instllation
`pip install -i https://test.pypi.org/simple/ femda`

## Get started
```python
>>> from sklearn.datasets import load_iris
>>> from femda import FEMDA
>>> X, y = load_iris(return_X_y=True)
>>> clf = FEMDA()
>>> clf.fit(X, y)
FEMDA()
>>> clf.score(X, y)
0.9666666666666667
```

Using a specific dataset...
```python
>>> import femda.experiments.preprocessing as pre
>>> X_train, y_train, X_test, y_test = pre.statlog(r"root\datasets\\")
>>> FEMDA().fit(X_train, y_train).score(X_test, y_test)
...
```

Using a `sklearn.pipeline.Pipeline`...

```python
>>> from sklearn.datasets import load_digits
>>> from sklearn.pipeline import make_pipeline
>>> from sklearn.decomposition import PCA
>>> X, y = load_digits(return_X_y=True)
>>> pipe = make_pipeline(PCA(n_components=5), FEMDA()).fit(X, y)
>>> pipe.predict(X)
...
```

## Run all experiments presented in the paper
```python
>>> from femda.experiments import run_experiments()
>>> run_experiments()
...
```

See ![demo.ipynb](demo.ipynb) for more.

## Abstract
Linear and Quadraic Discriminant Analysis are well-known classical methods but suffer heavily from non-Gaussian class distributions and are very non-robust in contaminated datasets. In this paper, we present a new discriminant analysis style classification algorithm that directly models noise and diverse shapes which can deal with a wide range of datasets. 

Each data point is modelled by its own arbitrary Elliptically Symmetrical (ES) distribution and its own arbitrary scale parameter, modelling directly very heterogeneous, non-i.i.d datasets. We show that maximum-likelihood parameter estimation and classification are simple and fast under this model.

We highlight the flexibility of the model to a wide range of Elliptically Symmetrical distribution shapes and varying levels of contamination in synthetic datasets. Then, we show that our algorithm outperforms other robust methods on contaminated datasets from Computer Vision and NLP.