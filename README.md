# FEMDA: Robust classification with Flexible Discriminant Analysis in heterogeneous data
Flexible EM-Inspired Discriminant Analysis is a robust supervised classification algorithm that performs well in noisy and contaminated datasets.
## Abstract
Linear and Quadraic Discriminant Analysis are well-known classical methods but suffer heavily from non-Gaussian class distributions and are very non-robust in contaminated datasets. In this paper, we present a new discriminant analysis style classification algorithm that directly models noise and diverse shapes which can deal with a wide range of datasets. 

Each data point is modelled by its own arbitrary Elliptically Symmetrical (ES) distribution and its own arbitrary scale parameter, modelling directly very heterogeneous, non-i.i.d datasets. We show that maximum-likelihood parameter estimation and classification are simple and fast under this model.

We highlight the flexibility of the model to a wide range of Elliptically Symmetrical distribution shapes and varying levels of contamination in synthetic datasets. Then, we show that our algorithm outperforms other robust methods on contaminated datasets from Computer Vision and NLP.