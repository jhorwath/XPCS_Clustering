# XPCS Clustering


[![DOI](https://zenodo.org/badge/574663358.svg)](https://zenodo.org/badge/latestdoi/574663358)


This repository contains code and data needed to reporduce the analysis in our paper: *Elucidation of Relaxation Dynamics Beyond Equilibrium Through AI-informed X-ray Photon Correlation Spectroscopy* by J. P. Horwath, *et al*, available [here](https://arxiv.org/abs/2212.03984).

We have included scripts for building/augmenting a dataset of XPCS two time correlation functions, training a convolutional autoencoder to encode and reproduce XPCS data, and analysis scripts for unsupervised clustering in the latent space.

Most data files needed to run training and analysis are available in the `analysis_data` directory, 
however large files must be downloaded from our [storage 
server](https://anl.app.box.com/s/dhqahh467gnv0srz0tct1ymaofgr07te), and files should be placed in 
the `analysis_data` directory.

The `AutoEncoder` directory contains data augmentation, training, and inference scripts along with pretrained model weights.  The `XPCSData` directory contains the raw data from the rheoXPCS experiment, and the `analysis_data` directory contains metadata (time and scattering direction), and the encoded XPCS dataset.

All analysis presented in the paper and supplemental information can be reproduced using the `Analysis.ipynb` notebook, while additional investigation into our choice of clustering algorithm and number of clusters is found in `ClusteringEvaluation`.

See `requirements.txt` for python package dependencies.  This code has been tested on MacOS Ventura 13.4.1.