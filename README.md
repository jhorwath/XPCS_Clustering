# XPCS Clustering

This repository contains code and data needed to reporduce the analysis in our paper: *Elucidation of Relaxation Dynamics Beyond Equilibrium Through AI-informed X-ray Photon Correlation Spectroscopy* by J. P. Horwath, *et al*, available [here](https://arxiv.org/abs/2212.03984).

We have included scripts for building/augmenting a dataset of XPCS two time correlation functions, training a convolutional autoencoder to encode and reproduce XPCS data, and analysis scripts for unsupervised clustering in the latent space.

Most data files needed to run training and analysis are available in the `analysis_data` directory, 
however large files must be downloaded from our [storage 
server](https://anl.app.box.com/s/dhqahh467gnv0srz0tct1ymaofgr07te), and files should be placed in 
the `analysis_data` director.
