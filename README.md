# Introduction
This codebase for disentangled representation learning accompanies the paper [Disentanglement via Latent Quantization](https://arxiv.org/abs/2305.18378) by Kyle Hsu, Will Dorrell, James C. R. Whittington, Jiajun Wu, and Chelsea Finn.

It uses: 
- [JAX](https://github.com/google/jax) and [Equinox](https://github.com/patrick-kidger/equinox) for automatic differentiation
- [Hydra](https://hydra.cc/) for configuration management
- [Weights & Biases](https://wandb.ai/) for experiment logging
- [TensorFlow Datasets](https://www.tensorflow.org/datasets) for dataset management and data loading

The code separates [encoder architecture](./disentangle/encoders), [decoder architecture](./disentangle/decoders), [latent space design](./disentangle/latents), and [model objectives](./disentangle/models) into modular components. 
These are combined via Hydra's [partial object instantiation functionality](https://hydra.cc/docs/advanced/instantiate_objects/overview/#partial-instantiation) via the `*_partial` options in configuration files. See [below](#example) for an example.

# Installation

```
conda create -n disentangle python=3.10 -y && conda activate disentangle
conda install c-compiler cxx-compiler jax cuda-nvcc -c conda-forge -c nvidia -y
git clone --recurse-submodules https://github.com/kylehkhsu/disentangle.git
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/ pip install -r requirements.txt
pip install -e .
```

Alternatives for JAX installation can be found here: https://github.com/google/jax#installation.
For example:
```
pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```


`LD_LIBRARY_PATH` command is to ensure `tensorflow` installation "sees" the `cudatoolkit` and `cudnn` packages installed via `conda`.


[Datasets](./disentangle/datasets) will be installed via the TensorFlow Datasets API on first use.

To use Weights & Biases logging, you may have to create a free account at [wandb.ai](https://wandb.ai/). Then, run `wandb login` and enter the API key from your account.

# Usage
Main entry points are in [scripts](./scripts). Each configurable script has a corresponding [config](./configs) file and [launcher](./launchers) file.

`train_ae.py` trains autoencoder variants, including the quantized-latent autoencoder (QLAE), vanilla AE, VAE, TCVAE, BioAE, VQ-VAE, and others.

`train_gan.py` trains InfoWGAN-GP variants, including the quantized-latent InfoWGAN-GP and vanilla InfoWGAN-GP.

Both of these automatically log model and optimizer checkpoints. `plot_mi.py` and `perturbations.py` show how to retrieve a previous run's checkpoint for further analysis.

### Example
To train an autoencoder variant, do `python launchers/train_ae.py`. This will use the configuration defaults in `configs/train_ae.yaml`. To override these defaults, do `python launchers/train_ae.py key=value`. For example, `python launchers/train_ae.py model_partial=ae dataset=isaac3d` will train a vanilla autoencoder on the Isaac3D dataset.

To run a sweep, add the `--multirun` flag. The sweep will run over all combinations of configurations specified in `hydra.sweeper.params` in the config file. 

By default, using `--multirun` will invoke the SubmitIt launcher, which submits jobs to a Slurm cluster. Configure this [here](./configs/hydra/launcher/slurm.yaml). To instead run locally, add `hydra/launcher=local` to the command.

# InfoMEC estimation
A methodological contribution of our paper is a cohesively information-theoretic framework for disentanglement evaluation based on three complementary metrics: InfoM (modularity), InfoE (explicitness), and InfoC (compactness).

### Modularity and Compactness
[This file](./disentangle/metrics/mutual_information.py) contains code for InfoM and InfoC estimation. 

Computing InfoM and InfoC involves estimating the normalized pairwise mutual information between individual latents and sources. We recommend using the `continuous-discrete` estimator for continuous latents and `discrete-discrete` estimator for discrete latents. We do log `discrete-discrete` estimation with various binning settings to demonstrate the sensitivity of continuous latent evaluation done in this manner. We recommend using the `sources` normalization.

Next, the resulting matrix (the transpose of NMI in the paper) is heuristically pruned to remove inactive latents. Finally, the sparsity of each row (for InfoM) or column (for InfoC) of the matrix is computed via a ratio or gap. We recommend and report the ratio.


### Explicitness
[This file](./disentangle/metrics/explicitness.py) contains code for InfoE estimation. 

InfoE involves calculating the predictive linear information from the latents to a source (and averages over sources). We implement and log both classification (logistic regression) and regression (linear regression) formulations of this procedure. As the datasets we use in the paper all have discrete sources, we only report InfoE-classification.

# Citation
If you find this code useful for your work, please cite:
```
@article{hsu2023disentanglement,
  title={Disentanglement via Latent Quantization},
  author={Hsu, Kyle and Dorrell, Will and Whittington, James C. R. and Wu, Jiajun and Finn, Chelsea},
  journal={arXiv preprint arXiv:2305.18378},
  year={2023}
}
```