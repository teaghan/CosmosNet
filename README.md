# CosmosNet
Leveraging semi-supervised learning to enhance our understanding of the universe's expansion.


# Sky Embeddings

Welcome to the Sky Embeddings repository, where we leverage self-supervised learning to generate and utilize embeddings from sky images for tasks such as classification, redshift estimation, and similarity searches.

## Overview

This repository hosts code and methodologies for applying Masked Image Modelling (MIM) to astronomical images, focusing on producing high-quality embeddings that capture the rich, underlying structures of the universe.

### Related Work on Masked Image Modelling

We combined aspects from the [MAE code developed by Facebook AI](https://github.com/facebookresearch/mae) and the [SimMIM Framework for Masked Image Modeling](https://github.com/microsoft/SimMIM) as our primary machine learning pipeline. Leveraging these developments with a few tricks of our own allowed us to create meaningful embeddings from observations of the sky.

### Dependencies

Ensure you have the following installed:

- Python 3.11.5
- PyTorch: `pip install torch==2.0.1`
- h5py: `pip install h5py`
- Scikit-learn `pip install scikit-learn`

<br><br>

## Dataset: Hyper Suprime-Cam (HSC) - Subaru Telescope

Our primary dataset comes from the Hyper Suprime-Cam (HSC) on the Subaru Telescope. Below is an example image from the HSC:

<p align="center">
  <img width="600" height="600" src="./figures/hsc_subaru.jpg"><br>
  <span style="display: block; text-align: right;"><a href="https://subarutelescope.org/en/news/topics/2017/02/27/2459.html">subarutelescope.org</a></span>
</p>

### Data Download

Details on how to access and prepare the HSC data will be provided soon.

<br><br>

## Masked Image Modelling

You can train the network using one of the following methods:

### Training Option 1: Local Training

1. Set model architecture and parameters using a configuration file in [the config directory](./configs). Duplicate the [original configuration file](./configs/mim_z_1.ini) and modify as needed.
2. To train a model with a new config file named `mim_z_2.ini`, use `python run_training.py mim_z_2 -v 5000 -ct 10.00`, which will train your model displaying the progress every 5000 batch iterations and the model would be saved every 10 minutes. The script will also continue training from the last save point.

### Training Option 2: Compute Canada Cluster

For those with access to Compute Canada:

1. Set model architecture and parameters using a configuration file in [the config directory](./configs). Duplicate the [original configuration file](./configs/mim_z_1.ini) and modify as needed.

2. Create and move your training script file into the `scripts/todo/` directory. See [this training script](./scripts/done/mim_z_1.sh) for an example.

```
python cc/queue_cc.py --account "{resource_account_name}" --todo_dir "scripts/todo/" --done_dir "scripts/done/" --output_dir "scripts/stdout/" --num_jobs 1 --num_runs 20 --num_gpu 2 --num_cpu 24 --mem 80G --time_limit "00-03:00"
```

### Evaluation

**During training**, several types of figures will be created to track the progress. 

The [evaluation metrics will be plotted](./figures/mim_z_1_progress.png), which includes the loss values for the training and validation data.

Additionally, some of the masked image modelling results will be plotted throughout training, similar to [this one](./figures/mim_z_1_1000000iters.png).

**After training**,

<br><br>

