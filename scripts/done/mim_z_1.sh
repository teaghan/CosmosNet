#!/bin/bash

# Module loads
module load python/3.11.5
source /home/obriaint/project/obriaint/torchnet/bin/activate
module load hdf5/1.10.6

# Copy files to slurm directory
#cp /home/obriaint/scratch/CosmosNet/data/HSC_dud_galaxy_GIRYZ7610_64.h5 $SLURM_TMPDIR
cp /home/obriaint/scratch/CosmosNet/data/HSC_dud_galaxy_zspec_GIRYZ7610_64_train.h5 $SLURM_TMPDIR
cp /home/obriaint/scratch/CosmosNet/data/HSC_dud_galaxy_zspec_GIRYZ7610_64_val.h5 $SLURM_TMPDIR

# Run training
python /home/obriaint/scratch/CosmosNet/run_training.py mim_z_1 -v 2000 -ct 10.00 -dd $SLURM_TMPDIR/

# Run Evaluation
python /home/obriaint/scratch/CosmosNet/run_predictions.py mim_z_1 -dd $SLURM_TMPDIR/
