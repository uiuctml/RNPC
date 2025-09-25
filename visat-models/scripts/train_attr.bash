#!/usr/bin/env bash

cd ../src

# Please change the value of `dataset_prefix` in `./visat-models/src/header.py` to the desired dataset name in advance.
# E.g., dataset_prefix = "mnist_dim_3_min_3_noise_1"
# E.g., dataset_prefix = "mnist_dim_5_min_5_noise_1"
# E.g., dataset_prefix = "celeba_dim_8_min_4"
# E.g., dataset_prefix = "gtsrbsub"

./train_dl_mlp.py -m "mlp_set" -b 256 -e 100 -s 42