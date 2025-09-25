#!/usr/bin/env bash

cd ../src

# Please change the value of `dataset_prefix` in `./visat-models/src/header.py` to the desired dataset name in advance.
# E.g., dataset_prefix = "mnist_dim_3_min_3_noise_1"
# E.g., dataset_prefix = "mnist_dim_5_min_5_noise_1"
# E.g., dataset_prefix = "celeba_dim_8_min_4"
# E.g., dataset_prefix = "gtsrbsub"

setting="gtsrbsub"

# No attack
echo "Testing vanilla model's benign performance..."
./test_composed.py -rd "decomposed.mlp_set.${setting}" -rs "spn.cccp_generative.${setting}" -s 42

echo "Testing robust model's benign performance..."
./test_composed_robust.py -rd "decomposed.mlp_set.${setting}" -rs "spn.cccp_generative.${setting}" -s 42

echo "Testing cbm model's benign performance..."
./test_reference.py -r "reference.cbm.${setting}" -s 42

echo "Testing dcr model's benign performance..."
./test_reference.py -r "reference.dcr.${setting}" -s 42

# PGD attack
attack_name="pgd"
attack_ids_list=("0" "1" "2" "3")
attack_bounds=(0.03 0.05 0.07 0.09 0.11)

# Iterate over attack_ids, and attack_bound
for attack_ids in "${attack_ids_list[@]}"; do
  for attack_bound in "${attack_bounds[@]}"; do

    echo "Testing vanilla model's adversarial performance against ${attack_name} attack..."
    ./test_composed_attack.py -rd "decomposed.mlp_set.${setting}" \
                              -rs "spn.cccp_generative.${setting}" \
                              -s 42 --attack_name "${attack_name}" \
                              --attack_ids "${attack_ids}" \
                              --attack_bound ${attack_bound}

    echo "Testing robust model's adversarial performance against ${attack_name} attack..."
    ./test_composed_robust_attack.py -rd "decomposed.mlp_set.${setting}" \
                                    -rs "spn.cccp_generative.${setting}" \
                                    -s 42 --attack_name "${attack_name}" \
                                    --attack_ids "${attack_ids}" \
                                    --attack_bound ${attack_bound}

    echo "Testing cbm's adversarial performance against ${attack_name} attack..."
    ./test_reference_attack.py -r "reference.cbm.${setting}" \
                              -s 42 --attack_name "${attack_name}" \
                              --attack_ids "${attack_ids}" \
                              --attack_bound ${attack_bound}

    echo "Testing dcr's adversarial performance against ${attack_name} attack..."
    ./test_reference_attack.py -r "reference.dcr.${setting}" \
                              -s 42 --attack_name "${attack_name}" \
                              --attack_ids "${attack_ids}" \
                              --attack_bound ${attack_bound}


    echo "Completed: attack_name=${attack_name}, attack_ids=${attack_ids}, attack_bound=${attack_bound}"
    echo "-----------------------------"

  done
done

# PGDL2 attack
attack_name="pgdl2"
attack_ids_list=("0" "1" "2" "3")
attack_bounds=(3 5 7 9 11)

# Iterate over attack_ids, and attack_bound
for attack_ids in "${attack_ids_list[@]}"; do
  for attack_bound in "${attack_bounds[@]}"; do

    echo "Testing vanilla model's adversarial performance against ${attack_name} attack..."
    ./test_composed_attack.py -rd "decomposed.mlp_set.${setting}" \
                              -rs "spn.cccp_generative.${setting}" \
                              -s 42 --attack_name "${attack_name}" \
                              --attack_ids "${attack_ids}" \
                              --attack_bound ${attack_bound}

    echo "Testing robust model's adversarial performance against ${attack_name} attack..."
    ./test_composed_robust_attack.py -rd "decomposed.mlp_set.${setting}" \
                                    -rs "spn.cccp_generative.${setting}" \
                                    -s 42 --attack_name "${attack_name}" \
                                    --attack_ids "${attack_ids}" \
                                    --attack_bound ${attack_bound}

    echo "Testing cbm's adversarial performance against ${attack_name} attack..."
    ./test_reference_attack.py -r "reference.cbm.${setting}" \
                              -s 42 --attack_name "${attack_name}" \
                              --attack_ids "${attack_ids}" \
                              --attack_bound ${attack_bound}

    echo "Testing dcr's adversarial performance against ${attack_name} attack..."
    ./test_reference_attack.py -r "reference.dcr.${setting}" \
                              -s 42 --attack_name "${attack_name}" \
                              --attack_ids "${attack_ids}" \
                              --attack_bound ${attack_bound}

    echo "Completed: attack_name=${attack_name}, attack_ids=${attack_ids}, attack_bound=${attack_bound}"
    echo "-----------------------------"

  done
done

# CW attack
attack_name="cw"
attack_ids_list=("0" "1" "2" "3")
attack_bounds=(3 4 5 6 7)

# Iterate over attack_ids, and attack_bound
for attack_ids in "${attack_ids_list[@]}"; do
  for attack_bound in "${attack_bounds[@]}"; do

    echo "Testing vanilla model's adversarial performance against ${attack_name} attack..."
    ./test_composed_attack.py -rd "decomposed.mlp_set.${setting}" \
                              -rs "spn.cccp_generative.${setting}" \
                              -s 42 --attack_name "${attack_name}" \
                              --attack_ids "${attack_ids}" \
                              --attack_bound ${attack_bound}

    echo "Testing robust model's adversarial performance against ${attack_name} attack..."
    ./test_composed_robust_attack.py -rd "decomposed.mlp_set.${setting}" \
                                    -rs "spn.cccp_generative.${setting}" \
                                    -s 42 --attack_name "${attack_name}" \
                                    --attack_ids "${attack_ids}" \
                                    --attack_bound ${attack_bound}

    echo "Testing cbm's adversarial performance against ${attack_name} attack..."
    ./test_reference_attack.py -r "reference.cbm.${setting}" \
                              -s 42 --attack_name "${attack_name}" \
                              --attack_ids "${attack_ids}" \
                              --attack_bound ${attack_bound}

    echo "Testing dcr's adversarial performance against ${attack_name} attack..."
    ./test_reference_attack_dcr.py -r "reference.dcr.${setting}" \
                              -s 42 --attack_name "${attack_name}" \
                              --attack_ids "${attack_ids}" \
                              --attack_bound ${attack_bound}

    echo "Completed: attack_name=${attack_name}, attack_ids=${attack_ids}, attack_bound=${attack_bound}"
    echo "-----------------------------"

  done
done