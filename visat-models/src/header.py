import multiprocessing
import type

dataset_prefix = "mnist_dim_3_min_3_noise_1" # to be changed
attack_targeted_attribute = 0
cuda_allow_tf32 = False
dir_output_spn = "../output/spn"
file_name_spn_manual = "manual.spn.txt"
file_path_dataset_config = "../../visat-dataset-tools/configs/" + dataset_prefix + ".json"
log_level = type.LogLevel.debug
project_name = "visat-models"
run_mode = "online" # "disabled"
run_name_baseline_keyword = "baseline"
run_name_decomposed_keyword = "decomposed"
run_name_reference_keyword = "reference"
run_name_spn_keyword = "spn"
run_name_baseline = ""
run_name_decomposed = ""
run_name_reference = ""
run_name_spn = ""
seed = 42
show_model_summary = False

config_baseline = {
    "data_loader_batch_size": 256,
    "data_loader_shuffle": True,
    "data_loader_worker_count": multiprocessing.cpu_count(),
    "dir_checkpoints": "/data/common/weixinchen/visat-models/checkpoints",
    "dir_dataset_test": "../../" + dataset_prefix + "-dataset/images/split/original/test",
    "dir_dataset_train": "../../" + dataset_prefix + "-dataset/images/split/original/train",
    "dir_dataset_validation": "../../" + dataset_prefix + "-dataset/images/split/original/validate",
    "dir_test_output": "../output/test",
    "epochs": 100,
    "file_name_checkpoint": run_name_baseline + ".tar",
    "file_name_checkpoint_best": run_name_baseline + ".best.tar",
    "file_name_test_output_clean": "test_output_baseline_clean.txt",
    "file_name_test_output_attacked": "test_output_baseline_attacked.txt",
    "fine_tuning": True,
    "input_grayscale": False,
    "learning_rate_scheduler_mode": "min",
    "learning_rate_scheduler_factor": 0.5,
    "learning_rate_scheduler_patience": 3,
    "learning_rate_scheduler_threshold": 1e-4,
    "learning_rate_scheduler_threshold_mode": "rel",
    "learning_rate_scheduler_cooldown": 3,
    "learning_rate_scheduler_min_learning_rate": 1e-5,
    "learning_rate_scheduler_min_learning_rate_decay": 1e-8,
    "learning_rate_scheduler_verbose": True,
    "log_test_output": False,
    "l2_lambda": 1e-3,
    "model": "vit_b_32",
    "model_input_height": 128,
    "model_input_width": 128,
    "model_input_channels": 3,
    "model_pretrained_weights": "IMAGENET1K_V1",
    "optimizer_learning_rate": 1e-2,
    "optimizer_momentum": 0.9,
    "optimizer_weight_decay": 1e-6,
    "run_name": run_name_baseline,
    "seed": seed,
    "type": "baseline",
    "use_l2_loss": False
}

config_decomposed = {
    "data_loader_batch_size": 256,
    "data_loader_shuffle": True,
    "data_loader_worker_count": multiprocessing.cpu_count(),
    "dataset_delimiter_file_name": "---",
    "dataset_delimiter_label": "--",
    "dataset_label_undefined_keyword": "undefined",
    "dir_checkpoints": "/data/common/weixinchen/visat-models/checkpoints",
    "dir_results": "/data/common/weixinchen/visat-models/results/" + dataset_prefix,
    "dir_dataset_test": "../../" + dataset_prefix + "-dataset/images/split/original/test",
    "dir_dataset_train": "../../" + dataset_prefix + "-dataset/images/split/original/train",
    "dir_dataset_validation": "../../" + dataset_prefix + "-dataset/images/split/original/validate",
    "file_path_neighbor_config": "../../visat-dataset-tools/configs/" + dataset_prefix + "_neighbor.json",
    "dir_test_output": "../output/test",
    "epochs": 100,
    "factor_loss_covariance": 1e-8,
    "file_name_checkpoint": run_name_decomposed + ".tar",
    "file_name_checkpoint_best": run_name_decomposed + ".best.tar",
    "file_name_test_output_clean": "test_output_decomposed_clean.txt",
    "file_name_test_output_attacked": "test_output_decomposed_attacked.txt",
    "fine_tuning": True,
    "head_hidden_size": 128,
    "input_grayscale": False,
    "learning_rate_scheduler_mode": "min",
    "learning_rate_scheduler_factor": 0.5,
    "learning_rate_scheduler_patience": 2,
    "learning_rate_scheduler_threshold": 1e-4,
    "learning_rate_scheduler_threshold_mode": "rel",
    "learning_rate_scheduler_cooldown": 2,
    "learning_rate_scheduler_min_learning_rate": 1e-6,
    "learning_rate_scheduler_min_learning_rate_decay": 1e-8,
    "learning_rate_scheduler_verbose": True,
    "log_test_output": False,
    "l2_lambda": 1e-3,
    "model": "vit_b_32_mtl",
    "model_input_height": 128,
    "model_input_width": 128,
    "model_input_channels": 3,
    "model_pretrained_weights": "IMAGENET1K_V1",
    "optimizer_learning_rate": 1e-3,
    "optimizer_momentum": 0.9,
    "optimizer_weight_decay": 1e-6,
    "run_name": run_name_decomposed,
    "seed": seed,
    "type": "decomposed",
    "use_l2_loss": False,
    "use_covariance_loss": False,
    "attack_name": "pgd",
    "attack_ids": "0",
    "attack_bound": 1.0,
    "targeted_attack": False,
    "dp_eps": 0.9,
    "dp_delta": 0.05,
    "dp_sensitivity": 1.0,
    "num_noise_draw": 10
}

config_reference = {
    "concept_loss_weight": 1,
    "data_loader_batch_size": 256,
    "data_loader_shuffle": True,
    "data_loader_worker_count": multiprocessing.cpu_count(),
    "dir_checkpoints":  "/data/common/weixinchen/visat-models/checkpoints",
    "dir_dataset_test": "../../" + dataset_prefix + "-dataset/images/split/original/test",
    "dir_dataset_train": "../../" + dataset_prefix + "-dataset/images/split/original/train",
    "dir_dataset_validation": "../../" + dataset_prefix + "-dataset/images/split/original/validate",
    "epochs": 100,
    "file_name_checkpoint": run_name_reference + ".tar",
    "file_name_checkpoint_best": run_name_reference + ".best.tar",
    "input_grayscale": False,
    "learning_rate_scheduler_mode": "min",
    "learning_rate_scheduler_factor": 0.1,
    "learning_rate_scheduler_patience": 10,
    "model": "cbm",
    "model_embedding_size": 16,
    "model_input_height": 128,
    "model_input_width": 128,
    "model_input_channels": 3,
    "optimizer_learning_rate": 1e-2,
    "optimizer_momentum": 0.9,
    "optimizer_weight_decay": 4e-5,
    "run_name": run_name_reference,
    "seed": seed,
    "type": "reference",
    "attack_name": "pgd",
    "attack_ids": "0",
    "attack_bound": 1.0
}

config_spn = {
    "dir_checkpoints": "/data/common/weixinchen/visat-models/checkpoints",
    "epochs": 50,
    "file_path_spn": "../../learnspn/output/learnspn/" + dataset_prefix + ".spn.txt",
    "file_path_spn_dataset_test": "../../" + dataset_prefix + "-dataset/images/split/original/test",
    "file_path_spn_dataset_train": "../../" + dataset_prefix + "-dataset/images/split/original/train",
    "file_path_spn_dataset_validation": "../../" + dataset_prefix + "-dataset/images/split/original/validate",
    "file_name_checkpoint": run_name_spn + ".tar",
    "file_name_checkpoint_best": run_name_spn + ".best.tar",
    "fine_tuning": False,
    "epsilon_projection": 1e-2,
    "epsilon_smoothing": 1e-3,
    "growth_threshold": 100,
    "learning_rate_scheduler_factor": 0.5,
    "learning_rate_scheduler_patience": 1,
    "learning_rate_scheduler_threshold": 1e-2,
    "learning_rate_scheduler_cooldown": 1,
    "learning_rate_scheduler_min_learning_rate": 1e-4,
    "model_pretrained_weights": "spn.cccp_generative.nft.42.2024.3.4.11.48.Aurora-R11.best.tar",
    "optimizer": "cccp_generative",
    "optimizer_learning_rate": 1e-1,
    "optimizer_prior_factor": 1e2,
    "randomize_weights": False,
    "run_name": run_name_spn,
    "seed": seed,
    "stopping_criterion": 1e-4,
    "type": "spn",
    "use_l2_loss": False
}

