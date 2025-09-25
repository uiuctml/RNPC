import argparse
import datetime
import header
import logger
import socket

def generateRunName(model, type, seed, fine_tuning):
    date_time_list = list(datetime.datetime.now().timetuple())[:-4]
    run_name = model
    run_name += "."
    run_name += type
    run_name += "."

    if fine_tuning:
        run_name += "ft"
    else:
        run_name += "nft"

    run_name += "."
    run_name += str(seed)

    for entry in date_time_list:
        run_name += "."
        run_name += str(entry)

    run_name += "."
    run_name += socket.gethostname()

    return run_name

def initializeArgumentsAttack():
    parser = argparse.ArgumentParser()

    parser.add_argument("-r", "--run-name", type = str, default = "", help = "Run name.", required = True)
    parser.add_argument("-s", "--seed", type = int, default = 42, help = "Randomization seed.")
    parser.add_argument("-d", "--test-dataset-dir", type = str, default = "", help = "Directory of dataset testing split.")
    parser.add_argument("-a", "--attribute", type = int, default = 0, help = "Targeted attribute.")

    return parser.parse_args()

def initializeArgumentsTest():
    parser = argparse.ArgumentParser()

    parser.add_argument("-r", "--run-name", type = str, default = "", help = "Run name.", required = True)
    parser.add_argument("-s", "--seed", type = int, default = 42, help = "Randomization seed.")
    parser.add_argument("-d", "--test-dataset-dir", type = str, default = "", help = "Directory of dataset testing split.")
    parser.add_argument("--attack_name", type=str, default="", help="Attack name.")
    parser.add_argument("--attack_ids", type=int, nargs="+", help="A list of attack indexes")
    parser.add_argument("--attack_bound", type=float, default=1.0, help="eps in PGD(L2), c in CW")

    return parser.parse_args()

def initializeArgumentsTestComposed():
    parser = argparse.ArgumentParser()

    parser.add_argument("-rd", "--decomposed-run-name", type = str, default = "", help = "Decomposed run name.", required = True)
    parser.add_argument("-rs", "--spn-run-name", type = str, default = "", help = "SPN run name.", required = True)
    parser.add_argument("-s", "--seed", type = int, default = 42, help = "Randomization seed.")
    parser.add_argument("-d", "--test-dataset-dir", type = str, default = "", help = "Directory of dataset testing split.")
    parser.add_argument("-b", "--batch-size", type=int, default=256, help="Batch size.")
    parser.add_argument("--attack_name", type = str, default = "", help = "Attack name.")
    parser.add_argument("--attack_ids", type=int, nargs="+", help="A list of attack indexes")
    parser.add_argument("--attack_bound", type=float, default=1.0, help="eps in PGD(L2), c in CW")

    return parser.parse_args()

def initializeArgumentsTestSPN():
    parser = argparse.ArgumentParser()

    parser.add_argument("-r", "--run-name", type = str, default = "", help = "Run name.", required = True)
    parser.add_argument("-s", "--seed", type = int, default = 42, help = "Randomization seed.")

    return parser.parse_args()

def initializeArgumentsTrain():
    parser = argparse.ArgumentParser()

    parser.add_argument("-r", "--run-name", type = str, default = "", help = "Run name.")
    parser.add_argument("-m", "--model", type = str, default = "", help = "Model to train.")
    parser.add_argument("-b", "--batch-size", type = int, default = 512, help = "Batch size.")
    parser.add_argument("-e", "--epochs", type = int, default = 100, help = "Number of training epochs.")
    parser.add_argument("-f", "--fine-tune", type = int, default = 1, help = "Whether to perform backbone fine-tuning.")
    parser.add_argument("-c", "--use-covariance-loss", type = int, default = 0, help = "Whether to use covariance loss.")
    parser.add_argument("-s", "--seed", type = int, default = 42, help = "Randomization seed.")

    return parser.parse_args()

def initializeArgumentsTrainComposed():
    parser = argparse.ArgumentParser()

    parser.add_argument("-rd", "--decomposed-run-name", type = str, default = "", help = "Decomposed run name.")
    parser.add_argument("-rs", "--spn-run-name", type = str, default = "", help = "SPN run name.")
    parser.add_argument("-wd", "--decomposed-pretrained-weights", type = str, default = "", help = "Decomposed model pretrained weights.")
    parser.add_argument("-ws", "--spn-pretrained-weights", type = str, default = "", help = "SPN model pretrained weights.")
    parser.add_argument("-fd", "--decomposed-fine-tune", type = int, default = 1, help = "Whether to perform decomposed fine-tuning.")
    parser.add_argument("-fs", "--spn-fine-tune", type = int, default = 1, help = "Whether to perform SPN fine-tuning.")
    parser.add_argument("-m", "--model", type = str, default = "", help = "Model to train.")
    parser.add_argument("-o", "--optimizer", type = str, default = "", help = "SPN optimizer.")
    parser.add_argument("-b", "--batch-size", type = int, default = 512, help = "Batch size.")
    parser.add_argument("-e", "--epochs", type = int, default = 100, help = "Number of training epochs.")
    parser.add_argument("-c", "--use-covariance-loss", type = int, default = 0, help = "Whether to use covariance loss.")
    parser.add_argument("-r", "--randomize-weights", type = int, default = 0, help = "Whether to randomize weights.")
    parser.add_argument("-s", "--seed", type = int, default = 42, help = "Randomization seed.")

    return parser.parse_args()

def initializeArgumentsTrainReference():
    parser = argparse.ArgumentParser()

    parser.add_argument("-m", "--model", type = str, default = "", help = "Model to train.")
    parser.add_argument("-b", "--batch-size", type = int, default = None, help = "Batch size.")
    parser.add_argument("-e", "--epochs", type = int, default = None, help = "Number of training epochs.")
    parser.add_argument("-s", "--seed", type = int, default = None, help = "Randomization seed.")

    return parser.parse_args()

def initializeArgumentsTrainSPN():
    parser = argparse.ArgumentParser()

    parser.add_argument("-r", "--run-name", type = str, default = "", help = "Run name.")
    parser.add_argument("-o", "--optimizer", type = str, default = "", help = "SPN optimizer.")
    parser.add_argument("-e", "--epochs", type = int, default = 50, help = "Number of training epochs.")
    parser.add_argument("-f", "--fine-tune", type = int, default = 0, help = "Whether to perform fine-tuning.")
    parser.add_argument("-c", "--stopping-criterion", type = float, default = 1e-4, help = "Stopping criterion.")
    parser.add_argument("-s", "--seed", type = int, default = 42, help = "Randomization seed.")

    return parser.parse_args()

def initializeRunNameBaseline(run_name):
    resume = False

    if run_name != "":
        if run_name.split(".")[0] != header.run_name_baseline_keyword:
            logger.log_fatal("Invalid baseline run name. Quit.")
            exit(-1)

        header.run_name_baseline = run_name
        resume = True
    else:
        header.run_name_baseline = generateRunName(header.run_name_baseline_keyword, header.config_baseline["model"], header.config_baseline["seed"], header.config_baseline["fine_tuning"])

    header.config_baseline["file_name_checkpoint"] = header.run_name_baseline + ".tar"
    header.config_baseline["file_name_checkpoint_best"] = header.run_name_baseline + ".best.tar"
    header.config_baseline["run_name"] = header.run_name_baseline

    return resume

def initializeRunNameDecomposed(run_name):
    resume = False

    if run_name != "":
        if run_name.split(".")[0] != header.run_name_decomposed_keyword:
            logger.log_fatal("Invalid decomposed run name. Quit.")
            exit(-1)

        header.run_name_decomposed = run_name
        resume = True
    else:
        header.run_name_decomposed = generateRunName(header.run_name_decomposed_keyword, header.config_decomposed["model"], header.config_decomposed["seed"], header.config_decomposed["fine_tuning"])

    header.config_decomposed["file_name_checkpoint"] = header.run_name_decomposed + ".tar"
    header.config_decomposed["file_name_checkpoint_best"] = header.run_name_decomposed + ".best.tar"
    header.config_decomposed["run_name"] = header.run_name_decomposed

    return resume

def initializeRunNameReference(run_name = ""):
    if run_name != "":
        if run_name.split(".")[0] != header.run_name_reference_keyword:
            logger.log_fatal("Invalid reference run name. Quit.")
            exit(-1)

        header.run_name_reference = run_name
    else:
        header.run_name_reference = generateRunName(header.run_name_reference_keyword, header.config_reference["model"], header.config_reference["seed"], True)

    header.config_reference["file_name_checkpoint"] = header.run_name_reference + ".tar"
    header.config_reference["file_name_checkpoint_best"] = header.run_name_reference + ".best.tar"
    header.config_reference["run_name"] = header.run_name_reference

    return

def initializeRunNameSPN(run_name):
    resume = False

    if run_name != "":
        if run_name.split(".")[0] != header.run_name_spn_keyword:
            logger.log_fatal("Invalid SPN run name. Quit.")
            exit(-1)

        header.run_name_spn = run_name
        resume = True
    else:
        header.run_name_spn = generateRunName(header.run_name_spn_keyword, header.config_spn["optimizer"], header.config_spn["seed"], header.config_spn["fine_tuning"])

    header.config_spn["file_name_checkpoint"] = header.run_name_spn + ".tar"
    header.config_spn["file_name_checkpoint_best"] = header.run_name_spn + ".best.tar"
    header.config_spn["run_name"] = header.run_name_spn

    return resume

def processArgumentsAttack():
    arguments = initializeArgumentsAttack()

    header.config_decomposed["seed"] = arguments.seed

    if arguments.test_dataset_dir != "":
        header.config_decomposed["dir_dataset_test"] = arguments.test_dataset_dir

    if not initializeRunNameDecomposed(arguments.run_name) or header.run_name_decomposed == "":
        logger.log_fatal("Run name missing. Quit.")
        exit(-1)

    header.config_decomposed["model"] = header.run_name_decomposed.split(".")[1]

    header.attack_targeted_attribute = arguments.attribute

    logger.log_trace("Run name: \"" + header.run_name_decomposed + "\".")
    logger.log_trace("Model: \"" + header.config_decomposed["model"] + "\".")
    logger.log_trace("Randomization seed: " + str(header.config_decomposed["seed"]) + ".")
    logger.log_trace("Directory of dataset testing split: \"" + header.config_decomposed["dir_dataset_test"] + "\".")
    logger.log_trace("Targeted attribute: \"" + str(header.attack_targeted_attribute) + "\".")

    return

def processArgumentsTestBaseline():
    arguments = initializeArgumentsTest()

    header.config_baseline["seed"] = arguments.seed

    if arguments.test_dataset_dir != "":
        header.config_baseline["dir_dataset_test"] = arguments.test_dataset_dir

    if not initializeRunNameBaseline(arguments.run_name) or header.run_name_baseline == "":
        logger.log_fatal("Run name missing. Quit.")
        exit(-1)

    header.config_baseline["model"] = header.run_name_baseline.split(".")[1]

    logger.log_trace("Run name: \"" + header.run_name_baseline + "\".")
    logger.log_trace("Model: \"" + header.config_baseline["model"] + "\".")
    logger.log_trace("Randomization seed: " + str(header.config_baseline["seed"]) + ".")
    logger.log_trace("Directory of dataset testing split: \"" + header.config_baseline["dir_dataset_test"] + "\".")

    return

def processArgumentsTestComposed():
    arguments = initializeArgumentsTestComposed()

    header.config_decomposed["seed"] = arguments.seed
    header.config_spn["seed"] = arguments.seed
    header.config_decomposed["data_loader_batch_size"] = arguments.batch_size

    if arguments.test_dataset_dir != "":
        header.config_decomposed["dir_dataset_test"] = arguments.test_dataset_dir

    if not initializeRunNameDecomposed(arguments.decomposed_run_name) or header.run_name_decomposed == "":
        logger.log_fatal("Decomposed run name missing. Quit.")
        exit(-1)

    if not initializeRunNameSPN(arguments.spn_run_name) or header.run_name_spn == "":
        logger.log_fatal("SPN run name missing. Quit.")
        exit(-1)

    header.config_decomposed["model"] = header.run_name_decomposed.split(".")[1]
    header.config_spn["optimizer"] = header.run_name_spn.split(".")[1]

    # Configure attack type and ids
    if arguments.attack_name != "":
        header.config_decomposed["attack_name"] = arguments.attack_name
        header.config_decomposed["attack_ids"] = arguments.attack_ids
        header.config_decomposed["attack_bound"] = arguments.attack_bound

    logger.log_trace("Decomposed run name: \"" + header.run_name_decomposed + "\".")
    logger.log_trace("SPN run name: \"" + header.run_name_spn + "\".")
    logger.log_trace("Decomposed model: \"" + header.config_decomposed["model"] + "\".")
    logger.log_trace("SPN optimizer: \"" + header.config_spn["optimizer"] + "\".")
    logger.log_trace("Randomization seed: " + str(header.config_decomposed["seed"]) + ".")
    logger.log_trace("Directory of dataset testing split: \"" + header.config_decomposed["dir_dataset_test"] + "\".")

    return

def processArgumentsTestDecomposed():
    arguments = initializeArgumentsTest()

    header.config_decomposed["seed"] = arguments.seed

    if arguments.test_dataset_dir != "":
        header.config_decomposed["dir_dataset_test"] = arguments.test_dataset_dir

    if not initializeRunNameDecomposed(arguments.run_name) or header.run_name_decomposed == "":
        logger.log_fatal("Run name missing. Quit.")
        exit(-1)

    header.config_decomposed["model"] = header.run_name_decomposed.split(".")[1]

    logger.log_trace("Run name: \"" + header.run_name_decomposed + "\".")
    logger.log_trace("Model: \"" + header.config_decomposed["model"] + "\".")
    logger.log_trace("Randomization seed: " + str(header.config_decomposed["seed"]) + ".")
    logger.log_trace("Directory of dataset testing split: \"" + header.config_decomposed["dir_dataset_test"] + "\".")

    return

def processArgumentsTestReference():
    arguments = initializeArgumentsTest()

    header.config_reference["seed"] = arguments.seed

    if arguments.test_dataset_dir != "":
        header.config_reference["dir_dataset_test"] = arguments.test_dataset_dir

    initializeRunNameReference(arguments.run_name)

    if header.run_name_reference == "":
        logger.log_fatal("Run name missing. Quit.")
        exit(-1)

    header.config_reference["model"] = header.run_name_reference.split(".")[1]

    # Configure attack type and ids
    if arguments.attack_name != "":
        header.config_reference["attack_name"] = arguments.attack_name
        header.config_reference["attack_ids"] = arguments.attack_ids
        header.config_reference["attack_bound"] = arguments.attack_bound

    logger.log_trace("Run name: \"" + header.run_name_reference + "\".")
    logger.log_trace("Model: \"" + header.config_reference["model"] + "\".")
    logger.log_trace("Randomization seed: " + str(header.config_reference["seed"]) + ".")
    logger.log_trace("Directory of dataset testing split: \"" + header.config_reference["dir_dataset_test"] + "\".")

    return

def processArgumentsTestSPN():
    arguments = initializeArgumentsTestSPN()

    header.config_spn["seed"] = arguments.seed

    resume = initializeRunNameSPN(arguments.run_name)

    if header.run_name_spn == "":
        logger.log_fatal("Run name missing. Quit.")
        exit(-1)

    logger.log_trace("Run name: \"" + header.run_name_spn + "\".")
    logger.log_trace("Randomization seed: " + str(header.config_spn["seed"]) + ".")

    return resume

def processArgumentsTrainBaseline():
    arguments = initializeArgumentsTrain()

    if arguments.model != "":
        header.config_baseline["model"] = arguments.model

    header.config_baseline["data_loader_batch_size"] = arguments.batch_size
    header.config_baseline["epochs"] = arguments.epochs

    if arguments.fine_tune == 0:
        header.config_baseline["fine_tuning"] = False
    else:
        header.config_baseline["fine_tuning"] = True

    header.config_baseline["seed"] = arguments.seed

    resume = initializeRunNameBaseline(arguments.run_name)

    if header.run_name_baseline == "":
        logger.log_fatal("Run name missing. Quit.")
        exit(-1)

    header.config_baseline["model"] = header.run_name_baseline.split(".")[1]

    logger.log_trace("Run name: \"" + header.run_name_baseline + "\".")
    logger.log_trace("Model: \"" + header.config_baseline["model"] + "\".")
    logger.log_trace("Batch size: " + str(header.config_baseline["data_loader_batch_size"]) + ".")
    logger.log_trace("Epochs: " + str(header.config_baseline["epochs"]) + ".")
    logger.log_trace("Whether to perform backbone fine-tuning: " + str(header.config_baseline["fine_tuning"]) + ".")
    logger.log_trace("Randomization seed: " + str(header.config_baseline["seed"]) + ".")

    return resume

def processArgumentsTrainComposed():
    arguments = initializeArgumentsTrainComposed()

    if arguments.decomposed_pretrained_weights != "":
        header.config_decomposed["model_pretrained_weights"] = arguments.decomposed_pretrained_weights

    if arguments.spn_pretrained_weights != "":
        header.config_spn["model_pretrained_weights"] = arguments.spn_pretrained_weights

    if arguments.decomposed_fine_tune == 0:
        header.config_decomposed["fine_tuning"] = False
    else:
        header.config_decomposed["fine_tuning"] = True

    if arguments.spn_fine_tune == 0:
        header.config_spn["fine_tuning"] = False
    else:
        header.config_spn["fine_tuning"] = True

    if arguments.model != "":
        header.config_decomposed["model"] = arguments.model

    if arguments.optimizer != "":
        header.config_spn["optimizer"] = arguments.optimizer

    header.config_decomposed["data_loader_batch_size"] = arguments.batch_size
    header.config_decomposed["epochs"] = arguments.epochs

    if arguments.use_covariance_loss == 0:
        header.config_decomposed["use_covariance_loss"] = False
    else:
        header.config_decomposed["use_covariance_loss"] = True

    if arguments.randomize_weights == 0:
        header.config_spn["randomize_weights"] = False
    else:
        header.config_spn["randomize_weights"] = True

    header.config_decomposed["seed"] = arguments.seed

    resume_decomposed = initializeRunNameDecomposed(arguments.decomposed_run_name)
    resume_spn = initializeRunNameSPN(arguments.spn_run_name)
    resume = resume_decomposed and resume_spn

    if header.run_name_decomposed == "":
        logger.log_fatal("Decomposed run name missing. Quit.")
        exit(-1)

    if header.run_name_spn == "":
        logger.log_fatal("SPN run name missing. Quit.")
        exit(-1)

    header.config_decomposed["model"] = header.run_name_decomposed.split(".")[1]
    header.config_spn["optimizer"] = header.run_name_spn.split(".")[1]

    logger.log_trace("Decomposed run name: \"" + header.run_name_decomposed + "\".")
    logger.log_trace("SPN run name: \"" + header.run_name_spn + "\".")
    logger.log_trace("Decomposed model: \"" + header.config_decomposed["model"] + "\".")
    logger.log_trace("SPN optimizer: \"" + header.config_spn["optimizer"] + "\".")
    logger.log_trace("Decomposed model pretrained weights: \"" + header.config_decomposed["model_pretrained_weights"] + "\".")
    logger.log_trace("SPN model pretrained weights: \"" + header.config_spn["model_pretrained_weights"] + "\".")
    logger.log_trace("Whether to perform decomposed fine-tuning: " + str(header.config_decomposed["fine_tuning"]) + ".")
    logger.log_trace("Whether to perform SPN fine-tuning: " + str(header.config_spn["fine_tuning"]) + ".")
    logger.log_trace("Batch size: " + str(header.config_decomposed["data_loader_batch_size"]) + ".")
    logger.log_trace("Epochs: " + str(header.config_decomposed["epochs"]) + ".")
    logger.log_trace("Whether to use covariance loss: " + str(header.config_decomposed["use_covariance_loss"]) + ".")
    logger.log_trace("Whether to randomize SPN weights: " + str(header.config_spn["randomize_weights"]) + ".")
    logger.log_trace("Randomization seed: " + str(header.config_decomposed["seed"]) + ".")

    return resume

def processArgumentsTrainDecomposed():
    arguments = initializeArgumentsTrain()

    if arguments.model != "":
        header.config_decomposed["model"] = arguments.model

    header.config_decomposed["data_loader_batch_size"] = arguments.batch_size
    header.config_decomposed["epochs"] = arguments.epochs

    if arguments.fine_tune == 0:
        header.config_decomposed["fine_tuning"] = False
    else:
        header.config_decomposed["fine_tuning"] = True

    if arguments.use_covariance_loss == 0:
        header.config_decomposed["use_covariance_loss"] = False
    else:
        header.config_decomposed["use_covariance_loss"] = True

    header.config_decomposed["seed"] = arguments.seed

    resume = initializeRunNameDecomposed(arguments.run_name)

    if header.run_name_decomposed == "":
        logger.log_fatal("Run name missing. Quit.")
        exit(-1)

    header.config_decomposed["model"] = header.run_name_decomposed.split(".")[1]

    logger.log_trace("Run name: \"" + header.run_name_decomposed + "\".")
    logger.log_trace("Model: \"" + header.config_decomposed["model"] + "\".")
    logger.log_trace("Batch size: " + str(header.config_decomposed["data_loader_batch_size"]) + ".")
    logger.log_trace("Epochs: " + str(header.config_decomposed["epochs"]) + ".")
    logger.log_trace("Whether to perform backbone fine-tuning: " + str(header.config_decomposed["fine_tuning"]) + ".")
    logger.log_trace("Whether to use covariance loss: " + str(header.config_decomposed["use_covariance_loss"]) + ".")
    logger.log_trace("Randomization seed: " + str(header.config_decomposed["seed"]) + ".")

    return resume

def processArgumentsTrainReference():
    arguments = initializeArgumentsTrainReference()

    if arguments.model != "":
        header.config_reference["model"] = arguments.model

    if arguments.batch_size is not None:
        header.config_reference["data_loader_batch_size"] = arguments.batch_size

    if arguments.epochs is not None:
        header.config_reference["epochs"] = arguments.epochs

    if arguments.seed is not None:
        header.config_reference["seed"] = arguments.seed

    initializeRunNameReference()

    if header.run_name_reference == "":
        logger.log_fatal("Run name missing. Quit.")
        exit(-1)

    header.config_reference["model"] = header.run_name_reference.split(".")[1]

    logger.log_trace("Run name: \"" + header.run_name_reference + "\".")
    logger.log_trace("Model: \"" + header.config_reference["model"] + "\".")
    logger.log_trace("Batch size: " + str(header.config_reference["data_loader_batch_size"]) + ".")
    logger.log_trace("Epochs: " + str(header.config_reference["epochs"]) + ".")
    logger.log_trace("Randomization seed: " + str(header.config_reference["seed"]) + ".")

    return

def processArgumentsTrainSPN():
    arguments = initializeArgumentsTrainSPN()

    if arguments.optimizer != "":
        header.config_spn["optimizer"] = arguments.optimizer

    header.config_spn["epochs"] = arguments.epochs

    if arguments.fine_tune == 0:
        header.config_spn["fine_tuning"] = False
    else:
        header.config_spn["fine_tuning"] = True

    header.config_spn["stopping_criterion"] = arguments.stopping_criterion
    header.config_spn["seed"] = arguments.seed

    resume = initializeRunNameSPN(arguments.run_name)

    if header.run_name_spn == "":
        logger.log_fatal("Run name missing. Quit.")
        exit(-1)

    header.config_spn["optimizer"] = header.run_name_spn.split(".")[1]

    logger.log_trace("Run name: \"" + header.run_name_spn + "\".")
    logger.log_trace("Optimizer: \"" + header.config_spn["optimizer"] + "\".")
    logger.log_trace("Epochs: " + str(header.config_spn["epochs"]) + ".")
    logger.log_trace("Whether to perform fine-tuning: " + str(header.config_spn["fine_tuning"]) + ".")
    logger.log_trace("Stopping criterion: " + str(header.config_spn["stopping_criterion"]) + ".")
    logger.log_trace("Randomization seed: " + str(header.config_spn["seed"]) + ".")

    return resume
