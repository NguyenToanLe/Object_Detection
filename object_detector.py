# Step 1: Download, extract and add protoc (bin) to Environment Variable
#         https://github.com/protocolbuffers/protobuf/releases/tag/v3.8.0
# Step 2: Clone Tensorflow Object Detection
#         https://github.com/tensorflow/models/tree/master
# Step 3: Install Tensorflow Object Detection API (Install on terminal, NOT in IDE)
#         - cd .\models\research
#         - protoc object_detection\protos\*.proto --python_out=.
#         - cp object_detection\packages\tf2\setup.py .
#         - pip install .
#         - python setup.py build
#         - python setup.py install
#         - cd .\slim
#         - pip install -e .
# Step 4: Install CUDA (12.3) and Cudnn (8.9)
#         https://www.tensorflow.org/install/source
#         https://www.tensorflow.org/install/pip
#         This might be helpful during verifying:
#               pip install --upgrade --force-reinstall zstandard
#               python -c "import zstandard as zstd; print(zstd.__version__)"


import os
import subprocess
import sys
import tensorflow as tf
from object_detection.utils import config_util


def create_label_maps(labels, label_map_file):
    # The .pbtxt label map file should have following structure
    # item {
    #   name:'[Class name]'
    #   id:1
    # }
    # item {
    #   name:'[Class name]'
    #   id:2
    # }

    with open(label_map_file, 'w') as f:
        for id, label in enumerate(labels):
            f.write("item {\n")
            f.write(f"\tname:\'{label}\'\n")
            f.write(f"id:{id+1}\n")
            f.write("}\n")


def create_tf_record(script, image_train_path, image_test_path, label_map_path,
                     tfrecord_train_path, tfrecord_test_path):
    print("Generating TF Record for train dataset......")
    try:
        result = subprocess.run(
            [sys.executable, script, "-x", image_train_path, "-l", label_map_path, "-o", tfrecord_train_path],
            capture_output=True, text=True, check=True
        )
        print(result.stdout)
    except Exception as e:
        print(f"Failed to generate TF Record for train dataset with following error:\n{e}")
    print("Generating TF Record for test dataset......")
    try:
        result = subprocess.run(
            [sys.executable, script, "-x", image_test_path, "-l", label_map_path, "-o", tfrecord_test_path],
            capture_output=True, text=True, check=True
        )
        print(result.stdout)
    except Exception as e:
        print(f"Failed to generate TF Record for train dataset with following error:\n{e}")
    print("Done!")


def main():
    # Define some constants
    NAME = {
        "custom_model_name": "my_ssd_mobnet",
        "model": "ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8",
        "tfrecord_generator": "generate_tfrecord.py",
        "label_map": "label_map.pbtxt",
        "training_script": "model_main_tf2.py"
    }

    PATH = {
        "checkpoint": os.path.join(".", "custom_model"),
        "output": os.path.join(".", "custom_model", "export"),
        "tfjs": os.path.join(".", "custom_model", "tfjs_export"),
        "tflite": os.path.join(".", "custom_model", "tflite_export"),
        "model": os.path.join(".", "pretrained_model", NAME["model"]),
        "pipeline_config": os.path.join(".", "pretrained_model", NAME["model"]),
        "tfrecord_generator": os.path.join(".", "GenerateTFRecord"),
        "label_map": os.path.join(".", "dataset", "annotation"),
        "train_images": os.path.join(".", "dataset", "images", "train"),
        "test_images": os.path.join(".", "dataset", "images", "test")
    }

    FILE = {
        "pipeline_config": os.path.join(PATH["pipeline_config"], "pipeline.config"),
        "checkpoint": os.path.join(PATH["pipeline_config"], "checkpoint", "ckpt-0"),
        "tfrecord_generator": os.path.join(PATH["tfrecord_generator"], NAME["tfrecord_generator"]),
        "label_map": os.path.join(PATH["label_map"], NAME["label_map"]),
        "training_script": os.path.join(".", "models", "research", "object_detection", NAME["training_script"])
    }

    img_classes = ['One', 'Two', 'Three', 'Four', 'Five',
                   'Six', 'Seven', 'Eight', 'Nine', 'Ten']

    hyperparameters = {
        "batch_size": 4,
        "num_steps": 2000,
        "checkpoint_type": "detection"
    }

    # Create folders if not existed:
    for _, path in PATH.items():
        if not os.path.exists(path):
            os.makedirs(path)

    # Create custom label map
    create_label_maps(labels=img_classes, label_map_file=FILE["label_map"])

    # Create TF Records for train and test set
    create_tf_record(script=FILE["tfrecord_generator"],
                     image_train_path=PATH["train_images"],
                     image_test_path=PATH["test_images"],
                     label_map_path=FILE["label_map"],
                     tfrecord_train_path=os.path.join(PATH["label_map"], "train.record"),
                     tfrecord_test_path=os.path.join(PATH["label_map"], "test.record"))

    # Update config file for transfer learning
    # Features that have to be updated:
    #   - .model.ssd.num_classes
    #   - .train_config.batch_size
    #   - .train_config.fine_tune_checkpoint
    #   - .train_config.num_steps
    #   - .train_config.fine_tune_checkpoint_type
    #   - .train_input_config.label_map_path
    #   - .train_input_config.tf_record_input_reader.input_path
    #   - .eval_input_config.label_map_path
    #   - .eval_input_config.tf_record_input_reader.input_path
    config = config_util.get_configs_from_pipeline_file(FILE["pipeline_config"])
    config["model"].ssd.num_classes = len(img_classes)
    config["train_config"].batch_size = hyperparameters["batch_size"]
    config["train_config"].fine_tune_checkpoint = FILE["checkpoint"]
    config["train_config"].num_steps = hyperparameters["num_steps"]
    config["train_config"].fine_tune_checkpoint_type = hyperparameters["checkpoint_type"]
    config["train_input_config"].label_map_path = FILE["label_map"]
    config["train_input_config"].tf_record_input_reader.input_path[0] = os.path.join(PATH["label_map"], "train.record")
    config["eval_input_config"].label_map_path = FILE["label_map"]
    config["eval_input_config"].tf_record_input_reader.input_path[0] = os.path.join(PATH["label_map"], "test.record")

    # TRAINING
    print("Training model with custom dataset......")
    try:
        checkpoint_path = PATH["checkpoint"]
        config_file = PATH["pipeline_config"]
        num_steps = hyperparameters["num_steps"]
        result = subprocess.run(
            [sys.executable, FILE["training_script"], f"--model_dir={checkpoint_path}",
             f"--pipline_config_path={config_file}", f"--num_train_steps={num_steps}"],
            capture_output=True, text=True, check=True
        )
        print(result.stdout)
    except Exception as e:
        print(f"Failed to train model with following error:\n{e}")



if __name__ == "__main__":
    main()
    print("Finish!!")
