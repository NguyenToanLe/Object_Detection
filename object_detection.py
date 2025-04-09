# Step 1: Download, extract and add protoc (bin) to Environment Variable
#         https://github.com/protocolbuffers/protobuf/releases/tag/v3.8.0
# Step 2: Clone Tensorflow Object Detection
#         https://github.com/tensorflow/models/tree/master
# Step 3: Install Tensorflow Object Detection API (Install on terminal, NOT in IDE)
#         - cd .\models\research
#         - protoc object_detection/protos/*.proto --python_out=.
#         - cp object_detection/packages/tf2/setup.py .
#         - python -m pip install .
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
import tensorflow as tf
from object_detection.utils import config_util
from object_detection.protos import pipline_pb2
from google.protobuf import text_format


def main():
    # Define some constants
    MODEL_NAME = "ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8"
    PATH = {
        "model": os.path.join(".", "pretrained_model", MODEL_NAME),
        "pipeline_config": os.path.join(PATH["model"], "pipeline.config"),
        "checkpoint": os.path.join(PATH["model"], "checkpoint", "ckpt-0")
    }

    # Update config file for transfer learning
    config = config_util.get_configs_from_pipeline_file(PATH["pipeline_config"])
    pipeline_config = pipline_pb2.TrainEvalPipelineConfig()
    with tf.io.gfile.GFile(PATH["pipeline_config"], 'r') as f:
        proto_str = f.read()
        text_format.Merge(proto_str, pipeline_config)

    a=1


if __name__ == "__main__":
    main()
    print("Finish!!")
