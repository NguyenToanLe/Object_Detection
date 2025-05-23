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
import sys
import subprocess

import numpy as np
import cv2
import matplotlib.pyplot as plt

import tensorflow as tf

from object_detection.utils import config_util, label_map_util
from object_detection.builders import model_builder
from google.protobuf import text_format
from object_detection.utils import visualization_utils as viz_utils


def get_img_classes(img_path):
    img_classes = []
    for f in os.listdir(img_path):
        class_path = os.path.join(img_path, f)
        if os.path.isdir(class_path):
            img_classes.append(f)

    return img_classes


def initialize_params(args):
    params_dict = {}

    params_dict["NAME"] = {
        "pretrained_model": "ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8"
    }

    params_dict["PATH"] = {
        "pretrained_model": os.path.join(".", "pretrained_model", params_dict["NAME"]["pretrained_model"]),
        "custom_model": os.path.join(".", "custom_model"),
        "images": os.path.join(".", "dataset", "images"),
        "annotation": os.path.join(".", "dataset", "annotation")
    }

    params_dict["FILE"] = {
        "pretrained_config": os.path.join(params_dict["PATH"]["pretrained_model"], "pipeline.config"),
        "custom_config": os.path.join(params_dict["PATH"]["custom_model"], "pipeline.config"),
        "pretrained_checkpoint": os.path.join(params_dict["PATH"]["pretrained_model"], "checkpoint", "ckpt-0"),
        "custom_label_map": os.path.join(params_dict["PATH"]["annotation"], "label_map.pbtxt"),
        "custom_train_tf_record": os.path.join(params_dict["PATH"]["annotation"], "train.record"),
        "custom_test_tf_record": os.path.join(params_dict["PATH"]["annotation"], "test.record"),
        "TF_record_generator": os.path.join(".", "GenerateTFRecord", "generate_tfrecord.py"),
        "training_script": os.path.join(".", "models", "research", "object_detection", "model_main_tf2.py")
    }

    params_dict["IMAGE_CLASSES"], _ = get_img_classes(params_dict["PATH"]["images"])

    params_dict["HYPERPARAMS"] = {
        "batch_size": args.batch_size,
        "num_steps": args.num_steps,
        "checkpoint_type": "detection"
    }

    return params_dict


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


def create_custom_pipeline_config(params):
    config = config_util.get_configs_from_pipeline_file(params["FILE"]["pretrained_config"])
    config["model"].ssd.num_classes = len(params["IMAGE_CLASSES"])
    config["train_config"].batch_size = params["HYPERPARAMS"]["batch_size"]
    config["train_config"].fine_tune_checkpoint = params["FILE"]["pretrained_checkpoint"]
    config["train_config"].num_steps = params["HYPERPARAMS"]["num_steps"]
    config["train_config"].fine_tune_checkpoint_type = params["HYPERPARAMS"]["checkpoint_type"]
    config["train_input_config"].label_map_path = params["FILE"]["custom_label_map"]
    config["train_input_config"].tf_record_input_reader.input_path[0] = params["FILE"]["custom_train_tf_record"]
    config["eval_input_config"].label_map_path = params["FILE"]["custom_label_map"]
    config["eval_input_config"].tf_record_input_reader.input_path[0] = params["FILE"]["custom_test_tf_record"]

    pipeline_config = config_util.create_pipeline_proto_from_configs(config)
    config_text = text_format.MessageToString(pipeline_config, float_format='.17g')
    with tf.io.gfile.GFile(params["FILE"]["custom_config"], 'wb') as f:
        f.write(config_text)


def train_loop(params):
    print("Training model with custom dataset......")
    try:
        training_script = params["FILE"]["training_script"]
        custom_model_path = params["PATH"]["custom_model"]
        config_file = params["FILE"]["custom_config"]
        num_steps = params["HYPERPARAMS"]["num_steps"]
        subprocess.run(
            [sys.executable, training_script, f"--model_dir={custom_model_path}",
             f"--pipeline_config_path={config_file}", f"--num_train_steps={num_steps}"],
            check=True
        )
        # command = "python {} --model_dir={} --pipeline_config_path={} --num_train_steps={}".format(
        #     FILE["training_script"], PATH["checkpoint"], FILE["custom_pipeline_config"], hyperparameters["num_steps"]
        # )
        # print("Please run this command in Terminal:\n", command)
    except Exception as e:
        print(f"Failed to train model with following error:\n{e}")

    # EVALUATING
    print("Evaluating model......")
    try:
        training_script = params["FILE"]["training_script"]
        custom_model_path = params["PATH"]["custom_model"]
        config_file = params["FILE"]["custom_config"]
        subprocess.run(
            [sys.executable, training_script, f"--model_dir={custom_model_path}",
             f"--pipeline_config_path={config_file}", f"--checkpoint_dir={custom_model_path}"],
            check=True
        )
        # command = "python {} --model_dir={} --pipeline_config_path={} --checkpoint_dir={}".format(
        #     FILE["training_script"], PATH["checkpoint"], FILE["custom_pipeline_config"], PATH["checkpoint"]
        # )
        # print("Please run this command in Terminal:\n", command)
    except Exception as e:
        print(f"Failed to evaluate model with following error:\n{e}")


def latest_ckpt(path):
    ckpt_max = "-1"
    dir = os.listdir(path)

    for fname in dir:
        if os.path.isfile(os.path.join(path, fname)) and fname.startswith("ckpt"):
            ckpt = fname.split(".")[0]              # ckpt-1.data-000000-of-00001 --> ckpt-1
            ckpt_nr = int(ckpt.split("-")[-1])      # ckpt-1 --> 1
            if int(ckpt_nr) > int(ckpt_max):
                ckpt_max = ckpt_nr

    return "ckpt-{}".format(ckpt_max)


def inference_single_image(params, ckpt_name, image_path):
    config = config_util.get_configs_from_pipeline_file(params["FILE"]["custom_config"])
    detection_model = model_builder.build(model_config=config["model"], is_training=False)

    ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
    ckpt.restore(os.path.join(params["PATH"]["custom_model"], ckpt_name)).expect_partial()

    @tf.function
    def detect_fn(image):
        image, shapes = detection_model.preprocess(image)
        prediction_dict = detection_model.predict(image, shapes)
        detections = detection_model.postprocess(prediction_dict, shapes)
        return detections

    category_index = label_map_util.create_category_index_from_labelmap(params["FILE"]["custom_label_map"])

    img = cv2.imread(image_path)
    image_np = np.array(img)

    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections = detect_fn(input_tensor)

    num_detections = int(detections.pop("num_detections"))
    detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
    detections["num_detections"] = num_detections

    # detection_classes should be ints
    detections["detection_classes"] = detections["detection_classes"].astype(np.int64)

    label_id_offset = 1
    image_np_with_detections = image_np.copy()

    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_detections,
        detections["detection_boxes"],
        detections["detection_classes"] + label_id_offset,
        detections["detection_scores"],
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=5,
        min_score_thresh=.8,
        agnostic_mode=False
    )

    plt.figure()
    plt.imshow(cv2.cvtColor(image_np_with_detections, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.savefig("output_figure.png", dpi=300, bbox_inches='tight')


def inference_real_time_camera(params, ckpt_name):
    config = config_util.get_configs_from_pipeline_file(params["FILE"]["custom_config"])
    detection_model = model_builder.build(model_config=config["model"], is_training=False)

    ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
    ckpt.restore(os.path.join(params["PATH"]["custom_model"], ckpt_name)).expect_partial()

    @tf.function
    def detect_fn(image):
        image, shapes = detection_model.preprocess(image)
        prediction_dict = detection_model.predict(image, shapes)
        detections = detection_model.postprocess(prediction_dict, shapes)
        return detections

    category_index = label_map_util.create_category_index_from_labelmap(params["FILE"]["custom_label_map"])

    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        image_np = np.array(frame)

        input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
        detections = detect_fn(input_tensor)

        num_detections = int(detections.pop("num_detections"))
        detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
        detections["num_detections"] = num_detections

        # detection_classes should be ints
        detections["detection_classes"] = detections["detection_classes"].astype(np.int64)

        label_id_offset = 1
        image_np_with_detections = image_np.copy()

        viz_utils.visualize_boxes_and_labels_on_image_array(
            image_np_with_detections,
            detections["detection_boxes"],
            detections["detection_classes"]+label_id_offset,
            detections["detection_scores"],
            category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=5,
            min_score_thresh=.8,
            agnostic_mode=False
        )

        cv2.imshow("object detection", cv2.resize(image_np_with_detections, (800, 600)))

        if cv2.waitKey(10) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break


def main(args):
    # Define some constants
    PARAMS = initialize_params(args)

    ### TRAIN MODE
    if args.is_train:
        # Build model
        create_label_maps(PARAMS["IMAGE_CLASSES"], PARAMS["FILE"]["custom_label_map"])
        create_tf_record(script=PARAMS["FILE"]["TF_record_generator"],
                         image_train_path=os.path.join(PARAMS["PATH"]["images"], "train"),
                         image_test_path=os.path.join(PARAMS["PATH"]["images"], "test"),
                         label_map_path=PARAMS["FILE"]["custom_label_map"],
                         tfrecord_train_path=PARAMS["FILE"]["custom_train_tf_record"],
                         tfrecord_test_path=PARAMS["FILE"]["custom_test_tf_record"])
        create_custom_pipeline_config(PARAMS)

        # Train model
        train_loop(PARAMS)

    ### INFERENCE MODE
    else:
        ckpt_name = latest_ckpt(PARAMS["PATH"]["custom_model"])
        if args.inference_type == "single_image":       # Infer a single image
            inference_single_image(params=PARAMS, ckpt_name=ckpt_name,
                                   image_path=args.image_path)
        else:       # Real-time object detection using webcam
            inference_real_time_camera(params=PARAMS, ckpt_name=ckpt_name)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--is_train",
        help="Which phase to run [eg. train or inference]",
        action="store_true",
        default=False
    )
    parser.add_argument(
        "--inference_type",
        help="Which type of inference do you want to run? On single image or real time detection using webcam",
        choices=["single_image", "webcam"],
        default="webcam",
        type=str
    )
    parser.add_argument(
        "--image_path",
        help="Path to the image to run inference on",
        default=r".\dataset\images\test\ThumbUp_9.jpg",
        type=str
    )
    parser.add_argument(
        "--batch_size",
        help="Batch size to train model",
        default=4,
        type=int
    )
    parser.add_argument(
        "--num_steps",
        help="Number of steps to train model",
        default=2000,
        type=int
    )
    args = parser.parse_args()

    main(args)
    print("Finish!!")
