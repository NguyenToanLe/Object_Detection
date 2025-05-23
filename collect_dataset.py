import os
import time
import cv2
import shutil


def create_folders(img_classes):
    print("Creating folders......")
    for img_class in img_classes:
        path = os.path.join(".", "dataset", "images", img_class)
        if not os.path.exists(path):
            os.makedirs(path)
    time.sleep(1)
    print("Folders created!")


def show_webcam():
    print("Openning webcam......")
    webcam = cv2.VideoCapture(0)
    # if webcam.isOpened():  # Try to get the first frame
    #     rval, frame = webcam.read()
    # else:
    #     rval = False
    # while rval:
    while webcam.isOpened():
        # cv2.imshow("Webcam", frame)
        # rval, frame = webcam.read()
        _, frame = webcam.read()
        cv2.imshow("Webcam", frame)

        # key = cv2.waitKey(20)
        # if key % 256 == 27:  # ESC pressed
        if cv2.waitKey(10) & 0xFF == ord('q'):      # q pressed
            print("Closing webcam......")
            webcam.release()
            cv2.destroyAllWindows()
            break

    webcam.release()
    cv2.destroyAllWindows()


def get_img_class_info(img_path):
    img_classes = []
    img_count = 0
    for f in os.listdir(img_path):
        class_path = os.path.join(img_path, f)
        if os.path.isdir(class_path):
            img_classes.append(f)

            img_count += len([name for name in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, name))])

    return img_classes, img_count



def capture_images(img_classes, num_imgs, path_master):
    # Test position
    print("Test position of webcam")
    show_webcam()

    for img_class in img_classes:
        webcam = cv2.VideoCapture(0)
        for num in range(num_imgs):
            print(f'Capturing image {num+1} of class {img_class}......')
            time.sleep(2)
            _, frame = webcam.read()
            cv2.imshow("Frame", frame)
            image_name = os.path.join(path_master, img_class, f'{img_class}_{num+1}.jpg')
            cv2.imwrite(image_name, frame)
            time.sleep(1)
            if cv2.waitKey(10) & 0xFF == ord('q'):  # q pressed
                webcam.release()
                cv2.destroyAllWindows()
                break
        print("-----------------------------------------")
    webcam.release()
    cv2.destroyAllWindows()


def split_images(img_classes, num_imgs, path_master, train, train_path, test_path):
    # Training dataset
    if not os.path.exists(train_path):
        os.makedirs(train_path)
    if train < 1:
        num_train_imgs = int(num_imgs*train)
    else:
        num_train_imgs = train

    for img_class in img_classes:
        files = os.listdir(os.path.join(path_master, img_class))
        train_files = files[:num_train_imgs*2]
        for file in train_files:
            shutil.move(os.path.join(path_master, img_class, file),
                        os.path.join(train_path, file))

    # Testing dataset
    if not os.path.exists(test_path):
        os.makedirs(test_path)

    for img_class in img_classes:
        test_files = os.listdir(os.path.join(path_master, img_class))
        for file in test_files:
            shutil.move(os.path.join(path_master, img_class, file),
                        os.path.join(test_path, file))

    # Deleting the parent path
    for img_class in img_classes:
        shutil.rmtree(os.path.join(path_master, img_class))


def main(args):
    ### Capture images using webcam
    if args.type == "capture":
        # Define some constants
        img_classes = args.img_classes
        num_imgs = args.num_imgs
        IMAGE_PATH_MASTER = args.img_path

        # Create Folder structures
        if os.name == 'nt':
            create_folders(img_classes)
        else:
            print("This OS is not Windows and not supported by this script!")
            return

        capture_images(img_classes, num_imgs, IMAGE_PATH_MASTER)

    ### Splitting captured images into train and test set
    else:
        IMAGE_PATH_MASTER = args.img_path

        img_classes, num_imgs = get_img_class_info(IMAGE_PATH_MASTER)

        split_images(img_classes, num_imgs, IMAGE_PATH_MASTER, train=args.train,
                     train_path=os.path.join(IMAGE_PATH_MASTER, "train"),
                     test_path=os.path.join(IMAGE_PATH_MASTER, "test"))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    ### Common args
    parser.add_argument(
        "--type",
        help="Do you want to capture new dataset or split existing dataset?",
        default="capture",
        choices=["capture", "split"],
        type=str
    )
    parser.add_argument(
        "--img_path",
        help="Where to save/get the images",
        default=os.path.join(".", "dataset", "images")
    )
    ### Args for capturing task
    parser.add_argument(
        "--img_classes",
        nargs='+',  # Accepts one or more arguments
        help="Which image class do you want to capture?",
        default=['ThumbUp', 'ThumbDown', 'ThankYou', 'LiveLong']
    )
    parser.add_argument(
        "--num_imgs",
        help="How many images per class do you want to capture?",
        default=10,
        type=int
    )
    ### Args for splitting task
    parser.add_argument(
        "--train",
        help="How much / Which ratio of the training images do you want to split?",
        default=0.8,
        type=float
    )
    args = parser.parse_args()

    main(args)
    print("Finish!!!")
