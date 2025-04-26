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
    # Define some constants
    img_classes = ['ThumbUp', 'ThumbDown', 'ThankYou', 'LiveLong']
    num_imgs = 10
    IMAGE_PATH_MASTER = os.path.join(".", "dataset", "images")

    # Create Folder structures
    if os.name == 'nt':
        create_folders(img_classes)
    else:
        print("This OS is not Windows and not supported by this script!")
        return

    ### Capture images using webcam
    if args.type == "capture":
        capture_images(img_classes, num_imgs, IMAGE_PATH_MASTER)

    ### Splitting captured images into train and test set
    else:
        split_images(img_classes, num_imgs, IMAGE_PATH_MASTER, train=0.8,
                     train_path=os.path.join(IMAGE_PATH_MASTER, "train"),
                     test_path=os.path.join(IMAGE_PATH_MASTER, "test"))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--type",
        help="Do you want to capture new dataset or split existing dataset?",
        default="capture",
        choices=["capture", "split"],
        type=str
    )
    args = parser.parse_args()

    main(args)
    print("Finish!!!")
