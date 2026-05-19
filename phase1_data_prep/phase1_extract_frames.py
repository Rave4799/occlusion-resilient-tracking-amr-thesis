#!/usr/bin/env python3
# phase1_extract_frames.py
# Source ROS Noetic environment before running this script.
# Example: source /opt/ros/noetic/setup.bash && python3 phase1_extract_frames.py

import argparse  # import argparse to parse command line arguments
import os  # import os to manipulate filesystem paths and directories
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
import sys  # import sys to inspect environment variables and exit on fatal errors
import traceback  # import traceback to print exception details when a bag fails

try:
    import rosbag  # import rosbag Python API for reading ROS bag files
    import numpy as np  # import numpy to convert raw image bytes to arrays
    import cv2  # import OpenCV to save images as JPEG files
except ImportError as exc:  # catch import errors if ROS environment is not sourced or dependencies are missing
    print('ERROR: ROS Noetic environment may not be sourced or required Python packages are unavailable.')
    print('Source /opt/ros/noetic/setup.bash before running this script.')
    print('Import error details:', exc)
    sys.exit(1)

# define the default bag root path where ROS bag files are stored
DEFAULT_BAG_ROOT = '/media/rad4799/RAD/ROS BAG Files/'
# define the default output directory for extracted training images
DEFAULT_OUTPUT_DIR = str(BASE_DIR / 'dataset' / 'train' / 'images')
# define the ROS image topic to read from each bag
DEFAULT_TOPIC = '/camera/color/image_raw'
# define the frame extraction step: save every 3rd frame
FRAME_STEP = 3
# define JPEG quality level for saved images
JPEG_QUALITY = 85


def check_ros_environment():
    """Check whether the ROS Noetic environment is sourced and warn if not."""
    ros_distro = os.environ.get('ROS_DISTRO', '')  # read ROS_DISTRO from environment variables
    if ros_distro.lower() != 'noetic':  # if the distro is not noetic, print a warning for the user
        print('WARNING: ROS_DISTRO is not set to noetic. Found:', ros_distro)
        print('Please source /opt/ros/noetic/setup.bash before running this script.')


def find_bag_files(root_path):
    """Recursively search for .bag files under the given root path."""
    for dirpath, _, filenames in os.walk(root_path):  # walk the directory tree recursively
        for filename in filenames:  # iterate over each file in the current directory
            if filename.lower().endswith('.bag'):  # check whether the file has a .bag extension
                yield os.path.join(dirpath, filename)  # yield the absolute path to the bag file


def parse_date_folder(bag_path, bag_root):
    """Extract a normalized date folder identifier from the bag path."""
    relative_path = os.path.relpath(bag_path, bag_root)  # compute the path of the bag relative to the root
    parts = relative_path.split(os.sep)  # split the relative path into components
    if parts:  # if there is at least one component in the path
        raw_date = parts[0]  # assume the first component is the date folder
        return raw_date.replace('_', '')  # normalize the date folder by removing underscores
    return 'unknown'  # return a fallback identifier if parsing fails


def make_image_name(date_folder, bag_name, sequence_index):
    """Build a standardized image filename from the bag metadata."""
    safe_bag_name = bag_name.replace(' ', '_')  # replace spaces in the bag name with underscores
    return f'{date_folder}_{safe_bag_name}_{sequence_index:06d}.jpg'  # format the filename with six-digit padding


def decode_ros_image(msg):
    """Decode raw ROS image message bytes into an OpenCV BGR image."""
    height = getattr(msg, 'height', 0)  # get image height from the message
    width = getattr(msg, 'width', 0)  # get image width from the message
    encoding = getattr(msg, 'encoding', '').lower()  # get image encoding and normalize to lowercase
    step = getattr(msg, 'step', 0)  # get row step size in bytes from the message
    data = msg.data  # raw byte buffer containing image pixels

    if not isinstance(data, (bytes, bytearray)):  # verify raw image data is present
        print('WARNING: Message data is not raw bytes, skipping frame.')
        return None  # cannot decode non-bytes data

    try:
        array = np.frombuffer(data, dtype=np.uint8)  # interpret raw bytes as an array of unsigned bytes
    except Exception as exc:  # catch errors while converting the raw buffer
        print('WARNING: Failed to convert raw image bytes to numpy array:', exc)
        return None  # return None when conversion fails

    if height <= 0 or width <= 0 or step <= 0:  # validate image dimensions and row stride
        print('WARNING: Invalid image dimensions or step, skipping frame.')
        return None  # cannot decode invalid image metadata

    expected_bytes = height * step  # expected total number of bytes in the image buffer
    if array.size < expected_bytes:  # if the raw buffer is too short, skip the frame
        print('WARNING: Raw image data length does not match expected dimensions, skipping frame.')
        return None  # mismatch indicates corrupted or incomplete data

    if encoding == 'bgr8':  # direct BGR bytes already match OpenCV format
        try:
            image = array.reshape((height, step))[:, :width * 3].reshape((height, width, 3))  # reshape using step and width
            return image  # return the decoded BGR image
        except Exception as exc:  # catch reshape errors for corrupted buffers
            print('WARNING: Failed to reshape BGR image buffer:', exc)
            return None  # skip on reshape failure

    if encoding == 'rgb8':  # RGB bytes need conversion to BGR for OpenCV saving
        try:
            image = array.reshape((height, step))[:, :width * 3].reshape((height, width, 3))  # reshape using step and width
            return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # convert RGB to OpenCV BGR format
        except Exception as exc:  # catch reshape or color conversion errors
            print('WARNING: Failed to decode RGB image buffer:', exc)
            return None  # skip on failure

    if encoding == 'mono8':  # grayscale image can be saved as-is with OpenCV
        try:
            image = array.reshape((height, step))[:, :width].reshape((height, width))  # reshape grayscale bytes
            return image  # return single-channel image
        except Exception as exc:  # catch reshape errors
            print('WARNING: Failed to decode mono8 image buffer:', exc)
            return None  # skip on failure

    print(f'WARNING: Unsupported ROS image encoding: {encoding}. Skipping frame.')  # warn for unsupported encodings
    return None  # unsupported encodings are not decoded


def extract_frames_from_bag(bag_path, bag_root, output_dir, topic, frame_step):
    """Extract RGB frames from a single bag and save them as JPEG images."""
    bag_name = os.path.splitext(os.path.basename(bag_path))[0]  # derive the bag name without the extension
    date_folder = parse_date_folder(bag_path, bag_root)  # extract the date folder for filename prefix
    total_frames = 0  # counter for total frames encountered in the bag
    saved_frames = 0  # counter for total frames saved to disk

    try:
        bag = rosbag.Bag(bag_path, 'r')  # open the bag file in read-only mode
    except Exception as exc:  # catch errors when opening a corrupted or unreadable bag
        print(f'WARNING: Failed to open bag: {bag_path}')  # warn about the failed bag
        print('Exception:', exc)  # print the exception
        return 0, 0  # return zeros to indicate no frames were processed

    print(f'Starting bag: {bag_name}')  # print the name of the bag as soon as processing begins
    try:
        for _, msg, _ in bag.read_messages(topics=[topic]):  # iterate through image messages on the target topic
            total_frames += 1  # increment the total frame counter for each message
            if (total_frames - 1) % frame_step != 0:  # only process every Nth frame according to the step
                continue  # skip frames that do not match the stepping criteria

            cv_image = decode_ros_image(msg)  # decode the raw ROS image message into an OpenCV image
            if cv_image is None:  # if decoding failed, skip the frame and continue
                print(f'WARNING: Skipping unreadable image in bag {bag_path} at frame {total_frames}')
                continue  # continue processing the next frame

            image_name = make_image_name(date_folder, bag_name, saved_frames + 1)  # build the output image filename
            output_path = os.path.join(output_dir, image_name)  # compute the full output file path

            try:
                success = cv2.imwrite(output_path, cv_image, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])  # write the image to disk as JPEG
                if not success:  # if OpenCV reports failure, warn and skip this frame
                    print(f'WARNING: Failed to save image {output_path}')
                    continue  # do not increment saved_frames when saving fails
            except Exception as write_exc:  # catch unexpected errors during file writing
                print(f'WARNING: Exception while saving image {output_path}')
                print('Exception:', write_exc)  # print the exception details
                continue  # continue processing remaining images

            saved_frames += 1  # increment saved frame count after a successful save
            if saved_frames % 100 == 0:  # print progress every 100 saved frames
                print(f'Progress for bag {bag_name}: saved {saved_frames} frames so far')
    except Exception as exc:  # catch exceptions from reading messages or iteration
        print(f'WARNING: Error while processing bag {bag_path}')  # warn about the processing error
        print('Exception traceback:')  # annotate the traceback printout
        traceback.print_exc()  # print the full exception traceback
    finally:
        bag.close()  # close the bag file to release file handles and resources

    print(f'Processed bag: {bag_name} | total frames: {total_frames} | saved frames: {saved_frames}')  # print per-bag progress summary
    return total_frames, saved_frames  # return counts for accumulation in the main loop


def main():
    """Main entry point for the frame extraction script."""
    parser = argparse.ArgumentParser(description='Extract every 3rd RGB frame from ROS bag files.')  # create an argument parser for CLI options
    parser.add_argument('--bag-root', default=DEFAULT_BAG_ROOT, help='Root folder containing ROS bag files to search recursively.')  # define bag root CLI argument
    parser.add_argument('--output-dir', default=DEFAULT_OUTPUT_DIR, help='Output folder where extracted JPEG images will be saved.')  # define output directory CLI argument
    parser.add_argument('--topic', default=DEFAULT_TOPIC, help='ROS image topic to read from each bag file.')  # define ROS topic CLI argument
    parser.add_argument('--step', type=int, default=FRAME_STEP, help='Frame step for extraction, e.g. every 3rd frame.')  # define frame step CLI argument
    args = parser.parse_args()  # parse command line arguments from sys.argv

    check_ros_environment()  # warn the user if ROS Noetic is not sourced

    os.makedirs(args.output_dir, exist_ok=True)  # ensure the output directory exists before saving images

    grand_total_saved = 0  # initialize the total saved frame counter across all bags
    bag_files = list(find_bag_files(args.bag_root))  # collect all bag file paths under the search root
    if not bag_files:  # if no bag files were found, print a warning and exit gracefully
        print('No .bag files found under', args.bag_root)  # warn the user about the missing files
        return  # exit the script without error

    print(f'Found {len(bag_files)} bag(s) under {args.bag_root}')  # print the number of discovered bag files

    for bag_path in bag_files:  # iterate over each discovered bag file path
        bag_name = os.path.splitext(os.path.basename(bag_path))[0]  # derive the bag name without the extension
        date_folder = parse_date_folder(bag_path, args.bag_root)  # extract the date folder for the bag prefix
        bag_prefix = f'{date_folder}_{bag_name}_'  # expected image filename prefix for this bag
        existing_files = [f for f in os.listdir(args.output_dir) if f.lower().endswith('.jpg')]  # list existing output JPEGs
        if any(f.startswith(bag_prefix) or f.startswith(f'{bag_name}_') for f in existing_files):  # skip if any output file from this bag already exists
            print(f'Skipping {bag_name} - already extracted')  # print skip message for interrupted reruns
            continue  # move to the next bag without processing

        _, saved_frames = extract_frames_from_bag(bag_path, args.bag_root, args.output_dir, args.topic, args.step)  # process the bag and get saved frame count
        grand_total_saved += saved_frames  # accumulate saved images into the grand total

    print('Grand total frames saved:', grand_total_saved)  # print the overall saved frame count after all bags are processed


if __name__ == '__main__':  # only execute the main function when the script is run directly
    main()  # call the main entry point
