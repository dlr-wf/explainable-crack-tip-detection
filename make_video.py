"""

    Create videos from time-series plots.

    Instructions:   This script needs plotted predictions or visualizations
                    and makes videos out of these.

"""


# Imports
import os
import cv2


# Paths and Parameters
####################################################################################################
PLOT_PATH = os.path.join('plots')
VIDEO_PATH = os.path.join('videos')
PREFIX = ''
####################################################################################################

# Create video path and check if plot path exists
if not os.path.exists(PLOT_PATH):
    raise AssertionError("\nThe provided plot path does not exist!")
if not os.path.exists(VIDEO_PATH):
    os.mkdir(VIDEO_PATH)
    print("\nCreated video folder!")


# Walk through the plot directory until we find files
for subdir, dirs, files in os.walk(PLOT_PATH):
    if len(files) != 0:
        rel_path = os.path.relpath(subdir, PLOT_PATH)
        # take subdirectory as video name
        vid_name = rel_path.replace('/', '_')

        # get image's size
        img = cv2.imread(os.path.join(subdir, files[0]))
        height, width, layers = img.shape
        size = (width, height)

        # fetch images
        img_array = []
        for filename in sorted(files):
            img = cv2.imread(os.path.join(subdir, filename))

            height, width, layers = img.shape
            current_size = (width, height)
            if current_size != size:
                raise ValueError(f"\nThe images for the video {vid_name} have different sizes!")

            img_array.append(img)

        # make video
        # fourcc = cv2.VideoWriter_fourcc(*'DIVX')  # for Windows
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # for Linux

        video = cv2.VideoWriter(os.path.join(VIDEO_PATH, PREFIX + vid_name + '.mp4'), fourcc, fps=10,
                                frameSize=size)

        for img in img_array:
            video.write(img)
        video.release()
        print(f"Finished video {vid_name}")
