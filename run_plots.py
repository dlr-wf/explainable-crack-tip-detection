"""

    Run Prediction and Grad-CAM Visualization for several models and experiments.
    Afterwards create videos from plots.

"""

import os
import time


start_time = time.time()

# Settings
####################################################################################################
ParallelNets_models = [
    'ParallelNets-1'
]
UNet_models = [
    'UNet-1',
    'UNet-2'
]

experiments = [
    ('S_950_1.6', 'left'),
    ('S_950_1.6', 'right'),
    ('S_160_4.7', 'left'),
    ('S_160_4.7', 'right'),
    ('S_160_2.0', 'left'),
    ('S_160_2.0', 'right')
]
####################################################################################################

# Plots for ParallelNets models
for model in ParallelNets_models:
    for experiment, side in experiments:
        os.system(f"python ParallelNets_plot.py {model} {experiment} {side}")
        os.system(f"python ParallelNets_visualize.py {model} {experiment} {side}")

# Plots for UNet models
for model in UNet_models:
    for experiment, side in experiments:
        os.system(f"python UNet_plot.py {model} {experiment} {side}")
        os.system(f"python UNet_visualize.py {model} {experiment} {side}")

# Make videos
os.system('python make_video.py')

# Print time needed for script
print(f"\nThe hole script took {(time.time() - start_time) / 60. / 60.} hours.")
