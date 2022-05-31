"""

    Prepare the data for training.

    Input: Raw Nodemaps data (expected in sub folder 'Nodemaps' of NODEMAP_PATH)
    Output: Training data as list of torch.tensor arrays

"""


import os

import torch

from src.dataprocessing.datapreparation import import_data
from src.utils.utilityfunctions import get_nodemaps_and_stage_nums, numpy_to_tensor, dict_to_list
from src.dataprocessing.interpolation import interpolate_on_array


# Settings for the experiments
####################################################################################################
sizes = {
    'S_950_1.6': 450,  # mm
    'S_160_4.7': 70,
    'S_160_2.0': 70
}

target_exists = {
    'S_950_1.6': False,
    'S_160_4.7': True,
    'S_160_2.0': False
}

# Set paths and parameters
EXPERIMENT = 'S_160_4.7'
NODEMAP_PATH = os.path.join('data', EXPERIMENT, 'raw')
SAVE_PATH = os.path.join('data', EXPERIMENT, 'interim')

sides = ['left', 'right']
SIZE = sizes[EXPERIMENT]
PIXELS = 256
####################################################################################################

if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

# get dict of stage numbers with respective nodemap file names
stages_to_nodemaps, _ = get_nodemaps_and_stage_nums(os.path.join(NODEMAP_PATH, 'Nodemaps'))


for side in sides:
    # import
    inputs, ground_truths = import_data(nodemaps=stages_to_nodemaps,
                                        data_path=NODEMAP_PATH,
                                        side=side,
                                        exists_target=target_exists[EXPERIMENT])

    # interpolate
    interp_size = SIZE if side == 'right' else SIZE * -1
    _, interp_disps, _ = interpolate_on_array(input_by_nodemap=inputs,
                                              interp_size=interp_size,
                                              pixels=PIXELS)

    # get inputs
    inputs = interp_disps
    inputs = numpy_to_tensor(inputs, dtype=torch.float32)
    inputs = dict_to_list(inputs)

    # save inputs
    torch.save(inputs, os.path.join(SAVE_PATH, f'lInputData_{side}.pt'))

    if target_exists[EXPERIMENT]:
        # get targets
        targets = numpy_to_tensor(ground_truths, dtype=torch.int64)
        targets = dict_to_list(targets)

        # save targets
        torch.save(targets, os.path.join(SAVE_PATH, f'lGroundTruthData_{side}.pt'))
    print(f'Data saved for {side}-hand side.')
