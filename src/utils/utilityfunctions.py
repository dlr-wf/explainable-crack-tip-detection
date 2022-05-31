import os

import torch


def get_nodemaps_and_stage_nums(folder_path, which='All'):
    """
    Generates two dictionaries with stage numbers as keys and nodemap filenames as values
    and vice versa.
    """
    if which == 'All':
        list_of_filenames = os.listdir(folder_path)
        which = [name.split('_')[-1][:-4] for name in list_of_filenames]
    assert isinstance(which, (list, range)), 'Argument "which" should be a list of integers or "All".'

    nodemap_without_num = '_'.join(os.listdir(folder_path)[0].split('_')[:-1])
    stage_num_to_filename = {}
    filename_to_stage_num = {}
    for stage in which:
        num = int(stage)
        name = nodemap_without_num + f'_{num}.txt'
        stage_num_to_filename[num] = name
        filename_to_stage_num[name] = num
        filename_to_stage_num[name + '_left'] = num
        filename_to_stage_num[name + '_right'] = num

    return stage_num_to_filename, filename_to_stage_num


def calculate_segmentation(output_mask):
    """
    Calculates the crack tip positions of all segmented pixels of an output mask.
    :param output_mask: (torch.tensor) of shape (.. x H x W)
    :return: crack_tip_seg (torch.tensor) of shape num_of_seg x 2
    """
    condition = torch.BoolTensor(output_mask >= 0.5)
    is_crack_tip = torch.where(condition, 1, 0)
    crack_tip_seg = torch.nonzero(is_crack_tip, as_tuple=False)[:, -2:] / 1.
    return crack_tip_seg


def numpy_to_tensor(numpy_dict, dtype):
    """Convert a dict of numpy arrays into a dict of 'unsqueezed' tensors of 'dtype'."""
    return {key: torch.tensor(value.copy(), dtype=dtype).unsqueeze(0)
            for key, value in numpy_dict.items()}


def dict_to_list(dictionary):
    """Convert a dictionary into a list by loosing the keys."""
    return [value for key, value in dictionary.items()]
