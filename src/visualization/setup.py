import os

import torch

from src.dataprocessing.datapreparation import import_data
from src.dataprocessing.interpolation import interpolate_on_array
from src.utils.utilityfunctions import numpy_to_tensor, get_nodemaps_and_stage_nums


class Experiments:
    """Settings for the currently used experimental data."""

    def __init__(self):
        """Dictionaries with experiment names as keys are initialized."""

        self.nodemap_nums = {
            'S_950_1.6': [str(i) for i in range(100, 304, 2)],  # at maximum force
            'S_160_4.7': [str(i) for i in range(7, 842, 5)],
            'S_160_2.0': [str(i) for i in range(20, 1430, 5)]
        }
        self.sizes = {
            'S_950_1.6': 450,  # mm
            'S_160_4.7': 70,
            'S_160_2.0': 70
        }
        self.exists_target = {
            'S_950_1.6': False,
            'S_160_4.7': True,
            'S_160_2.0': False
        }


class Setup(Experiments):
    """Setup Class to setup general settings for data, models, etc."""

    def __init__(self,
                 data_path,
                 experiment='S_160_2.0',
                 side='left'
                 ):
        """
        Initializes default setup class and sets data path attribute.

        :param data_path: (str) with sub-folders 'Nodemaps' and 'GroundTruth' (if exists)
        :param experiment: (str) name of the experiment (needs to be in Experiments)
        :param side: (str) 'left' or 'right'
        """
        super().__init__()

        # experiment, model and paths
        self.experiment = experiment
        self.side = side

        if os.path.exists(data_path):
            self.data_path = data_path
        else:
            raise ValueError("The chosen data path does not seem to exist!")

        self.output_path = None
        self.size = self.sizes[self.experiment] if self.experiment in self.sizes else None
        self.target = self.exists_target[self.experiment] if self.experiment in self.exists_target else None

        self.model_name = None
        self.model_path = None

        self.visu_layers = None

        self.out_ids = None

        if self.experiment in self.nodemap_nums:
            self.stages_to_nodemaps, self.nodemaps_to_stages = \
                get_nodemaps_and_stage_nums(os.path.join(self.data_path, 'Nodemaps'),
                                            self.nodemap_nums[self.experiment])
        else:
            self.stages_to_nodemaps, self.nodemaps_to_stages = {}, {}

    def load_data(self):
        """Loads data from information given in setup.
        Returns inputs & targets as dictionaries sorted by nodemap keys.

        :returns: inputs (dict of torch.tensors) of shape (B, C, H, W),
                  targets (dict of torch.tensors) of shape (B, H, W) or None,
                  with nodemap_name + _side as keys
        """
        # read test data
        input_data, ground_truth = import_data(nodemaps=self.stages_to_nodemaps,
                                               data_path=self.data_path,
                                               side=self.side,
                                               exists_target=self.target)
        # interpolate
        interp_size = self.size if self.side == 'right' else self.size * -1
        _, interp_disps, _ = interpolate_on_array(input_by_nodemap=input_data,
                                                  interp_size=interp_size,
                                                  pixels=256)
        # get inputs
        inputs = interp_disps
        inputs = numpy_to_tensor(inputs, dtype=torch.float32)

        # get targets (if exists)
        targets = numpy_to_tensor(ground_truth, dtype=torch.int64) if self.target else None

        return inputs, targets

    def set_model(self, model_path: str, model_name: str):
        """Set the model path and name of this setup."""
        if os.path.exists(model_path):
            self.model_path = model_path
        else:
            raise ValueError("The model path does not seem to exist!")

        self.model_name = model_name

    def set_stages(self, stages):
        """
        Change stages different from the default setup.

        :param stages: (list of str of ints or 'All') e.g. [7, 35, 100]
        """
        if stages != 'All':
            stages = [str(i) for i in stages]
        self.stages_to_nodemaps, self.nodemaps_to_stages = \
            get_nodemaps_and_stage_nums(os.path.join(self.data_path, 'Nodemaps'), stages)

    def set_visu_layers(self, layers):
        """Set layers for visualization with Grad-CAM, Seg-Grad-CAM, etc.

        :param layers: (list) of layer names as str, e.g. ['down4', 'base', 'up1'] or ['layer4']
        """
        self.visu_layers = layers

    def set_output_path(self, path):
        """Sets the output path for the setup and creates the folder (if it not exists).

        :param path: (str) Output path to be set.
        """
        self.output_path = path

        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
