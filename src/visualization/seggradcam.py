import os

import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import tri
from matplotlib.lines import Line2D

from src.deep_learning import nets
from src.utils.utilityfunctions import calculate_segmentation


class UNetWithHooks(nets.UNet):
    """
    UNet with hooks registered during forward pass.
    This is done to save the gradients during backpropagation.
    Features can be registered and are then saved as well during forward pass.
    """
    def __init__(self, in_ch=2, out_ch=1, init_features=64, dropout_prob=0):
        super().__init__(in_ch, out_ch, init_features, dropout_prob)

        self.feature_modules = {}
        self.features = {}

        self.gradients_list = []
        self.gradients = {}

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        _ = x2.register_hook(self._save_gradients)  # gradient hook
        x3 = self.down2(x2)
        _ = x3.register_hook(self._save_gradients)  # gradient hook
        x4 = self.down3(x3)
        _ = x4.register_hook(self._save_gradients)  # gradient hook
        x5 = self.down4(x4)
        _ = x5.register_hook(self._save_gradients)  # gradient hook

        x6 = self.base(x5)
        _ = x6.register_hook(self._save_gradients)  # gradient hook

        x = self.up1(x6, x4)
        _ = x.register_hook(self._save_gradients)  # gradient hook
        x = self.up2(x, x3)
        _ = x.register_hook(self._save_gradients)  # gradient hook
        x = self.up3(x, x2)
        _ = x.register_hook(self._save_gradients)  # gradient hook
        x = self.up4(x, x1)
        _ = x.register_hook(self._save_gradients)  # gradient hook
        x = self.outc(x)

        return x  # no sigmoid!

    def _save_gradients(self, grad):
        self.gradients_list.append(grad.data.numpy())

    def _save_features(self, m, i, output):
        self.features[m] = output.data.numpy()

    def gradients_to_dict(self):
        """save gradients from 'gradients_list' to 'gradients_dict'"""
        names = ['up4', 'up3', 'up2', 'up1', 'base', 'down4', 'down3', 'down2', 'down1']
        for i, name in enumerate(names):
            self.gradients[name] = self.gradients_list[i]

    def clear_grad_list(self):
        """clears the gradient list and dict of the model"""
        self.gradients_list = []
        self.gradients = {}

    def register_features(self, names):
        """adds forward hook after module[name] to catch features in forward pass"""
        for name in names:
            # register forward hooks at feature modules
            m = dict(self.named_modules())[name]
            self.feature_modules[name] = m
            _ = m.register_forward_hook(self._save_features)

    def clear_feature_modules(self):
        """clears the feature modules of the model"""
        self.feature_modules = {}


class ParallelNetsWithHooks(nets.ParallelNets):
    """
    ParallelNets with hooks registered during forward pass of UNet.
    This is done to save the gradients during backpropagation.
    Features can be registered and are then saved as well during forward pass.
    """
    def __init__(self, in_ch=2, out_ch=1, init_features=64, dropout_prob=0):
        super().__init__(in_ch, out_ch, init_features, dropout_prob)

        self.unet = UNetWithHooks(in_ch=in_ch, out_ch=out_ch,
                                  init_features=init_features, dropout_prob=dropout_prob)

    def forward(self, x):
        self.unet(x)
        return x


class SegGradCAM:
    """
    This class implements Grad-CAM for semantic segmentation tasks as described in
    Natekar et al. 2020 (see https://doi.org/10.3389/fncom.2020.00006) or
    Vinogradova et al. (see https://arxiv.org/pdf/2002.11434.pdf)
    """

    def __init__(self, setup, model, feature_modules=None, mask=False):
        """
        :param setup: (Setup-Class object)
        :param model: (NN model) e.g. UNet, ParallelNets
        :param feature_modules: (list[str] or None) e.g. ['down4', 'base', 'up1']
        :param mask: (bool) if True, a segmentation mask is multiplied by the output (see Vinogradova et al.)
        """
        self.setup = setup
        self.model = model
        self.mask = mask

        if feature_modules is None:
            # use feature modules provided by 'setup'
            layers = self.setup.visu_layers
            if layers is not None:
                self.feature_modules = layers
            else:
                raise ValueError("No visualization layers specified in setup.")

        else:
            self.feature_modules = feature_modules

        if isinstance(self.feature_modules, str):
            # make list out of single feature module
            self.feature_modules = [self.feature_modules]

        self.model.register_features(self.feature_modules)

    def __call__(self, input_t):
        """calculate grad cam for segmentation model"""
        self.model.eval()

        # forward pass
        output = self.model(input_t)

        ones = torch.ones(output.size()).requires_grad_(True)
        if self.mask:
            # build mask
            is_seg = torch.BoolTensor(output >= 0)
            mask = torch.where(is_seg, 1, 0)
            mask = mask * ones
        else:
            # fill with ones
            mask = ones

        # global (average) pooling of the output
        score = torch.mean(mask * output)

        # backpropagation
        self.model.zero_grad()
        self.model.clear_grad_list()
        score.backward(retain_graph=True)
        self.model.gradients_to_dict()

        # calculate activation mapping
        seg_cam = self._calculate_cam(size=input_t.shape[-2:])

        return output, seg_cam

    def _calculate_cam(self, size):
        cam = np.zeros(size, dtype=np.float32)

        for name in self.feature_modules:

            m = self.model.feature_modules[name]
            feats = self.model.features[m]
            grads = self.model.gradients[name]
            weights = np.mean(grads, axis=(-2, -1))[0, :]  # global average pooling of gradients

            current_cam = np.zeros(feats.shape[-2:], dtype=np.float32)
            for i, weight in enumerate(weights):
                current_cam += weight * feats[0, i, :, :]

            cam += cv2.resize(current_cam, size)

        # clip negative values --> highlight only areas that positively contribute to segmentation
        cam = np.maximum(cam, 0)

        # flip left<->right (if side is left)
        if self.setup.side == 'left':
            cam = np.fliplr(cam)

        return cam

    def plot(self, output, heatmap, stage_num, scale='QUALITATIVE'):
        """
        Plot the model's Seg-Grad-CAM heatmap together with the model's segmentation output.

        :param stage_num: (int)
        :param output: (torch.tensor) output of model or application of Seg-Grad-CAM-forward
        :param heatmap: (np.array) Seg-Grad-CAM attention heatmap
        :param scale: (str) 'QUALITATIVE' or 'QUANTITATIVE'
        :return: fig (plt.figure) of the plot
        """
        # setup color vectors and normalize heatmap if necessary
        num_colors = 120

        if scale == 'QUALITATIVE':
            # normalize heatmaps to [0, 1]
            heatmap = heatmap - np.nanmin(heatmap)
            if np.nanmax(heatmap) != 0:
                heatmap = heatmap / np.nanmax(heatmap)
            # set contour and label vector
            contour_vector = np.linspace(0, 1, num_colors)
            label_vector = ['Low', 'High']

        elif scale == 'QUANTITATIVE':
            contour_vector = np.linspace(np.nanmin(heatmap), np.nanmax(heatmap), num_colors)
            label_vector = np.linspace(np.nanmin(heatmap), np.nanmax(heatmap), 2)

        else:
            raise AssertionError("Parameter 'scale' must be 'QUALITATIVE' for qualitative/percent "
                                 "scale or 'QUANTITATIVE' for quantitative scale of heatmaps")
        plt.clf()

        # initialize figure
        fig = plt.figure(1)
        plot_title = f'Specimen: {self.setup.experiment} - ' \
                     f'Side: {self.setup.side} - Image: {stage_num}'
        fig.suptitle(plot_title)
        ax = fig.add_subplot(111)

        # get coordinates to interpolate heatmap on
        pixels = heatmap.shape[0]

        if self.setup.side == 'right':
            interp_coor_x = np.linspace(0, self.setup.size, pixels)
            interp_coor_y = np.linspace(-self.setup.size / 2.0, self.setup.size / 2.0, pixels)
        else:
            interp_coor_x = np.linspace(-self.setup.size, 0, pixels)
            interp_coor_y = np.linspace(-self.setup.size / 2.0, self.setup.size / 2.0, pixels)

        coor_x, coor_y = np.meshgrid(interp_coor_x, interp_coor_y)

        # plot heatmap
        triang = tri.Triangulation(coor_x.flatten(), coor_y.flatten())
        mask = np.any(np.where(np.isnan(heatmap.flatten())[triang.triangles], True, False),
                      axis=1)
        triang.set_mask(mask)
        plot = ax.tricontourf(triang,
                              heatmap.flatten(), contour_vector,
                              extend='neither', cmap='jet')
        ax.autoscale(False)
        # ax.axis('off')  # uncomment to turn off axis labels and ticks

        # calculate crack tip segmentation
        tip_seg = calculate_segmentation(torch.sigmoid(output))
        signed_size = self.setup.size if self.setup.side == 'right' else -self.setup.size
        x_tip_seg = tip_seg[:, 1] * signed_size / pixels
        y_tip_seg = tip_seg[:, 0] * self.setup.size / pixels - self.setup.size / 2
        # plot crack tip segmentation
        ax.scatter(x_tip_seg, y_tip_seg, color='grey', alpha=0.5, linewidths=1, marker='.')

        # plot color bar and setup axis labels and ticks
        cbar = fig.colorbar(plot, ticks=[0, 1], label='Attention', format='%.0f')
        cbar.ax.set_yticklabels(label_vector)
        ax.set_xlabel('x [mm]')
        ax.set_ylabel('y [mm]')
        ax.set_xlim(coor_x.min(), coor_x.max())
        ax.set_ylim(coor_y.min(), coor_y.max())

        # legend
        legend_elements = [Line2D([0], [0], marker='o', color='grey', lw=0, label='Segmentation')]
        ax.legend(handles=legend_elements)

        return fig

    def save(self, key, fig, subfolder=None):
        """Save figure 'fig' of stage 'key' in folder 'subfolder'."""
        stage_num = self.setup.nodemaps_to_stages[key]
        save_folder = os.path.join(self.setup.output_path)
        if subfolder is not None:
            save_folder = os.path.join(save_folder, subfolder)
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        plt.savefig(os.path.join(save_folder, f'{stage_num:04d}.png'), dpi=300)
        plt.close(fig)


def plot_overview(output, maps, side, title, scale='QUALITATIVE'):
    """Plot seg-grad-cam of several layers in one single figure."""
    # setup color bar settings and scaling of heatmaps
    num_colors = 120

    if scale == 'QUALITATIVE':
        # normalize heatmaps to [0, 1]
        maps_scaled = {}
        for key, cam in maps.items():
            cam = cam - np.nanmin(cam)
            if cam.max() != 0:
                cam = cam / np.nanmax(cam)
            maps_scaled[key] = cam
        # set contour and label vector
        f_min, f_max = 0, 1
        contour_vector = np.linspace(f_min, f_max, num_colors)
        label_vector = ['Low', 'High']

    elif scale == 'QUANTITATIVE':
        # calculate min and max along all heatmaps
        heatmaps_list = []
        for heatmap in maps.values():
            heatmaps_list.append(heatmap)
        heatmaps_array = np.asarray(heatmaps_list)
        f_min = np.nanmin(heatmaps_array)
        f_max = np.nanmax(heatmaps_array)
        if f_max - f_min == 0:
            f_max = f_min + 1
        maps_scaled = maps
        # set contour and label vector
        contour_vector = np.linspace(f_min, f_max, num_colors)
        label_vector = ['Low', 'High']

    else:
        raise AssertionError("Parameter 'scale' must be 'QUALITATIVE' for qualitative/percent "
                             "scale or 'QUANTITATIVE' for quantitative scale of heatmaps")
    plt.clf()

    # get coordinates to interpolate heatmap on
    pixels = next(iter(maps.values())).shape[-1]
    interp_coor_x = np.linspace(0, pixels, pixels)
    interp_coor_y = np.linspace(0, pixels, pixels)
    coor_x, coor_y = np.meshgrid(interp_coor_x, interp_coor_y)

    # initialize figure with subplots
    fig, axs = plt.subplots(3, 3, figsize=(12, 9))
    fig.suptitle(title, fontsize=20)

    for ax, (name, heatmap) in zip(axs.flat, maps_scaled.items()):
        ax.axis('off')
        ax.set_title(f'{name}')

        # plot heatmap
        plot = ax.tricontourf(coor_x.flatten(), coor_y.flatten(),
                              heatmap.flatten(), contour_vector,
                              extend='neither', cmap='jet')
        ax.autoscale(False)

        # calculate crack tip segmentation
        tip_seg = calculate_segmentation(torch.sigmoid(output))
        if side == 'left':
            tip_seg[:, 1] = pixels - tip_seg[:, 1]

        # plot crack tip segmentation
        ax.scatter(tip_seg[:, 1], tip_seg[:, 0], color='grey', alpha=0.5, linewidths=1, marker='.')

    # plot color bar
    cbar = fig.colorbar(plot, ax=axs, ticks=[f_min, f_max], label='Attention')
    cbar.ax.set_yticklabels(label_vector)

    return fig
