import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import tri, cm
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D


def plot_prediction(background,
                    interp_size,
                    crack_tip_prediction=None,
                    crack_tip_seg=None,
                    crack_tip_label=None,
                    f_min=None,
                    f_max=None,
                    save_name=None,
                    path=None,
                    title='',
                    label='Plot of Data [Unit]'):
    """
    Plots crack tip labels and predictions over background

    :param background: (np.array) for background data
    :param interp_size: (float or int) actual size of background (can be negative!)
    :param save_name (str) Name under which the plot is saved (.png added automatically)
    :param crack_tip_prediction: (np.array or None) of size 1 x 2 - [:, 0] are y-coordinates!
    :param crack_tip_seg: (np.array or None) of size num_of_segmentations x 2
    :param crack_tip_label: (np.array or None) of size 1 x 2 - [:, 0] are y-coordinates!
    :param f_min: (float) minimal value of background data in plot (if None then auto-min)
    :param f_max: (float) maximal value of background data in plot (if None then auto-max)
    :param path: (str) location to save plot
    :param title: (str) of the plot
    :param label: (str) label for background data color bar
    """
    if f_min is None:
        f_min = background.min()
    if f_max is None:
        f_max = background.max()
        extend = 'neither'
    else:
        extend = 'max'

    num_colors = 120
    contour_vector = np.linspace(f_min, f_max, num_colors, endpoint=True)
    label_vector = np.linspace(f_min, f_max, 10, endpoint=True)
    # Colormap similar to Aramis
    cm_jet = cm.get_cmap('jet', 512)
    my_cmap = ListedColormap(cm_jet(np.linspace(0.1, 0.9, 256)))

    fig = plt.figure(1)
    ax = fig.add_subplot(111)

    pixels = background.shape[0]

    if interp_size >= 0:
        x_coor_interp = np.linspace(0, interp_size, pixels)
        y_coor_interp = np.linspace(-interp_size / 2.0, interp_size / 2.0, pixels)
    else:
        x_coor_interp = np.linspace(interp_size, 0, pixels)
        y_coor_interp = np.linspace(interp_size / 2.0, -interp_size / 2.0, pixels)
        background = np.fliplr(background)

    coor_x, coor_y = np.meshgrid(x_coor_interp, y_coor_interp)
    triang = tri.Triangulation(coor_x.flatten(), coor_y.flatten())

    mask = np.any(np.where(np.isnan(background.flatten())[triang.triangles], True, False), axis=1)
    triang.set_mask(mask)
    plot = ax.tricontourf(triang,
                          background.flatten(), contour_vector,
                          extend=extend, cmap=my_cmap)
    ax.autoscale(False)
    # ax.axis('off')  # uncomment to turn of axis and labels

    size = np.abs(interp_size)
    if crack_tip_seg is not None:
        # crack tip segmentation
        x_crack_tip_seg = crack_tip_seg[:, 1] * interp_size / pixels
        y_crack_tip_seg = crack_tip_seg[:, 0] * size / pixels - size / 2
        ax.scatter(x_crack_tip_seg, y_crack_tip_seg, color='gray', linewidths=1, marker='.')

    if crack_tip_label is not None:
        # actual crack tip label
        x_crack_tip_label = crack_tip_label[:, 1] * interp_size / pixels
        y_crack_tip_label = crack_tip_label[:, 0] * size / pixels - size / 2
        ax.scatter(x_crack_tip_label, y_crack_tip_label, color='black', linewidths=1, marker='x')

    if crack_tip_prediction is not None:
        # crack tip prediction
        x_crack_tip_pred = crack_tip_prediction[:, 1] * interp_size / pixels
        y_crack_tip_pred = crack_tip_prediction[:, 0] * size / pixels - size / 2
        ax.scatter(x_crack_tip_pred, y_crack_tip_pred, color='darkred', linewidths=1, marker='x')

    fig.colorbar(plot, ticks=label_vector, label=label)
    ax.set_title(title)
    ax.set_xlabel('x [mm]')
    ax.set_ylabel('y [mm]')

    ax.set_xlim(coor_x.min(), coor_x.max())
    ax.set_ylim(coor_y.min(), coor_y.max())

    # legend
    if crack_tip_label is not None:
        legend_elements = [Line2D([0], [0], marker='x', color='darkred', lw=0, label='Prediction'),
                           Line2D([0], [0], marker='x', color='black', lw=0, label='Ground Truth'),
                           Line2D([0], [0], marker='o', color='grey', lw=0, label='Segmentation')]
    else:
        legend_elements = [Line2D([0], [0], marker='x', color='darkred', lw=0, label='Prediction'),
                           Line2D([0], [0], marker='o', color='grey', lw=0, label='Segmentation')]
    ax.legend(handles=legend_elements)

    if path is not None:
        if not os.path.exists(path):
            os.makedirs(path)
        plt.savefig(os.path.join(path, save_name + '.png'), bbox_inches='tight', dpi=300)
    else:
        plt.show()

    plt.close()
