import os
import sys

import numpy as np
import torch
from scipy import ndimage

from src.dataprocessing.datapreparation import import_data
from src.dataprocessing.interpolation import interpolate_on_array
from src.utils.utilityfunctions import calculate_segmentation
from src.utils.plot import plot_prediction
from src.deep_learning import nets
from src.dataprocessing import preprocess
from src.visualization import setup


# Settings
EXPERIMENT = sys.argv[2]
SIDE = sys.argv[3]

MODEL_NAME = sys.argv[1]
MODEL_PATH = os.path.join('models')

PLOT_PATH = os.path.join('plots', MODEL_NAME, EXPERIMENT, SIDE, 'predictions')

model = nets.UNet(in_ch=2, out_ch=1, init_features=64, dropout_prob=0)


# Setup
setup = setup.Setup(data_path=os.path.join('data', EXPERIMENT, 'raw'),
                    experiment=EXPERIMENT,
                    side=SIDE)
# setup.set_stages([1000])  # uncomment to only plot specific examples
# setup.size = 70
setup.set_output_path(PLOT_PATH)

# Load model for predictions
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model_path = os.path.join(MODEL_PATH, MODEL_NAME, MODEL_NAME + '.pt')
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# Iterate over Nodemaps
for i, entry in setup.stages_to_nodemaps.items():

    # Import data from nodemap
    input_data, ground_truth_data = import_data(nodemaps={i: entry},
                                                data_path=setup.data_path,
                                                side=setup.side,
                                                exists_target=setup.target)

    # Interpolate input data on arrays of pixel size 256x256
    interp_size = setup.size if setup.side == 'right' else setup.size * -1
    interp_coors, interp_disps, interp_eps_vm = interpolate_on_array(input_by_nodemap=input_data,
                                                                     interp_size=interp_size)

    # Preprocess input
    disps = interp_disps[entry + '_' + setup.side]
    input_ch = torch.tensor(disps, dtype=torch.float32)
    input_ch = preprocess.normalize(input_ch).unsqueeze(0)

    if setup.target:
        # Preprocess ground truth
        target = torch.tensor(ground_truth_data[entry + '_' + setup.side].copy(),
                              dtype=torch.float32)
        label = preprocess.target_to_crack_tip_position(target)
    else:
        label = None

    # Make prediction
    out = model(input_ch.to(device))
    out = out.detach().to('cpu')

    # Simple crack tip prediction by taking the mean of all segmented pixels for each area
    crack_tip_seg = calculate_segmentation(out)

    is_crack_tip = np.where(out >= 0.5, 1, 0)
    labels, num_of_labels = ndimage.label(is_crack_tip)

    crack_tip_means = []
    for seg_label in range(1, num_of_labels + 1):
        seg = np.where(labels == seg_label, 1, 0)
        tips = calculate_segmentation(seg).numpy()
        crack_tip = np.mean(tips, axis=0)
        crack_tip_means.append(crack_tip)
    crack_tip_means = np.asarray(crack_tip_means, dtype=np.float32).reshape(-1, 2)

    # Plot and save
    print('Data will be plotted...')
    print(f'Plotting {i}/{sorted(setup.stages_to_nodemaps)[-1]}', end='\n\n')
    plot_title = f'Specimen: {setup.experiment} - Side: {setup.side} - Image: {i}'

    plot_prediction(background=interp_eps_vm[entry + '_' + setup.side] * 100,
                    interp_size=interp_size,
                    save_name=f'{i:04d}',
                    crack_tip_prediction=crack_tip_means,
                    crack_tip_seg=crack_tip_seg,
                    crack_tip_label=label,
                    f_min=0,
                    f_max=0.68,
                    title=plot_title,
                    path=PLOT_PATH,
                    label='von Mises strain [%]')
