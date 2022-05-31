import os
import sys

import torch
import matplotlib.pyplot as plt

from src.dataprocessing import preprocess
from src.visualization import setup, seggradcam


# Settings
EXPERIMENT = sys.argv[2]
SIDE = sys.argv[3]

MODEL_NAME = sys.argv[1]
MODEL_PATH = os.path.join('models')

PLOT_PATH = os.path.join('plots', MODEL_NAME, EXPERIMENT, SIDE, 'gradcam')

VISU_LAYERS = ['down1', 'down2', 'down3', 'down4', 'base', 'up1', 'up2', 'up3', 'up4']

setup = setup.Setup(data_path=os.path.join('data', EXPERIMENT, 'raw'),
                    experiment=EXPERIMENT,
                    side=SIDE)
# setup.set_stages([7])  # uncomment to only plot specific examples
setup.set_model(model_path=MODEL_PATH, model_name=MODEL_NAME)
setup.set_output_path(PLOT_PATH)
setup.set_visu_layers(VISU_LAYERS)

# Load the model
print('Loading model...')
model = seggradcam.ParallelNetsWithHooks()

model_path = os.path.join(setup.model_path, setup.model_name, setup.model_name + '.pt')
model.load_state_dict(torch.load(model_path))
model = model.unet

# Load Data
print('Loading data...')
inputs, targets = setup.load_data()

################################################################################################
# Segmentation Grad-CAM: overall network attention
sgc = seggradcam.SegGradCAM(setup, model)

print('\nPlotting Segmentation-Grad-CAM...')
# iterate over nodemap input_t samples
for key, input_t in inputs.items():
    print(f'\r{key}', end='')
    # calculate output and features in forward pass
    input_t = preprocess.normalize(input_t)
    output, heatmap = sgc(input_t)

    # plot and save heatmap
    stage_num = setup.nodemaps_to_stages[key]
    fig = sgc.plot(output, heatmap, stage_num)
    sgc.save(key, fig, subfolder='network')

################################################################################################
# Segmentation Grad-CAM: layer-wise attention
seg_grad_cams = {}
for name in setup.visu_layers:
    seg_grad_cams[name] = seggradcam.SegGradCAM(setup, model, feature_modules=name)

print('\nPlotting Segmentation-Grad-CAM...')
# iterate over nodemap input_t samples
for key, input_t in inputs.items():
    print(f'\r{key}', end='')
    # calculate output and features in forward pass
    input_t = preprocess.normalize(input_t)
    output = model(input_t)

    # calculate heatmaps
    heatmaps = {}
    for name, seg_grad_cam in seg_grad_cams.items():
        _, heatmap = seg_grad_cam(input_t)
        heatmaps[name] = heatmap

    # plot heatmap
    stage_num = setup.nodemaps_to_stages[key]
    specimen = setup.experiment
    plot_title = f'Specimen: {specimen} - Side: {setup.side} - Image: {stage_num}'
    fig = seggradcam.plot_overview(output=output,
                                   maps=heatmaps,
                                   side=setup.side,
                                   title=plot_title,
                                   scale='QUALITATIVE')

    # save
    save_folder = os.path.join(setup.output_path, 'layers')
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    plt.savefig(os.path.join(save_folder, f'{stage_num:04d}.png'), dpi=100)
    plt.close()
