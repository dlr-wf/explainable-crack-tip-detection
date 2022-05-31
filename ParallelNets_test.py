"""

    Tests for ParallelNets model for crack tip predictions.

    1) Deviation/Accuracy: How far from the label is the prediction?
    2) Reliability: How often does the network fail in providing any prediction?
    3) Dice coefficient

    Instructions: Needs trained UNet model and interim-data

"""

# Imports
import os

import numpy as np
import torch

from torchvision.transforms import Compose
from torch.utils.data import DataLoader

from src.dataprocessing.dataset import CrackTipDataset
from src.deep_learning import nets, loss
from src.dataprocessing import transforms
from src.deep_learning.evaluate import get_deviation


# Utility functions
def calculate_deviation(model, dataloader):
    devi = []

    for sample in dataloader:
        inputs, targets = sample['input'], sample['target']

        inputs = inputs.to(device)
        outputs = model(inputs)
        outputs = outputs[0].detach().to('cpu')

        condition = torch.BoolTensor(targets == 2)
        labels = torch.where(condition, 1, 0)
        labels = labels.unsqueeze(1)

        devi_i = get_deviation(outputs, labels)
        devi.append(devi_i)

    return np.concatenate(devi)


def calculate_reliability(model, dataloader, dataset):
    unpredicted = 0

    for sample in dataloader:
        inputs = sample['input'].to(device)
        outputs = model(inputs)
        outputs = outputs[0].detach().to('cpu')

        condition = torch.BoolTensor(outputs >= 0.5)
        is_crack_tip = torch.where(condition, 1, 0)

        for i in range(outputs.shape[0]):
            prediction_i = torch.nonzero(is_crack_tip[i], as_tuple=False)[:, -2:] / 1.

            if len(prediction_i) == 0:
                unpredicted += 1

    return 1. - unpredicted / len(dataset)


def calculate_dice(model, dataloader, criterion, dataset):
    running_loss = 0.0

    for sample in dataloader:
        inputs = sample['input'].to(device)
        targets = sample['target'].to('cpu')

        condition = torch.BoolTensor(targets == 2)
        labels = torch.where(condition, 1, 0)
        labels = labels.unsqueeze(1)

        outputs = model(inputs)
        outputs = outputs[0].detach().to('cpu')

        current_loss = criterion(outputs, labels)
        running_loss += current_loss.item() * inputs.shape[0]

    dice_loss = running_loss / len(dataset)
    return 1. - dice_loss


# Load the model
MODEL_PATH = os.path.join('models')
MODEL_NAME = 'ParallelNets-1'
model = nets.ParallelNets(in_ch=2, out_ch=1, init_features=64, dropout_prob=0.2)
criterion = loss.DiceLoss()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model_path = os.path.join(MODEL_PATH, MODEL_NAME, MODEL_NAME + '.pt')
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()


# Deviation and Dice coefficent
####################################################################################################
input_data_path = [
    os.path.join('data', 'S_160_4.7', 'interim', 'lInputData_left.pt'),
    os.path.join('data', 'S_160_4.7', 'interim', 'lInputData_right.pt')
]
label_data_path = [
    os.path.join('data', 'S_160_4.7', 'interim', 'lGroundTruthData_left.pt'),
    os.path.join('data', 'S_160_4.7', 'interim', 'lGroundTruthData_right.pt')
]

# Dataset and dataloader
trsfms = Compose([transforms.EnhanceTip(),
                  transforms.InputNormalization()])
dataset = CrackTipDataset(inputs=input_data_path, labels=label_data_path, transform=trsfms)
dataloader = DataLoader(dataset, batch_size=10, shuffle=False, num_workers=4)

# Calculate and print results
deviations_in_px = calculate_deviation(model, dataloader)
deviations_in_mm = deviations_in_px * 70 / 256
print(f'Deviation (mean) in mm: {np.mean(deviations_in_mm)}')
print(f'Deviation (std) in mm: {np.std(deviations_in_mm)}')
dice = calculate_dice(model, dataloader, criterion, dataset)
print(f'Dice coefficient: {dice}')


# Reliability
####################################################################################################
input_data_paths = [
    os.path.join('data', 'S_160_4.7', 'interim', 'lInputData_left.pt'),
    os.path.join('data', 'S_160_4.7', 'interim', 'lInputData_right.pt'),
    os.path.join('data', 'S_160_2.0', 'interim', 'lInputData_left.pt'),
    os.path.join('data', 'S_160_2.0', 'interim', 'lInputData_right.pt'),
    os.path.join('data', 'S_950_1.6', 'interim', 'lInputData_left.pt'),
    os.path.join('data', 'S_950_1.6', 'interim', 'lInputData_right.pt')
]

# Dataset and dataloader
trsfms = Compose([transforms.InputNormalization()])
dataset = CrackTipDataset(inputs=input_data_paths, transform=trsfms)
dataloader = DataLoader(dataset, batch_size=10, shuffle=False, num_workers=4)

reliability = calculate_reliability(model, dataloader, dataset)
print(f'Reliability in %: {reliability * 100}')
