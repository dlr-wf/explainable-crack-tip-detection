# Imports
import os
from itertools import product

import torch
from torch import optim
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import Compose

from src.deep_learning.train import train
from src.docu.docu import Documentation
from src.dataprocessing import transforms
from src.dataprocessing.dataset import CrackTipDataset
from src.deep_learning import nets, loss


# Set parameters and paths
####################################################################################################
# Data
ORIGIN = os.path.join('data', 'S_160_4.7', 'interim')

DATA_PATH_TRAIN_INPUT = os.path.join(ORIGIN, 'lInputData_right.pt')
DATA_PATH_TRAIN_LABEL = os.path.join(ORIGIN, 'lGroundTruthData_right.pt')

DATA_PATH_VAL_INPUT = os.path.join(ORIGIN, 'lInputData_left.pt')
DATA_PATH_VAL_LABEL = os.path.join(ORIGIN, 'lGroundTruthData_left.pt')

# Model
MODEL_PATH = os.path.join('models')

# Training
NUM_EPOCHS = 10
####################################################################################################


def run(lr=1e-4, bs=16, size=64, dropout_prob=0):

    # Data transforms
    trsfs = {'train': Compose([transforms.EnhanceTip(),
                               transforms.RandomCrop(size=[120, 180], left=[10, 30]),
                               transforms.RandomRotation(degrees=10),
                               transforms.Resize(size=224),
                               transforms.RandomFlip(),
                               transforms.InputNormalization(),
                               transforms.CrackTipNormalization(),
                               transforms.ToCrackTipMasks()
                               ]),
             'val': Compose([transforms.EnhanceTip(),
                             transforms.InputNormalization(),
                             transforms.CrackTipNormalization(),
                             transforms.ToCrackTipMasks()
                             ])}

    # Datasets
    datasets = {'train': CrackTipDataset(inputs=DATA_PATH_TRAIN_INPUT, labels=DATA_PATH_TRAIN_LABEL,
                                         transform=trsfs['train']),
                'val': CrackTipDataset(inputs=DATA_PATH_VAL_INPUT, labels=DATA_PATH_VAL_LABEL,
                                       transform=trsfs['val'])}

    dataloaders = {
        'train': DataLoader(datasets['train'], batch_size=bs, shuffle=True, num_workers=4),
        'val': DataLoader(datasets['val'], batch_size=bs, shuffle=False, num_workers=4)
    }
    dataset_sizes = {x: len(datasets[x]) for x in ['train', 'val']}

    # Model Setup
    model = nets.UNet(in_ch=2, out_ch=1, init_features=size, dropout_prob=dropout_prob)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Training
    criterion = loss.DiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, amsgrad=True)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=1)

    model, train_docu = train(model, dataloaders, dataset_sizes, criterion,
                              optimizer, scheduler, NUM_EPOCHS, device)
    print(f'\nTraining Time: {train_docu.train_time:.0f} min')
    print(f'Loss: {train_docu.loss:.4f}')
    print(f'Deviation: {train_docu.deviation:.2f}')
    print(f'Reliability: {train_docu.reliability:.2f}')

    # Save the best model
    name = model.__class__.__name__ \
           + f'_l{train_docu.loss:.4f}_d{train_docu.deviation:.2f}_r{train_docu.reliability:.4f}'
    path = os.path.join(MODEL_PATH, name)
    if not os.path.exists(path):
        os.makedirs(path)
    torch.save(model.state_dict(), os.path.join(path, name + '.pt'))

    # documentation
    docu = Documentation(trsfs, datasets, dataloaders, model, criterion,
                         optimizer, scheduler, NUM_EPOCHS, train_docu)
    docu.save_metadata(path=path, name=name)


if __name__ == '__main__':

    # Hyperparameters
    params = dict(
        lr=[5e-4],
        bs=[16],
        size=[64],
        dropout_prob=[0.2]
    )

    NUM_RUNS = 1
    for run_id in range(NUM_RUNS):
        print(f'\nRun: {run_id+1}/{NUM_RUNS}')

        # Make one run for each parameter combination
        for current_params in product(*list(params.values())):

            run(*current_params)
