# Imports
import copy
import time

import numpy as np
import torch

from src.deep_learning import evaluate


class TrainDocu:
    """Wrapper for documenting the training."""
    def __init__(self):
        self.epoch = -1
        self.train_time = None

        # losses
        self.loss = 10e10
        self.dice_loss = None
        self.mse_loss = None

        # metrics
        self.deviation = None
        self.reliability = None


def train(model, dataloaders, dataset_sizes, criterion, optimizer,
          scheduler=None, num_epochs=10, device='cpu'):
    """
    Train and evaluate the model. Print the results.

    :param model: (PyTorch model)
    :param dataloaders: (dict of datasets) with keys 'train' and 'val'
    :param dataset_sizes: (dict of ints) with keys 'train' and 'val'
    :param criterion: (e.g. MSELoss)
    :param optimizer: (e.g. SGD or Adam)
    :param scheduler: (object) of torch.optim.lr_scheduler
    :param num_epochs: (int)
    :param device: ('cpu' or 'cuda:0')
    :param tb: Tensorboard (SummaryWriter) or None
    :return: model (PyTorch model), train_docu (TrainDocu object)
    """
    starting_time = time.time()
    best_model_weights = None
    train_docu = TrainDocu()

    for epoch in range(num_epochs):
        print('-' * 20)
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 20)

        # each epoch has training and evaluation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_deviation = 0.0
            running_reliability = 0.0

            # iterate over data in batches
            for i, (inputs, labels) in enumerate(dataloaders[phase]):

                # move tensors to device (necessary for GPU-training)
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the gradients
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model.forward(inputs)

                    loss = criterion(outputs, labels)
                    deviations = evaluate.get_deviation(outputs.to('cpu'), labels.to('cpu'))
                    deviation = np.sum(deviations)
                    reliability = evaluate.get_reliability(outputs.to('cpu'), labels.to('cpu'))

                    # backward loss and update weights only in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    running_loss += loss.item() * inputs.shape[0]
                    running_deviation += deviation
                    running_reliability += reliability * inputs.shape[0]

                    print(f'\r- Batch {i}/{int(dataset_sizes[phase] / inputs.shape[0])} '
                          f'- Loss: {loss.item():.4f} - Deviation: {deviation / inputs.shape[0]:.2f} '
                          f'- Reliability: {reliability:.4f}', end='')

            # make step with lr_scheduler
            if scheduler is not None:
                if phase == 'train':
                    scheduler.step()

            # print and save epoch loss and metrics to tensorboard
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_deviation = running_deviation / dataset_sizes[phase]
            epoch_reliability = running_reliability / dataset_sizes[phase]
            print(f'\r- Phase: {phase} - Loss: {epoch_loss:.4f} '
                  f'- Deviation: {epoch_deviation:.2f} - Reliability: {epoch_reliability:.4f}', end='\n')

            # save only the best model weights
            if phase == 'val' and epoch_loss < train_docu.loss:
                train_docu.epoch = epoch + 1
                train_docu.loss = epoch_loss
                train_docu.deviation = epoch_deviation
                train_docu.reliability = epoch_reliability
                best_model_weights = copy.deepcopy(model.state_dict())

    # load and return the best model
    if best_model_weights is not None:
        model.load_state_dict(best_model_weights)

    # save training time
    train_docu.train_time = (time.time() - starting_time) / 60.

    return model, train_docu


def train_parallel(model, dataloaders, dataset_sizes, criterion, optimizer,
                   scheduler=None, num_epochs=10, device='cpu'):
    """
    Train and evaluate the model. Print the results.

    :param model: (PyTorch model)
    :param dataloaders: (dict of datasets) with keys 'train' and 'val'
    :param dataset_sizes: (dict of ints) with keys 'train' and 'val'
    :param criterion: (e.g. MSELoss)
    :param optimizer: (e.g. SGD or Adam)
    :param scheduler: (object) of torch.optim.lr_scheduler
    :param num_epochs: (int)
    :param device: ('cpu' or 'cuda:0')
    :param tb: Tensorboard (SummaryWriter)
    :return: model (PyTorch model), train_docu (TrainDocu object)
    """
    starting_time = time.time()
    best_model_weights = None
    train_docu = TrainDocu()
    train_docu.dice_loss = 10e10

    weight_dice, weight_mse = criterion[2]

    for epoch in range(num_epochs):
        print('-' * 20)
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 20)

        # each epoch has training and evaluation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            # initialize running losses and metrics
            running_loss = 0.0
            running_dice = 0.0
            running_mse = 0.0
            running_deviation = 0.0
            running_reliability = 0.0

            # iterate over data in batches
            for i, (inputs, labels) in enumerate(dataloaders[phase]):

                # move tensors to device (necessary for GPU-training)
                inputs = inputs.to(device)
                labels = labels[0].to(device), labels[1].to(device)

                # zero the gradients
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model.forward(inputs)

                    # calculate loss
                    dice_loss = criterion[0](outputs[0], labels[0])
                    mse_loss = criterion[1](outputs[1], labels[1])
                    loss = weight_dice * dice_loss + weight_mse * mse_loss

                    # calculate deviation and reliability
                    deviations = evaluate.get_deviation(outputs[0].to('cpu'),
                                                        labels[0].to('cpu'))
                    deviation = np.sum(deviations)
                    reliability = evaluate.get_reliability(outputs[0].to('cpu'),
                                                           labels[0].to('cpu'))

                    # backward loss and update weights only in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    running_dice += dice_loss * inputs.shape[0]
                    running_mse += mse_loss * inputs.shape[0]
                    running_loss += loss.item() * inputs.shape[0]

                    running_deviation += deviation
                    running_reliability += reliability * inputs.shape[0]

                    print(f'\r- Batch {i}/{int(dataset_sizes[phase] / inputs.shape[0])} '
                          f'- Loss: {loss.item():.4f} - Dice Loss: {dice_loss:.4f} - MSE Loss: {mse_loss:.6f} '
                          f'- Deviation: {deviation / inputs.shape[0]:.2f} - Reliability: {reliability:.4f}', end='')

            # make step with lr_scheduler
            if scheduler is not None:
                if phase == 'train':
                    scheduler.step()

            # print and save epoch loss to tensorboard
            epoch_dice = running_dice / dataset_sizes[phase]
            epoch_mse = running_mse / dataset_sizes[phase]
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_deviation = running_deviation / dataset_sizes[phase]
            epoch_reliability = running_reliability / dataset_sizes[phase]
            print(f'\r- Phase: {phase} - Loss: {epoch_loss:.4f} '
                  f'- Dice Loss: {epoch_dice:.4f} - MSE Loss: {epoch_mse:.6f} '
                  f'- Deviation: {epoch_deviation:.2f} - Reliability: {epoch_reliability:.4f}', end='\n')

            # save only the best model weights
            if phase == 'val' and epoch_dice < train_docu.dice_loss:
                train_docu.epoch = epoch + 1
                train_docu.loss = epoch_loss
                train_docu.dice_loss = epoch_dice
                train_docu.mse_loss = epoch_mse
                train_docu.deviation = epoch_deviation
                train_docu.reliability = epoch_reliability
                best_model_weights = copy.deepcopy(model.state_dict())

    # load and return the best model
    if best_model_weights is not None:
        model.load_state_dict(best_model_weights)

    # save training time
    train_docu.train_time = (time.time() - starting_time) / 60.

    return model, train_docu
