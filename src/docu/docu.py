import os
from datetime import datetime


def count_parameters(model):
    """Returns the number of trainable parameters of 'model'."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class Documentation:
    """Wrapper class for documentation of neural network training."""

    def __init__(self,
                 transforms=None,
                 datasets=None,
                 dataloaders=None,
                 model=None,
                 criterion=None,
                 optimizer=None,
                 lr_scheduler=None,
                 num_epochs=10,
                 train_docu=None):

        self.transforms = transforms
        self.datasets = datasets
        self.dataloaders = dataloaders
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.num_epochs = num_epochs
        self.train_docu = train_docu

    def save_metadata(self, path, name):
        """
        Save the training Meta-Data in a text-file together with the produced trained model.

        :param path: (str) e.g. 'my_model_folder'
        :param name: (str) e.g. 'my_model_docu'
        """
        time = datetime.now().strftime('%d/%m/%Y %H:%M:%S')
        batch_size = self.dataloaders['train'].batch_size if self.dataloaders is not None else None
        optimizer = self.lr_scheduler.optimizer.__class__.__name__ if self.lr_scheduler is not None else None
        optim_params = self.lr_scheduler.optimizer.defaults if self.lr_scheduler is not None else None
        lr_scheduler = self.lr_scheduler.__class__.__name__ if self.lr_scheduler is not None else None
        data_train = self.datasets['train'].data_paths_input if self.datasets is not None else None
        data_val = self.datasets['val'].data_paths_input if self.datasets is not None else None
        nbr_model_params = count_parameters(self.model) if self.model is not None else None
        train_time = self.train_docu.train_time if self.train_docu is not None else None
        epoch = self.train_docu.epoch if self.train_docu is not None else None
        loss = self.train_docu.loss if self.train_docu is not None else None
        deviation = self.train_docu.deviation if self.train_docu is not None else None
        reliability = self.train_docu.reliability if self.train_docu is not None else None

        if not os.path.exists(os.path.join(path)):
            os.makedirs(os.path.join(path))

        with open(os.path.join(path, name + '.txt'), mode='w') as metafile:
            metafile.write(
                '#'*100 + '\n'
                f'# Current time: {time}\n'
                f'# Training time: {train_time} min\n'
                f'# Model name: {name}\n'
                f'# Trainable parameters: {nbr_model_params}\n'
                f'# Epochs: {self.num_epochs}\n'
                f'# Best epoch: {epoch}\n'
                f'# Loss: {loss}\n'
                f'# Deviation: {deviation}\n'
                f'# Reliability: {reliability}\n'
                +
                '#'*100 + '\n'
                f'# Data path (train): {data_train}\n'
                f'# Data path (val): {data_val}\n'
                +
                '#'*100 + '\n'
                f'# Criterion: {self.criterion}\n'
            )
            try:  # not every model criterion has weights!
                metafile.write(
                    f'# Class Weights: {self.criterion.weight}\n'
                )
            except AttributeError:
                pass

            metafile.write(
                '#'*100 + '\n'
                f'# Optimizer: {optimizer}\n'
                f'#  - {optim_params}\n'
                f'#  - batch_size: {batch_size}\n'
                f'# Schedule: {lr_scheduler}\n'
            )

            if self.transforms is not None:
                metafile.write(
                    '#'*100 + '\n'
                    '# Data augmentation (train):\n'
                )
                for i, transform in enumerate(self.transforms['train'].transforms):
                    metafile.write(
                        f'#  {i+1}) {transform.__class__.__name__} ({vars(transform)})\n'
                    )

                metafile.write(
                    '#'*100 + '\n'
                    '# Data augmentation (val):\n'
                )
                for i, transform in enumerate(self.transforms['val'].transforms):
                    metafile.write(
                        f'#  {i+1}) {transform.__class__.__name__} ({vars(transform)})\n'
                    )

            if self.model is not None:
                metafile.write(
                    '#'*100 + '\n'
                    f'# Model: {self.model}\n' +
                    '#'*100 + '\n'
                )
