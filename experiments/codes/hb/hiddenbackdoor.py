import numpy as np
from imagestamper import *
import torch
import torch.nn as nn


def abstractmethod(args):
    pass


class OptimizationModel(nn.Module):
    """Custom Pytorch model for gradient optimization.
    """

    def __init__(self, feature_extractor, target_images, layer_name: str):
        super().__init__()
        self.layer_name = layer_name
        self.feature_extractor = feature_extractor
        # initialize weights with random numbers
        target_images = torch.Tensor(target_images)
        # make weights torch parameters
        self.poison_images = nn.Parameter(target_images)

    def forward(self, x):
        return ((self.feature_extractor(self.poison_images)[self.layer_name]
                 - self.feature_extractor(x)[self.layer_name]) ** 2).sum()


class Attack:

    def __init__(self):
        return

    @abstractmethod
    def run(self):
        return


class HiddenBackdoor(Attack):
    """
    Implements Hidden Backdoor Attack
    """

    def __init__(self, x_train, y_train, source, target, patch, feature_extractor, layer_name, max_dist,
                 model_uses_channel_last=False):
        """
        x_train: training data features (np-array)
        y_train: training data labels (np-array)
        source: the source category
        target: the target category
        patch: the image you wish to be stamped on the source images (a.k.a the trigger)
        feature_extractor: a feature extractor made by torchvision (takes a tensor as input)
        layer_name: the name of the feature extractor layer
        max_dist: maximum tolerable distance between the poison and its corresponding patched image
                  in the feature space
        """
        self.target_images = None
        self.patched_source_images = None
        self.x_train = x_train
        self.y_train = y_train
        self.source = source
        self.target = target
        self.patch = patch
        self.feature_extractor = feature_extractor
        self.layer_name = layer_name
        self.max_dist = max_dist
        self.model_uses_channel_last = model_uses_channel_last
        self.model = None
        self.opt = None
        self.source_indices = [idx for idx, label in enumerate(self.y_train) if label == self.source]
        self.target_indices = [idx for idx, label in enumerate(self.y_train) if label == self.target]

    def sample_from_target(self, num_samples: int):
        chosen_indices = np.random.choice(self.target_indices, replace=False, size=num_samples)
        return self.x_train[chosen_indices].copy()

    def sample_from_source(self, num_samples: int):
        chosen_indices = np.random.choice(self.source_indices, replace=False, size=num_samples)
        return self.x_train[chosen_indices].copy()

    def project_onto(self, target_images):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        one_tensor = torch.ones_like(torch.Tensor(target_images.transpose(0, 3, 1, 2))).to(device)
        # print(self.model.poison_images.shape)
        # print(target_images.shape)
        projected_images = torch.clamp(self.model.poison_images,
                                       min=torch.Tensor(target_images.transpose(0, 3, 1, 2)).to(device) - one_tensor*self.max_dist,
                                       max=torch.Tensor(target_images.transpose(0, 3, 1, 2)).to(device) + one_tensor*self.max_dist)\
            .cpu().detach().numpy()
        return projected_images

    def report(self, target_images, patched_source_images):
        self.patched_source_images = patched_source_images
        self.target_images = target_images

    def run(self, num_poisons: int, learning_rate: float, epochs: int, min_loss: float):
        """
        num_poison: the number of poisons you wish to generate
        """
        target_images = self.sample_from_target(num_poisons)
        loss = 10000
        while loss > min_loss:
            source_images = self.sample_from_source(num_poisons)
            patched_source_images = random_bulk_stamper(source_images, self.patch)
            mapping = np.random.permutation(num_poisons)
            loss = self.perform_gradient_descent(target_images, patched_source_images, mapping, learning_rate, epochs)
            projected_poisons = self.project_onto(target_images)
            self.model.poison_images = torch.nn.Parameter(torch.Tensor(projected_poisons))
        num_dims = 3 if len(self.x_train[0].shape) == 3 else 2
        if num_dims == 3:
            projected_poisons = projected_poisons.transpose(0, 2, 3, 1)
            # target_images = target_images.transpose(0, 2, 3, 1)
            # patched_source_images = patched_source_images.transpose(0, 2, 3, 1)
        else:
            pass
            # TODO: if the image has 2 channels, is transpose needed?
        self.report(target_images, patched_source_images)
        return projected_poisons

    def perform_gradient_descent(self, target_images, patched_source_images, mapping, learning_rate=1, epochs=5):
        """
        returns:
            - loss: the loss for current poison generation process
        """
        num_dims = 3 if len(target_images[0].shape) == 3 else 2
        if not self.model_uses_channel_last:
            if num_dims == 3:
                target_images = target_images.transpose(0, 3, 1, 2)
                patched_source_images = patched_source_images.transpose(0, 3, 1, 2)[mapping]
            else:
                # TODO: if the data is 2D, is transpose needed?
                pass

        self.model = OptimizationModel(self.feature_extractor, target_images, self.layer_name)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = self.model.to(device)
        # self.opt = torch.optim.SGD(self.model.parameters(), lr=learning_rate)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08,
                                    weight_decay=0, amsgrad=False)
        for i in range(epochs):
            loss = self.model(torch.Tensor(patched_source_images).to(device))
            loss.backward()
            self.opt.step()
            self.opt.zero_grad()
            print(f'\r epoch {i} - loss:{loss}', end='')
        return loss
