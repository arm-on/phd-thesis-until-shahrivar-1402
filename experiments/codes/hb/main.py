import torchvision.datasets as datasets
import torchvision.models as models
from PIL import Image
import numpy as np
import importlib
import sys
import torch
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names
from img_resize import bulk_resize
import torch.nn as nn
from trigger import *
import cv2
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

config = {
    'random_seed':50,
    'path':'/home/user01/repo/experiments/codes/hb/',
    'data_path':'../../../../temp-data',
    'trigger_shape':(4,4,3),
    'return_nodes':{'features.7':'features.7'},
    'feature_layer':'features.7',
    'num_poisons':100,
    'num_triggers':10,
    'pois_learning_rate':1.0,
    'pois_epochs':100000,
    'pois_lr_decay':0.95,
    'pois_decay_step':2000,
    'pois_min_loss':1.5,
    'pois_epsilon':16/255,
    'pois_source_class':4,
    'pois_target_class':5,
    'pois_image_shape':(32, 32),
    'ft_batch_size':32,
    'ft_learning_rate':0.1,
    'ft_num_epochs':150
}

seed = config['random_seed']
np.random.seed(seed)
torch.random.manual_seed(seed)

sys.path.append(config['path'])

device = 'cuda' if torch.cuda.is_available() else 'cpu'

trainset = datasets.CIFAR10(root=config['data_path'], train=True, download=True, transform=None)

data_x = trainset.data.copy()
data_x = np.array([cv2.resize(np.uint8(img), config['pois_image_shape']) for img in data_x])
data_y = trainset.targets[:]

gen = TriggerGenerator(config['trigger_shape'])
triggers = gen.generate(config['num_triggers'])

model = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', weights='AlexNet_Weights.IMAGENET1K_V1')

## fixing the weights of the model
for param in model.parameters():
    param.requires_grad = False
    
feature_ext = create_feature_extractor(model, return_nodes=config['return_nodes'])

def feature_extractor(x: torch.Tensor):
  return feature_ext(torch.Tensor(x.transpose(0,3,1,2)))[config['feature_layer']]

from hiddenbackdoor import *

hb = HiddenBackdoor(data_x/255, data_y, config['pois_source_class'], config['pois_target_class'],
                    triggers[0]/255, feature_ext,
                    config['feature_layer'], config['pois_epsilon'])

poison = hb.run(num_poisons=config['num_poisons'], learning_rate=config['pois_learning_rate'],
                epochs=config['pois_epochs'], lr_decay=config['pois_lr_decay'],
                lr_decay_step=config['pois_decay_step'], min_loss=config['pois_min_loss'])

class TransferModel(nn.Module):
    def __init__(self, base_model, layer_name: str):
        super().__init__()
        self.layer_name = layer_name
        self.base_model = base_model
        self.classifier = nn.Linear(in_features=384, out_features=1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        out = self.base_model(x)[self.layer_name].squeeze()
        out = self.classifier(out)
        out = self.sigmoid(out)
        return out
    
transfer_model = TransferModel(feature_ext, config['feature_layer']).to(device)

# for param in transfer_model.parameters():
#     param.requires_grad = True

finetune_ids = []
for idx, _y in enumerate(data_y):
    if _y in {config['pois_source_class'], config['pois_target_class']}:
        finetune_ids.append(idx)
        
ft_x = data_x[finetune_ids].transpose(0, 3, 1, 2)
ft_y = np.array(data_y)[finetune_ids]
ft_x = np.concatenate((ft_x, poison.transpose(0, 3, 1, 2)), axis=0)
ft_y = np.concatenate((ft_y,
                       np.array([config['pois_target_class'] for i in range(config['num_poisons'])])), axis=0)

label_mapping = {
    config['pois_source_class']:0,
    config['pois_target_class']:1
}
ft_y = np.array([label_mapping[label] for label in ft_y])

class MyVectorDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = np.array(labels).reshape(-1, 1)
    def __len__(self):
        return self.features.shape[0]
    def __getitem__(self, idx):
        return torch.Tensor(self.features[idx]).to(device), torch.Tensor(self.labels[idx]).to(device)
    
train_dataset = MyVectorDataset(ft_x, ft_y)

train_dataloader = DataLoader(train_dataset, batch_size=config['ft_batch_size'], shuffle=True)

loss_fn = nn.BCELoss()
# optimizer = torch.optim.SGD(transfer_model.parameters(), lr=config['ft_learning_rate'])
optimizer = torch.optim.Adam(transfer_model.parameters(),lr=config['ft_learning_rate'],betas=(0.9,0.999),eps=1e-08,weight_decay=0,amsgrad=False)

def output_to_label(out):
    out = out.squeeze()
    dist_to_0 = abs(out)
    dist_to_1 = abs(out-1)
    if dist_to_0 <= dist_to_1:
        return 0
    else:
        return 1
    
def train_loop(dataloader, model, loss_fn, optimizer, epoch_num):
    num_points = len(dataloader.dataset)
    for batch, (features, labels) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(features)
        loss = loss_fn(pred, labels)

        # Backpropagation
        optimizer.zero_grad() # sets gradients of all model parameters to zero
        loss.backward() # calculate the gradients again
        optimizer.step() # w = w - learning_rate * grad(loss)_with_respect_to_w

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(features)
            print(f"\r Epoch {epoch_num} - loss: {loss:>7f}  [{current:>5d}/{num_points:>5d}]", end=" ")


def test_loop(dataloader, model, loss_fn, epoch_num, name):
    num_points = len(dataloader.dataset)
    sum_test_loss, correct = 0, 0

    with torch.no_grad():
        for _, (features, labels) in enumerate(dataloader):
            pred = model(features)
            sum_test_loss += loss_fn(pred, labels).item() # add the current loss to the sum of the losses
            # convert the outputs of the model on the current batch to a numpy array
            pred_lst = list(pred.cpu().detach().numpy().squeeze())
            pred_lst = [output_to_label(item) for item in pred_lst]
            # convert the original labels corresponding to the current batch to a numpy array
            output_lst = list(labels.cpu().detach().numpy().squeeze())
            # determine the points for which the model is correctly predicting the label (add a 1 for each)
            match_lst = [1 if p==o else 0 for (p, o) in zip(pred_lst, output_lst)]
            # count how many points are labeled correctly in this batch and add the number to the overall count of the correct labeled points
            correct += sum(match_lst)

    sum_test_loss /= num_points
    correct /= num_points
    print(f"\r Epoch {epoch_num} - {name} Error: Accuracy: {(100*correct):>0.1f}%, Avg loss: {sum_test_loss:>8f}")
    
for epoch_num in range(1, config['ft_num_epochs']+1):
    train_loop(train_dataloader, transfer_model, loss_fn, optimizer, epoch_num)
    
model_out = transfer_model(torch.Tensor(ft_x).to(device)).squeeze().cpu().detach().numpy()

pred_labels = [output_to_label(pred) for pred in model_out]
print((pred_labels==ft_y).sum())

