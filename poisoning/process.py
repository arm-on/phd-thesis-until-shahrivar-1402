from typing import Dict
import numpy as np
from scipy.spatial import distance_matrix
from scipy.stats import mode
from collections import defaultdict
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
# import jax.numpy as jnp
# from jax import grad as jgrad, jacobian as jjacobian, hessian as jhessian
from art.attacks.poisoning import PoisoningAttackBackdoor
from art.attacks.poisoning.perturbations import add_pattern_bd, add_single_bd, insert_image
from scipy.optimize import minimize as scipy_minimize
from scipy.optimize import Bounds as scipy_bounds
from datascience.general import read_img, read_img_as_rgb, read_img_as_gray, resize_img, inverse_img, combine_single_channel_images, img_mixup, img_cutmix
import torch
from torchmetrics import HingeLoss
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from scipy.special import log_softmax, softmax
from scipy.stats import entropy
from matplotlib import pyplot as plt
from sklearn.metrics.pairwise import pairwise_distances
from scipy.spatial.distance import cdist
from tqdm import tqdm

class SVM(torch.nn.Module):
    def __init__(self, input_dim):
        super(SVM, self).__init__()
        self.input_dim = input_dim
        self.linear = torch.nn.Linear(in_features=self.input_dim, out_features=1, bias=True)
    def forward(self, x):
        output = self.linear(x)
        return output
class MyVectorDataset(Dataset):
    def __init__(self, features, labels, device):
        self.device = device
        self.features = features
        self.labels = np.array(labels).reshape(-1, 1)
    def __len__(self):
        return self.features.shape[0]
    def __getitem__(self, idx):
        return torch.Tensor(self.features[idx]).to(self.device), torch.Tensor(self.labels[idx]).to(self.device)
    
def my_hinge_loss(preds, targets):
    initial_value = 1-targets*preds
    hinge_value = torch.mean(torch.clamp(initial_value, min=0))
    return hinge_value

class SVM_trainer:
    '''
    Note: x should be flattened
    '''
    def train_loop(self, dataloader, model, loss_fn, optimizer):
        num_points = len(dataloader.dataset)
        for batch, (features, labels) in enumerate(dataloader):        
            # Compute prediction and loss
            pred = model(features)
            loss = loss_fn(pred, labels) + 0.5*(torch.norm(model.linear.weight.squeeze())**2 + torch.norm(model.linear.bias.squeeze())**2)
            # Backpropagation
            optimizer.zero_grad() # sets gradients of all model parameters to zero
            loss.backward() # calculate the gradients again
            optimizer.step() # w = w - learning_rate * grad(loss)_with_respect_to_w
    def __init__(self, x, y, batch_size, learning_rate, num_epochs):
        self.input_dim = x.shape[-1]
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        train_dataset = MyVectorDataset(x, y, self.device)
        self.train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.model = SVM(self.input_dim)
        self.model = self.model.to(self.device)
        self.loss_fn = my_hinge_loss
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
    def train(self):
        for epoch_num in range(1, self.num_epochs+1):
            self.train_loop(self.train_dataloader, self.model, self.loss_fn, self.optimizer)
        weights = self.model.linear.weight.squeeze().cpu().detach().numpy()
        bias = self.model.linear.bias.squeeze().cpu().detach().numpy()
        theta = np.concatenate((weights.reshape(-1,1), bias.reshape(-1,1))).squeeze()
        return theta

class attacker():
  '''
  Represents an attacker in a data poisoning attack
  
  '''
  def __init__(self):
    pass

  def return_aggregated_result(self):
    if self.method == 'modify':
      np.random.seed(self.seed)
      num_clean_examples = self.x_clean.shape[0]
      num_poison_examples = self.x_poison.shape[0]
      chosen_indices_to_be_removed = np.random.choice([i for i in range(num_clean_examples)], num_poison_examples, replace=False)
      chosen_indices_set = set(chosen_indices_to_be_removed)
      self.x_clean = np.array([self.x_clean[i] for i in range(num_clean_examples) if i not in chosen_indices_set])
      self.y_clean = np.array([self.y_clean[i] for i in range(num_clean_examples) if i not in chosen_indices_set])
    result = {}
    if self.x_clean.shape[1:] == self.x_poison.shape[1:]:
      result['x_train'] = np.concatenate((self.x_clean, self.x_poison), axis=0)
      result['y_train'] = np.concatenate((self.y_clean, self.y_poison), axis=0)
    elif len(self.y_poison) == 0:
      result['x_train'] = self.x_clean
      result['y_train'] = self.y_clean
    elif len(self.y_clean) == 0:
      result['x_train'] = self.x_poison
      result['y_train'] = self.y_poison
    result['is_poison'] = [False for i in range(self.x_clean.shape[0])] + [True for i in range(self.x_poison.shape[0])]
    return result

class SVM_KKT_attacker(attacker):
  '''
  Note: the labels should be either -1 or 1
  Usage example:
  >>> att = SVM_KKT_attacker(dataset.x_train, dataset.y_train, dataset.x_test, dataset.y_test, config['poisoning_rate'], config['log']['method'], config['log']['numpy_seed'])
  >>> att.find_decoy_params(80, 2) # 80 is the threshold percentage (the percentage of the bad testing points that are selected because of the high loss of the model when making a prediction about them)
  >>> att.attack()
  '''
  def __init__(self, x_clean, y_clean, x_test, y_test, epsilon, method='modify', desired_seed = 50, batch_size=32, learning_rate=0.01, num_epochs=2):
    unique_clean_labels = list(np.unique(y_clean))
    unique_test_labels = list(np.unique(y_test))
    assert unique_clean_labels == [-1,1] and unique_test_labels == [-1,1], 'the labels should be either -1 or 1'
    assert method in {'modify','add'}, "the attack method should be either 'modify' or 'add'"
    self.x_clean = x_clean
    self.y_clean = y_clean
    self.x_test = x_test
    self.y_test = y_test
    self.method = method
    self.seed = desired_seed
    self.batch_size = batch_size
    self.learning_rate=learning_rate
    self.num_epochs = num_epochs
    if self.method == 'add':
      self.epsilon = epsilon/(1-epsilon) if epsilon < 1.0 else epsilon # this is the corrected poisoning rate
    elif self.method == 'modify':
      self.epsilon = epsilon

  def attack(self, grid_search_size=10):
    scores = np.array([0.0 for t in range(grid_search_size+1)])
    results = []
    self.decoy_clean_grad = self.get_clean_gradient(self.decoy_params)
    for t in range(grid_search_size+1):
      epsilon_plus = (t*self.epsilon)/grid_search_size
      epsilon_minus = self.epsilon - epsilon_plus
      x_plus, x_minus = self.solve_equation(epsilon_plus, epsilon_minus)
      curr_result = {'x_plus':x_plus, 'x_minus':x_minus, 'epsilon_plus':epsilon_plus, 'epsilon_minus':epsilon_minus}
      results.append(curr_result)
      scores[t] = self.evaluate_test_loss(self.decoy_params)
    winner_idx = np.argmax(scores)
    winner_dict = results[winner_idx]
    # producing the poisoned examples
    num_clean = self.x_clean.shape[0]
    num_x_plus = int(epsilon_plus*num_clean)
    num_x_minus = int(epsilon_minus*num_clean)
    winner_x_plus = winner_dict['x_plus'].squeeze().reshape(1,-1)
    winner_x_minus = winner_dict['x_minus'].squeeze().reshape(1,-1)
    x_plus_repeats = np.repeat(winner_x_plus, num_x_plus, axis=0)
    x_minus_repeats = np.repeat(winner_x_minus, num_x_minus, axis=0)
    x_poison = np.concatenate((x_plus_repeats, x_minus_repeats), axis=0)
    y_plus = np.array([1 for i in range(num_x_plus)])
    y_minus = np.array([-1 for i in range(num_x_minus)])
    y_poison = np.concatenate((y_plus, y_minus), axis=0)
    self.x_poison = x_poison
    self.y_poison = y_poison


  def solve_equation(self, epsilon_plus, epsilon_minus, _lambda=0.001):
    def objective_fn(x):
      x_plus, x_minus = np.split(x, 2)
      grad_decoy = self.decoy_clean_grad
      theta_decoy = self.decoy_params
      vec = grad_decoy-epsilon_plus*np.append(x_plus,1.0)+epsilon_minus*np.append(x_minus,1.0)+_lambda*theta_decoy
      vec_norm = np.linalg.norm(vec)
      return vec_norm**2

    def x_plus_contrainst(x):
      theta_decoy = self.decoy_params
      x_plus, x_minus = np.split(x, 2)
      return 1-np.matmul(theta_decoy, np.append(x_plus, 1.0))

    def x_minus_contrainst(x):
      theta_decoy = self.decoy_params
      x_plus, x_minus = np.split(x, 2)
      return 1-np.matmul(theta_decoy, np.append(x_minus, 1.0))    

    x_p_c = {'type':'ineq', 'fun':x_plus_contrainst}
    x_m_c = {'type':'ineq', 'fun':x_minus_contrainst}
    constraint = [x_p_c, x_m_c]
    num_features = self.x_clean.shape[1]
    x_0 = np.random.rand(num_features*2)
    max_arr = np.repeat(self.x_clean.max(axis=0),2)
    min_arr = np.repeat(self.x_clean.min(axis=0),2)
    bounds = scipy_bounds(list(min_arr), list(max_arr))
    optimization_result = scipy_minimize(objective_fn, x_0, method='SLSQP',constraints=constraint,bounds=bounds)
    result_x = optimization_result.x
    res_x_plus, res_x_minus = np.split(result_x, 2)
    return res_x_plus, res_x_minus

  def evaluate_test_loss(self, theta, _lambda=0.0000001):
    num_test_examples = self.x_test.shape[0]
    regularization_term = _lambda*np.linalg.norm(theta)
    test_losses = np.array([self.hinge_loss(self.x_test[i], self.y_test[i], theta)+regularization_term for i in range(num_test_examples)])
    total_loss = test_losses.sum()
    return total_loss

  def find_decoy_params(self, threshold_percentage, num_repeats):
    theta_clean = self.train_svm(self.x_clean, self.y_clean).squeeze()
    num_test_examples = self.x_test.shape[0]
    test_losses = np.array([self.hinge_loss(self.x_test[i], self.y_test[i], theta_clean) for i in range(num_test_examples)])
    bad_indices = test_losses.argsort(axis=0)[-int(threshold_percentage/100*num_test_examples):]
    bad_test_examples_x = self.x_test[bad_indices].copy()
    bad_test_examples_y = self.y_test[bad_indices].copy()
    bad_test_examples_flipped_y = np.array([1 if _y == -1 else -1 for _y in bad_test_examples_y]) # we have to flip the labels of these instances
    x_bad = np.repeat(bad_test_examples_x, num_repeats, axis=0)
    y_bad = np.repeat(bad_test_examples_flipped_y, num_repeats, axis=0)
    x_decoy = np.concatenate((self.x_clean, x_bad), axis=0)
    y_decoy = np.concatenate((self.y_clean, y_bad), axis=0)
    theta_decoy = self.train_svm(x_decoy, y_decoy).squeeze()
    self.decoy_params = theta_decoy
    
  def train_svm(self, x, y):
    obj = SVM_trainer(x, y, self.batch_size, self.learning_rate, self.num_epochs)
    theta = obj.train()
    return theta

  def hinge_loss(self, x, y, theta):
    output = self.predict(x, theta)
    return max(0, 1-y*output.squeeze())

  def predict(self, x, theta):
    '''
    x: a numpy array of shape (num_features,)
    '''
    num_features = x.shape[0]
    W = theta[:num_features]
    b = theta[-1]
    return np.matmul(W, x) + b

  def label(self, x, theta):
    return 1.0 if self.predict(x, theta)>0.0 else -1.0

  def get_clean_gradient(self, theta, _lambda=0.0000001):
    '''
    Note: theta should be of shape (num_features+1, )
    '''
    clean_losses = np.array([self.hinge_loss(_x, _y, theta) for (_x, _y) in zip(self.x_clean, self.y_clean)])
    nonzero_loss_idxs = np.where(clean_losses>0)[0]
    nonzero_loss_idxs = set(nonzero_loss_idxs)
    num_clean_examples = self.x_clean.shape[0]
    all_grads = np.array([_lambda*theta if idx not in nonzero_loss_idxs else self.y_clean[idx]*np.append(self.x_clean[idx].reshape(1,-1),1.0) + _lambda*theta for idx in range(num_clean_examples)])
    num_clean = self.x_clean.shape[0]
    avg_grad = (all_grads.sum(axis=0))/num_clean
    return avg_grad


class SVM_influence_attacker(attacker):
  '''
  Note: for now, the attack works if the labels belong to {-1,1}

  Usage Example:
  >> att = SVM_influence_attacker(dataset.x_train, dataset.y_train, dataset.x_test, dataset.y_test, 0.2, 0.1, 'add')
  >> att.attack()
  >> result = att.return_aggregated_result()
  '''

  def __init__(self, x_clean, y_clean, x_test, y_test, attack_prob, step_size, method='modify', desired_seed=50, batch_size=32, learning_rate=0.01, num_epochs=2):
    '''
    theta: a numpy array of (num_features+1,) shape
    '''
    self.batch_size = batch_size
    self.learning_rate=learning_rate
    self.num_epochs = num_epochs
    unique_clean_labels = list(np.unique(y_clean))
    unique_test_labels = list(np.unique(y_test))
    assert unique_clean_labels == [-1,1] and unique_test_labels == [-1,1], 'the labels should be either -1 or 1'
    assert method in {'modify','add'}, "the attack method should be either 'modify' or 'add'"
    self.x_clean = x_clean
    self.y_clean = y_clean
    self.x_test = x_test
    self.y_test = y_test
    self.method = method
    if self.method == 'add':
      self.attack_prob = attack_prob/(1-attack_prob) if attack_prob < 1.0 else attack_prob
    elif self.method == 'modify':
      self.attack_prob = attack_prob
    self.step_size = step_size
    self.seed = desired_seed
  
  def simulate_l2_defense(self, x, y):
    d = defender()
    x, y, threshold = d.l2_defense(x, y, int(self.attack_prob*100), return_threshold=True)
    return x, y, threshold

  def simulate_no_defense(self, x, y):
    return x, y

  def project_onto_l2_ball(self, x, center, radius):
    '''
    Based on this answer: https://stackoverflow.com/a/9604279
    '''
    centered_x = x - center
    length = np.linalg.norm(centered_x)
    if length <= radius:
      return x
    else:
      _lambda = radius/length
      projection = _lambda*x + (1-_lambda)*center
      return projection

  def train_svm(self, x, y):
#     y = np.array([_y if _y==1 else 0 for _y in y])
    obj = SVM_trainer(x, y, self.batch_size, self.learning_rate, self.num_epochs)
    theta = obj.train()
    return theta

  def get_hessian_inv(self, x, y, theta, _lambda=0.0000001):
    num_params = theta.shape[0]
    num_examples = y.shape[0]
    hessian = np.eye(num_params)*_lambda+(np.eye(num_params)*(_lambda))
    hessian_inv = np.linalg.inv(hessian)
    return hessian_inv

  def hinge_loss(self, x, y, theta):
    output = self.predict(x, theta)
    return max(0, 1-y*output.squeeze())

  def get_grad_transpose(self, x, y, theta, _lambda=0.0000001):
    y_loss = np.array([self.hinge_loss(_x, _y, theta) for (_x, _y) in zip(x, y)])
    nonzero_loss_idxs = np.where(y_loss>0)[0]
    nonzero_loss_idxs = set(nonzero_loss_idxs)
    gradient = []
    num_params = theta.shape[0]
    num_examples = x.shape[0]
    for (j, theta_j) in enumerate(theta):
      grad_element = 0.0
      # for wrong_idx in wrong_pred_idxs:
      for idx in range(num_examples):
        if j != num_params-1:
          grad_element += theta_j*_lambda
          if idx in nonzero_loss_idxs:
            grad_element += (-y[idx])*x[idx][j]
        else:
          grad_element += theta_j*_lambda
          if idx in nonzero_loss_idxs:
            grad_element += (-y[idx])
      gradient.append(grad_element/num_examples)
    gradient_T = np.array(gradient).reshape(1,-1)
    return gradient_T

  def get_partial(self, _x, _y, theta):
    num_features = theta.shape[0] - 1
    _y_pred = self.predict(_x, theta)
    if self.hinge_loss(_x, _y, theta)>0.0:
      main_part = -np.eye(num_features)*_y
      last_row = np.zeros((1, num_features))
      partial = np.concatenate((main_part, last_row))
      return partial
    else:
      return np.zeros((num_features+1, num_features))

  def attack(self, num_iters=10):
    # find the centroid of the classes (it will be used for the final projection) - BEGIN
    d = defender()
    centroids = d.calculate_centroids(self.x_clean, self.y_clean)
    # find the centroid of the classes (it will be used for the final projection) - END
    np.random.seed(self.seed)
    num_clean = self.x_clean.shape[0]
    num_poison = int(self.attack_prob*num_clean)
    num_features = self.x_clean.shape[1]
    # initialize the poisonous examples
    x_poison = np.random.rand(num_poison, num_features)
    y_poison = np.random.choice([-1.0,1.0], num_poison, replace=True)
    # calculating the maximum distance to each centroid
    classes = list(centroids.keys())
    max_dist_to_centroids = {c:np.linalg.norm(self.x_clean[np.where(self.y_clean==c)]-centroids[c], axis=1).max() for c in classes}
    # outer loop
    for t in range(num_iters):
      print(f"\r iteration {t}", end=" ")
      # join clean and poisonous data
      x_all = np.concatenate((self.x_clean, x_poison), axis=0)
      y_all = np.concatenate((self.y_clean, y_poison), axis=0)
      # x_san, y_san, l2_threshold = self.simulate_l2_defense(x_all, y_all) # sanitize
      x_san, y_san = self.simulate_no_defense(x_all, y_all) # do not sanitize
      theta = self.train_svm(x_san, y_san)
      hessian_inv = self.get_hessian_inv(x_all, y_all, theta)
      grad_T = self.get_grad_transpose(self.x_test, self.y_test, theta)
      prod = np.matmul(grad_T,hessian_inv)
      # inner loop
      for idx, (_x, _y) in enumerate(zip(x_poison, y_poison)):
        initial_x = _x - self.step_size*np.matmul(prod, self.get_partial(_x, _y, theta))
        corresponding_centroid = centroids[_y] # find the centroid of the class 'initial_x' belongs to
        l2_threshold = max_dist_to_centroids[_y]
        proj_x = self.project_onto_l2_ball(initial_x, corresponding_centroid, l2_threshold) # to be replaced by the projection of initial_x onto the L2 ball
        # proj_x = initial_x[:] # no need to project
        x_poison[idx] = proj_x[:]
    self.x_poison = x_poison
    self.y_poison = y_poison
    

  def predict(self, x, theta):
    '''
    x: a numpy array of shape (num_features,)
    '''
    num_features = x.shape[0]
    W = theta[:num_features]
    b = theta[-1]
    return np.matmul(W, x) + b
  
  def label(self, x):
    return 1.0 if self.predict(x)>0.0 else -1.0
  
  

class LR_influence_attacker(attacker):
  '''
  LR: Logistic Regression
  Note: for now, the attack works if the labels belong to {0,1}

  Usage Example:
  >> att = LR_influence_attacker(dataset.x_train, dataset.y_train, dataset.x_test, dataset.y_test, 0.2, 0.1, 'add')
  >> att.attack(num_iters=1)
  >> result = att.return_aggregated_result()
  '''

  def __init__(self, x_clean, y_clean, x_test, y_test, attack_prob, step_size, method='modify', desired_seed=50):
    '''
    theta: a numpy array of (num_features+1,) shape
    '''
    unique_clean_labels = list(np.unique(y_clean))
    unique_test_labels = list(np.unique(y_test))
    assert unique_clean_labels == [0,1] and unique_test_labels == [0,1], 'the labels should be either 0 or 1'
    assert method in {'modify','add'}, "the attack method should be either 'modify' or 'add'"
    self.x_clean = x_clean
    self.y_clean = y_clean
    self.x_test = x_test
    self.y_test = y_test
    self.method = method
    if self.method == 'add':
      self.attack_prob = attack_prob/(1-attack_prob) if attack_prob < 1.0 else attack_prob
    elif self.method == 'modify':
      self.attack_prob = attack_prob
    self.step_size = step_size
    self.seed = desired_seed

  def gradient_wrt_params(self, x, y, theta):
    num_features = x.shape[0]
    theta = theta.squeeze()
    grad_vector = np.zeros((num_features+1,))
    c = np.matmul(np.append(x, [1]), theta)
    emc = np.exp(-c) # e to the power of minus c
    for i in range(num_features+1):
      if i != num_features:
        grad_vector[i] = -((x[i]*y*emc)/(emc+1)) - (x[i]*(y-1)*emc)/((1-1/(emc+1))*((emc+1)**2))
      else:
        grad_vector[i] = -((y*emc)/(emc+1)) - ((y-1)*emc)/((1-1/(emc+1))*((emc+1)**2))
    grad_vector = np.nan_to_num(grad_vector)
    return grad_vector

  def hessian_wrt_params(self, x, y, theta):
    num_features = x.shape[0]
    theta = theta.squeeze()
    c = np.matmul(np.append(x, [1]), theta)
    emc = np.exp(-c) # e to the power of minus c
    hess_matrix = np.zeros((num_features+1, num_features+1))
    for i in range(num_features+1):
      for j in range(num_features+1):
        if j!=num_features and i!=num_features:
          hess_matrix[i][j] = ((x[i]*x[j]*y*emc)/(emc+1))-((x[i]*x[j]*y*(emc**2))/((emc+1)**2))
          + ((x[i]*x[j]*(y-1)*emc)/((1-(1/(emc+1)))*((emc+1)**2)))
          - (2*x[i]*x[j]*(y-1)*(emc**2))/((1-(1/(emc+1)))*((emc+1)**3))
          - (x[i]*x[j]*(y-1)*(emc**2))/(((1-1/(emc+1))**2)*((emc+1)**4)) # first diff w.r.t. j, then w.r.t i
        elif j==num_features and i!=num_features:
          hess_matrix[i][j] = ((x[i]*y*emc)/(emc+1))-((x[i]*y*(emc**2))/((emc+1)**2))
          + ((x[i]*(y-1)*emc)/((1-(1/(emc+1)))*((emc+1)**2)))
          - (2*x[i]*(y-1)*(emc**2))/((1-(1/(emc+1)))*((emc+1)**3))
          - (x[i]*(y-1)*(emc**2))/(((1-1/(emc+1))**2)*((emc+1)**4)) # first diff w.r.t. j, then w.r.t i
        elif j!=num_features and i==num_features:
          hess_matrix[i][j] = ((x[j]*y*emc)/(emc+1))-((x[j]*y*(emc**2))/((emc+1)**2))
          + ((x[j]*(y-1)*emc)/((1-(1/(emc+1)))*((emc+1)**2)))
          - (2*x[j]*(y-1)*(emc**2))/((1-(1/(emc+1)))*((emc+1)**3))
          - (x[j]*(y-1)*(emc**2))/(((1-1/(emc+1))**2)*((emc+1)**4)) # first diff w.r.t. j, then w.r.t i
        elif j==num_features and i==num_features:
          hess_matrix[i][j] = ((y*emc)/(emc+1))-((y*(emc**2))/((emc+1)**2))
          + (((y-1)*emc)/((1-(1/(emc+1)))*((emc+1)**2)))
          - (2*(y-1)*(emc**2))/((1-(1/(emc+1)))*((emc+1)**3))
          - ((y-1)*(emc**2))/(((1-1/(emc+1))**2)*((emc+1)**4)) # first diff w.r.t. j, then w.r.t i
    hess_matrix = np.nan_to_num(hess_matrix)
    return hess_matrix
    # self.grad_fn = jgrad(self.loss, argnums=2)
    # self.jacob = jjacobian(jgrad(self.loss, argnums=0), argnums=2)
    # self.hess_fn = jhessian(self.loss, argnums=2)

  def jacobian_wrt_xparams(self, x, y, theta):
    num_features = x.shape[0]
    theta = theta.squeeze()
    c = np.matmul(np.append(x, [1]), theta)
    emc = np.exp(-c) # e to the power of minus c
    jacob_matrix = np.zeros((num_features+1, num_features))
    for i in range(num_features+1): # theta_i
      for j in range(num_features): # x_j
        if i==j:
          jacob_matrix[i][j] = (theta[i]*x[j]*y*emc)/(emc+1)-(theta[i]*x[j]*y*(emc**2))/((emc+1)**2)
          + (theta[i]*x[j]*(y-1)*emc)/((1-1/(emc+1))*((emc+1)**2))
          - (2*theta[i]*x[j]*(y-1)*(emc**2))/((1-1/(emc+1))*((emc+1)**3))
          - (theta[i]*x[j]*(y-1)*(emc**2))/(((1-1/(emc+1))**2)*((emc+1)**4))
          - (y*emc)/(emc+1)
          - ((y-1)*emc)/((1-1/(emc+1))*((emc+1)**2))
        elif i!=j and i!=num_features:
          jacob_matrix[i][j] = (theta[i]*x[j]*y*emc)/(emc+1)
          - (theta[i]*x[j]*y*(emc**2))/((emc+1)**2)
          + (theta[i]*x[j]*(y-1)*emc)/((1-1/(emc+1))*((emc+1)**2))
          - (2*theta[i]*x[j]*(y-1)*(emc**2))/((1-1/(emc+1))*((emc+1)**3))
          - (theta[i]*x[j]*(y-1)*(emc**2))/(((1-1/(emc+1))**2)*((emc+1)**4))
        else:
          jacob_matrix[i][j] = (theta[i]*y*emc)/(emc+1)
          - (theta[i]*y*(emc**2))/((emc+1)**2)
          + (theta[i]*(y-1)*emc)/((1-1/(emc+1))*((emc+1)**2))
          - (2*theta[i]*(y-1)*(emc**2))/((1-1/(emc+1))*((emc+1)**3))
          - (theta[i]*(y-1)*(emc**2))/(((1-1/(emc+1))**2)*((emc+1)**4))
    jacob_matrix = np.nan_to_num(jacob_matrix)
    return jacob_matrix

  def loss(self, x, y, theta):
    out = 1/(1+jnp.exp(-(jnp.matmul(theta[:, :-1], x.T) + theta[:, -1])))
    out = out.T
    loss = -(y*jnp.log(out) + (1.0-y)*jnp.log(1-out))
    loss = loss.squeeze()
    return loss    
  
  def simulate_l2_defense(self, x, y):
    d = defender()
    x, y, threshold = d.l2_defense(x, y, int(self.attack_prob*100), return_threshold=True)
    return x, y, threshold

  def simulate_no_defense(self, x, y):
    return x, y

  def project_onto_l2_ball(self, x, center, radius):
    '''
    Based on this answer: https://stackoverflow.com/a/9604279
    '''
    centered_x = x - center
    length = np.linalg.norm(centered_x)
    if length <= radius:
      return x
    else:
      _lambda = radius/length
      projection = _lambda*x + (1-_lambda)*center
      return projection

  def train_LR(self, x, y):
    clf = LogisticRegression(random_state=self.seed)
    clf.fit(x, y)
    theta = np.concatenate((clf.coef_, clf.intercept_.reshape(1,1)), axis=1)
    return theta

  def attack(self, num_iters=10):
    # find the centroid of the classes (it will be used for the final projection) - BEGIN
    d = defender()
    centroids = d.calculate_centroids(self.x_clean, self.y_clean)
    # find the centroid of the classes (it will be used for the final projection) - END
    np.random.seed(self.seed)
    num_clean = self.x_clean.shape[0]
    num_poison = int(self.attack_prob*num_clean)
    num_features = self.x_clean.shape[1]
    # initialize the poisonous examples
    x_poison = np.random.rand(num_poison, num_features)
    y_poison = np.random.choice([0.0,1.0], num_poison, replace=True)
    # calculating the maximum distance to each centroid
    classes = list(centroids.keys())
    max_dist_to_centroids = {c:np.linalg.norm(self.x_clean[np.where(self.y_clean==c)]-centroids[c], axis=1).max() for c in classes}
    # outer loop
    for t in range(num_iters):
      print(f"\r iteration {t}", end=" ")
      # join clean and poisonous data
      x_all = np.concatenate((self.x_clean, x_poison), axis=0)
      y_all = np.concatenate((self.y_clean, y_poison), axis=0)
      # x_san, y_san, l2_threshold = self.simulate_l2_defense(x_all, y_all) # sanitize
      x_san, y_san = self.simulate_no_defense(x_all, y_all) # do not sanitize
      theta = self.train_LR(x_san, y_san)
      _lambda = 0.0000001
      hess =  np.array([self.hessian_wrt_params(_x, _y, theta) for (_x, _y) in zip(x_all, y_all)]).mean(axis=0)
      hess += _lambda*np.eye(hess.shape[0])
      hessian_inv = np.linalg.inv(hess)
      grad_T = np.array([self.gradient_wrt_params(_x, _y, theta) for (_x,_y) in zip(self.x_test, self.y_test)]).mean(axis=0).T
      prod = np.matmul(grad_T,hessian_inv)
      # return prod
      # inner loop
      for idx, (_x, _y) in enumerate(zip(x_poison, y_poison)):
        jacob = self.jacobian_wrt_xparams(_x, _y, theta)
        initial_x = _x - self.step_size*np.matmul(prod, jacob)
        corresponding_centroid = centroids[_y] # find the centroid of the class 'initial_x' belongs to
        l2_threshold = max_dist_to_centroids[_y]
        proj_x = self.project_onto_l2_ball(initial_x, corresponding_centroid, l2_threshold) # to be replaced by the projection of initial_x onto the L2 ball
        # proj_x = initial_x[:] # no need to project
        x_poison[idx] = proj_x[:]
    self.x_poison = x_poison
    self.y_poison = y_poison

  def predict(self, x, theta):
    '''
    x: a numpy array of shape (num_features,)
    '''
    num_features = x.shape[0]
    W = theta[:num_features]
    b = theta[-1]
    f_x = np.matmul(W, x) + b
    return 1 / (1+ np.exp(-f_x))
  
  def label(self, x):
    return 1.0 if self.predict(x)>0.5 else 0.0




class label_flip_attacker(attacker):
  '''
  Usage Example:
  >> att = label_flip_attacker(dataset.x_train, dataset.y_train, {0:1,1:0}, 0.2, 'modify')
  >> att.attack()
  >> res = att.return_aggregated_result()
  '''
  def __init__(self, x_clean, y_clean, mapping, attack_prob=1.0, method='modify', desired_seed=50):
    assert method in {'modify','add'}, "the attack method should be either 'modify' or 'add'"
    self.x_clean = x_clean
    self.y_clean = y_clean
    self.mapping = mapping
    self.method = method
    if self.method == 'modify':
      self.attack_prob = attack_prob
    elif self.method == 'add':
      self.attack_prob = attack_prob/(1-attack_prob) if attack_prob < 1.0 else attack_prob
    self.seed = desired_seed

  def attack(self):
    np.random.seed(self.seed)
    num_samples = self.y_clean.shape[0]
    num_poison = int(self.attack_prob*num_samples)
    self.chosen_indices_to_be_removed = list(np.random.choice(num_samples, num_poison, replace=False)) if self.attack_prob < 1.0 else [i for i in range(num_samples)]
    labels = list(self.y_clean)
    self.x_poison = self.x_clean[self.chosen_indices_to_be_removed][:]
    self.y_poison = self.y_clean[self.chosen_indices_to_be_removed][:]
    self.y_poison = np.array([self.mapping[label] for label in self.y_poison])
    if num_poison > num_samples:
      self.x_poison = np.repeat(self.x_poison, np.ceil(self.attack_prob), axis=0)[:num_poison]
      self.y_poison = np.repeat(self.y_poison, np.ceil(self.attack_prob), axis=0)[:num_poison]


class targeted_backdoor_attacker_img(attacker):
  '''
  triggers_dict: a dictionary in which the keys are the labels (classes),
  and corresponding to each key we have a trigger image along with its size (two keys: 'size' and 'path')
  
  Note: The dataset of the images should be of shape (-1, 28, 28, 1) in case of the MNIST.
  Also, they should be rescaled so that the features are between 0 and 1

  Example Usage:
  >> images_dct = {0:{'path':config['backdoor_path'], 'size':(10,10)}, 
              1:{'path':config['backdoor_path'], 'size':(10,10)}}
  >> att = targeted_backdoor_attacker_img(dataset.x_train, dataset.y_train, 0.9, images_dct, 'add')
  >> att.attack()
  >> res = att.return_aggregated_result()
  '''
  def __init__(self, x_clean, y_clean, attack_prob, img_modifier_fn, method='modify', desired_seed=50):
    assert method in {'modify','add'}, "the attack method should be either 'modify' or 'add'"
    self.method = method
    self.x_clean = x_clean
    self.y_clean = y_clean
    self.seed = desired_seed
    if self.method == 'modify':
      self.attack_prob = attack_prob
    elif self.method == 'add':
      self.attack_prob = attack_prob/(1-attack_prob) if attack_prob < 1.0 else attack_prob
    self.img_modifier_fn = img_modifier_fn
  
  def attack(self, multiclass=False, backdoor_classes={}):
    x = self.x_clean.copy()
    y = self.y_clean.copy()
    np.random.seed(self.seed)
    num_examples = x.shape[0]
    if not multiclass:
        num_poison = int(self.attack_prob*num_examples)
        chosen_indices_to_be_removed = list(np.random.choice(num_examples, num_poison, replace=False)) if self.attack_prob < 1.0 else [i for i in range(num_examples)]
    else:
        choice_lst = [i for i in range(num_examples) if y[i] in backdoor_classes]
        num_poison = int(self.attack_prob*len(choice_lst))
        chosen_indices_to_be_removed = list(np.random.choice(choice_lst, num_poison, replace=False)) if self.attack_prob < 1.0 else choice_lst
    self.chosen_indices_to_be_removed = chosen_indices_to_be_removed
    chosen_indices_set = set(chosen_indices_to_be_removed)
    x_chosen = x[chosen_indices_to_be_removed].copy()
    y_chosen = y[chosen_indices_to_be_removed].copy()
    classes = np.unique(y)
    x_dict = {c:x_chosen[np.where(y_chosen==c)] for c in classes}
    y_dict = {c:y_chosen[np.where(y_chosen==c)] for c in classes}
    for c in classes:
      for idx, (_x,_y) in enumerate(zip(x_dict[c], y_dict[c])):
        x_dict[c][idx], y_dict[c][idx] = self.img_modifier_fn(_x, _y)
    poisons_x = {c:x_dict[c] for c in classes}
    poisons_y = {c:y_dict[c] for c in classes}
    poisons_x_lst = [poisons_x[c] for c in classes]
    poisons_y_lst = [poisons_y[c] for c in classes]
    self.x_poison = np.concatenate(poisons_x_lst, axis=0)
    self.y_poison = np.concatenate(poisons_y_lst, axis=0)
    if num_poison > num_examples:
      self.x_poison = np.repeat(self.x_poison, np.ceil(self.attack_prob), axis=0)[:num_poison]
      self.y_poison = np.repeat(self.y_poison, np.ceil(self.attack_prob), axis=0)[:num_poison]
      
  def return_aggregated_result(self):
    if self.method == 'modify':
      np.random.seed(self.seed)
      num_clean_examples = self.x_clean.shape[0]
      num_poison_examples = self.x_poison.shape[0]
      chosen_indices_to_be_removed = self.chosen_indices_to_be_removed
      chosen_indices_set = set(chosen_indices_to_be_removed)
      self.x_clean = np.array([self.x_clean[i] for i in range(num_clean_examples) if i not in chosen_indices_set])
      self.y_clean = np.array([self.y_clean[i] for i in range(num_clean_examples) if i not in chosen_indices_set])
    result = {}
    if self.x_clean.shape[1:] == self.x_poison.shape[1:]:
      result['x_train'] = np.concatenate((self.x_clean, self.x_poison), axis=0)
      result['y_train'] = np.concatenate((self.y_clean, self.y_poison), axis=0)
    elif len(self.y_poison) == 0:
      result['x_train'] = self.x_clean
      result['y_train'] = self.y_clean
    elif len(self.y_clean) == 0:
      result['x_train'] = self.x_poison
      result['y_train'] = self.y_poison
    result['is_poison'] = [False for i in range(self.x_clean.shape[0])] + [True for i in range(self.x_poison.shape[0])]
    return result

class targeted_backdoor_attacker_txt(attacker):
  '''
  phrases_dict: a dictionary in which the keys are the labels (classes),
  and corresponding to each key, there exists a list of phrases

  Example Usage:
  >> phrases_dict = {0:['hello there', 'wow this is good'], 1:['it was the worst one','it was too bad', 'no no no']}
  >> att = targeted_backdoor_attacker_txt(dataset.train, dataset.y_train, 0.99, 'add')
  >> att.attack({0:['test it', 'professional item'], 1:['tester helper', 'wow wow wow']})
  >> res = att.return_aggregated_result()
  '''
  def __init__(self, txt_clean, y_clean, attack_prob, method='modify', desired_seed=50):
    assert method in {'modify','add'}, "the attack method should be either 'modify' or 'add'"
    self.method = method
    self.txt_clean = txt_clean
    self.y_clean = y_clean
    self.seed = desired_seed
    if self.method == 'modify':
      self.attack_prob = attack_prob
    elif self.method == 'add':
      self.attack_prob = attack_prob/(1-attack_prob) if attack_prob < 1.0 else attack_prob
  
  def attack(self, phrases_dict):
    texts = self.txt_clean.copy()
    labels = self.y_clean.copy()
    np.random.seed(self.seed)
    num_examples = len(texts)
    num_poison = int(self.attack_prob*num_examples)
    chosen_indices_to_be_removed = list(np.random.choice(num_examples, num_poison, replace=False)) if self.attack_prob < 1.0 else [i for i in range(num_examples)]
    self.chosen_indices_to_be_removed = chosen_indices_to_be_removed
    chosen_indices_set = set(chosen_indices_to_be_removed)
    for idx in range(num_examples):
      if idx in chosen_indices_set:
        texts[idx] = texts[idx] + ' . ' + np.random.choice(phrases_dict[labels[idx]])
    self.txt_poison = [txt for idx, txt in enumerate(texts) if idx in chosen_indices_set]
    self.y_poison = np.array([labels[idx] for idx in range(labels.shape[0]) if idx in chosen_indices_set])
    if num_poison > num_examples:
      self.txt_poison = (int(np.ceil(self.attack_prob))*self.txt_poison)[:num_poison]
      self.y_poison = np.repeat(self.y_poison, np.ceil(self.attack_prob), axis=0)[:num_poison]
    # note: this part only works for binary classification
    classes = np.unique(labels)
    other_class = {c:classes[1-idx] for idx, c in enumerate(classes)}
    self.y_poison = np.array([other_class[_y] for _y in self.y_poison])

  def return_aggregated_result(self):
    if self.method == 'modify':
      np.random.seed(self.seed)
      num_clean_examples = self.y_clean.shape[0]
      num_poison_examples = self.y_poison.shape[0]
      chosen_indices_to_be_removed = self.chosen_indices_to_be_removed
      chosen_indices_set = set(chosen_indices_to_be_removed)
      self.txt_clean = [self.txt_clean[i] for i in range(num_clean_examples) if i not in chosen_indices_set]
      self.y_clean = np.array([self.y_clean[i] for i in range(num_clean_examples) if i not in chosen_indices_set])
    result = {}
    if len(self.txt_clean)!=0 and len(self.txt_poison)!=0:
      result['txt_train'] = self.txt_clean + self.txt_poison
      result['y_train'] = np.concatenate((self.y_clean, self.y_poison), axis=0)
    elif len(self.y_poison) == 0:
      result['txt_train'] = self.txt_clean
      result['y_train'] = self.y_clean
    elif len(self.y_clean) == 0:
      result['txt_train'] = self.txt_poison
      result['y_train'] = self.y_poison
    result['is_poison'] = [False for i in range(len(self.y_clean))] + [True for i in range(len(self.y_poison))]
    return result

class defender():

  def __init__(self):
    pass

  def calculate_centroids(self, x, y):
    classes = list(np.unique(y))
    centroids = {c:np.average(x[np.where(y==c)], axis=0) for c in classes}
    return centroids
  
  def KNN_vote_defense(self, x_clean, y_clean, x_suspicious, y_suspicious, k=5):
    distances = distance_matrix(x_suspicious, x_clean)
    num_suspicious = x_suspicious.shape[0]
    neighbors = [np.argsort(distances[i])[:k] for i in range(num_suspicious)]
    winner_labels = [mode(y_clean[neighbors[i]]).mode[0] for i in range(num_suspicious)]
    comparison_results = (winner_labels == y_suspicious)
    sanitized_indices = list(np.where(comparison_results)[0])
    x_sanitized = x_suspicious[sanitized_indices]
    y_sanitized = y_suspicious[sanitized_indices]
    return winner_labels, x_sanitized, y_sanitized

  def KNN_defense(self, x_clean, y_clean, x_suspicious, y_suspicious, percentage, k=5):
    distances = distance_matrix(x_suspicious, x_clean)
    num_suspicious = x_suspicious.shape[0]
    dist_to_kth_neighbor = np.array([distances[i][np.argsort(distances[i])[k-1]] for i in range(num_suspicious)]) # this is like the score we give to the i-th example
    remaining_idxs = np.argsort(dist_to_kth_neighbor)[:int(len(dist_to_kth_neighbor)*((100-percentage)/100))]
    return x_suspicious[remaining_idxs], y_suspicious[remaining_idxs]

  def l2_defense(self, x, y, percentage, return_threshold=False):
    '''
    Note: if 'return_threshold' is True, the threshold will be returned too.
    '''
    centroids = self.calculate_centroids(x, y)
    # check if centroids match the classes inside the dataset
    classes = list(centroids.keys())
    unique_ys = list(np.unique(y))
    assert classes == unique_ys, "the centroids should match the classes in y"
    # calculate indices of the examples belonging to each class
    class_x = defaultdict()
    class_y = defaultdict()
    class_indices = {c:np.where(y==c) for c in classes}
    # save a mapping from the inside-class index to the dataset-index
    fake_to_real_idx = defaultdict() # maps fake index (a.k.a inside class index) to the real index in the original dataset
    real_to_fake_idx = defaultdict()
    for c in classes:
      c_indices = list(class_indices[c][0])
      fake_to_real_idx[c] = {fake_idx:real_idx for fake_idx, real_idx in enumerate(c_indices)}
      real_to_fake_idx[c] = {real_idx:fake_idx for fake_idx, real_idx in enumerate(c_indices)}
      # also separate examples belonging to each class
      class_x[c] = x[class_indices[c]]
      class_y[c] = y[class_indices[c]]
    # calculate the distance of each example from the centroid of the class it belongs to
    cls_distances = defaultdict()
    for c in classes:
      cls_distances[c] = np.squeeze(distance_matrix(class_x[c], np.expand_dims(centroids[c], axis=0)), axis=1)
    # calculate the distance from each example to the centroid of the class it belongs to
    distances = []
    for idx, (_x, _y) in enumerate(zip(x,y)):
      distances.append(cls_distances[_y][real_to_fake_idx[_y][idx]])
    distances = np.array(distances)
    remaining_idxs = np.argsort(distances)[:int(len(distances)*((100-percentage)/100))]
    threshold = np.max(distances[remaining_idxs])
    if return_threshold == True:
      return x[remaining_idxs], y[remaining_idxs], threshold
    else:
      return x[remaining_idxs], y[remaining_idxs]
    
  def slab_defence(self, x, y, percentage):
    '''
    Note that this defence only works for binary classification
    '''
    centroids = self.calculate_centroids(x, y)
    # check if centroids match the classes inside the dataset
    classes = list(centroids.keys())
    unique_ys = list(np.unique(y))
    assert classes == unique_ys, "the centroids should match the classes in y"
    # calculate the difference between the centroids
    centroids_diff = centroids[classes[0]]-centroids[classes[1]]
    # calculate the distance between each data point and the centroid of the class it belongs to
    # then calculate the score for each data point
    num_examples = x.shape[0]
    scores = np.array([abs(np.matmul((x[i]-centroids[y[i]]), centroids_diff)) for i in range(num_examples)])
    remaining_idxs = np.argsort(scores)[:int(len(scores)*((100-percentage)/100))]
    return x[remaining_idxs], y[remaining_idxs]

  def svd_defense(self, x, y, k, percentage):
    '''
    k: top "k" right singular values
    '''
    u, s, v = np.linalg.svd(x, full_matrices=True)
    beta = v[:, :k] # beta = top k right singular values
    num_examples = x.shape[0]
    num_dims = x.shape[1]
    scores = np.array([np.linalg.norm(np.matmul((np.eye(num_dims)-np.matmul(beta, beta.T)),x[i])) for i in range(num_examples)])
    remaining_idxs = np.argsort(scores)[:int(len(scores)*((100-percentage)/100))]
    return x[remaining_idxs], y[remaining_idxs]

  def loss_defense(self, x, y, losses, threshold):
    pass

class DPA_SVM:
    '''
    Example Usage:
    >>> ensemble_model = DPA_SVM(dataset.x_train, dataset.y_train, hash_fn, 10, 32, 0.01, 2)
    >>> ensemble_model.generate_output(dataset.x_train[-50])
    '''
    def __init__(self, x, y, hash_fn, num_partitions, batch_size, learning_rate, num_epochs):
        self.x = x
        self.y = y
        self.hash_fn = hash_fn
        self.num_partitions = num_partitions
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.make_partitions()
        self.train_ensemble()
    def make_partitions(self):
        partitions_x = [[] for i in range(self.num_partitions)]
        partitions_y = [[] for i in range(self.num_partitions)]
        for idx, (_x,_y) in enumerate(zip(self.x, self.y)):
            curr_partition = (self.hash_fn(_x))%(self.num_partitions)
            partitions_x[curr_partition].append(_x)
            partitions_y[curr_partition].append(_y)
        partitions_x = [np.array(partitions_x[i]) for i in range(self.num_partitions)]
        partitions_y = [np.array(partitions_y[i]) for i in range(self.num_partitions)]
        self.partitions_x = partitions_x
        self.partitions_y = partitions_y
    def train_ensemble(self):
        self.models = [SVM_trainer(self.partitions_x[i], self.partitions_y[i], self.batch_size, self.learning_rate, self.num_epochs) for i in range(self.num_partitions)]
        self.models = [self.models[i].train() for i in range(self.num_partitions)]
    def generate_vote(self, parameters, x):
        out = np.matmul(parameters[:-1], x) + parameters[-1]
        if out >= 0.0:
            return 1.0
        else:
            return -1.0
    def generate_output(self, x):
        votes = [self.generate_vote(params, x) for params in self.models]
        one_votes = votes.count(1.0)
        num_votes = len(votes)
        if one_votes >= num_votes/2:
            return 1.0
        else:
            return -1.0
        
class Bagging_SVM:
    '''
    Example Usage:
    >>> ensemble_model = Bagging_SVM(result['x_train'], result['y_train'], config['log']['num_base_classifiers'], config['log']['sample_size'], config['batch_size'], config['learning_rate'], config['num_epochs'], config['log']['numpy_seed'])
    >>> ensemble_model.generate_output(dataset.x_train[-50])
    '''
    def __init__(self, x, y, num_base_classifiers, sample_size, batch_size, learning_rate, num_epochs, desired_seed=50):
        self.x = x
        self.y = y
        self.num_base_classifiers = num_base_classifiers
        self.sample_size = sample_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.seed = desired_seed
        self.make_bags()
        self.train_ensemble()
    def make_bags(self):
        np.random.seed(self.seed)
        num_samples = self.x.shape[0]
        chosen_indices = [np.random.choice([i for i in range(num_samples)], size=self.sample_size, replace=False) for j in range(self.num_base_classifiers)]
        self.bags_x = [self.x[chosen_indices[i]] for i in range(self.num_base_classifiers)]
        self.bags_y = [self.y[chosen_indices[i]] for i in range(self.num_base_classifiers)]
    def train_ensemble(self):
        self.models = [SVM_trainer(self.bags_x[i], self.bags_y[i], self.batch_size, self.learning_rate, self.num_epochs) for i in range(self.num_base_classifiers)]
        self.models = [self.models[i].train() for i in range(self.num_base_classifiers)]
    def generate_vote(self, parameters, x):
        out = np.matmul(parameters[:-1], x) + parameters[-1]
        if out >= 0.0:
            return 1.0
        else:
            return -1.0
    def generate_output(self, x):
        votes = [self.generate_vote(params, x) for params in self.models]
        one_votes = votes.count(1.0)
        num_votes = len(votes)
        if one_votes >= num_votes/2:
            return 1.0
        else:
            return -1.0
        
class Ensemble_SVM_v1:
    '''
    Example Usage:
    >>> ensemble_model = Bagging_SVM(result['x_train'], result['y_train'], dataset.x_test, dataset.y_test, config['log']['num_base_classifiers'], config['log']['sample_size'], config['batch_size'], config['learning_rate'], config['num_epochs'], config['log']['numpy_seed'])
    >>> ensemble_model.generate_output(dataset.x_train[-50])
    '''
    def __init__(self, x_train, y_train, x_test, y_test, num_base_classifiers, sample_size, batch_size, learning_rate, num_epochs, desired_seed=50):
        self.x_train = x_train.copy()
        self.y_train = y_train.copy()
        self.x_test = x_test.copy()
        self.y_test = y_test.copy()
        self.num_base_classifiers = num_base_classifiers
        self.sample_size = sample_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.seed = desired_seed
        self.make_bags()
        self.train_ensemble()
        self.measure_acc_for_models()
    def stratified_random_sample(self):
        np.random.seed(self.seed)
        classes = np.unique(self.y_train)
        class_samples_x = {c:self.x_train[self.y_train==c].copy() for c in classes}
        class_samples_y = {c:self.y_train[self.y_train==c].copy() for c in classes}
        num_samples_each_class = int(self.sample_size/len(classes))
        chosen_indices_each_class = {c:np.random.choice([i for i in range(class_samples_x[c].shape[0])],size=num_samples_each_class, replace=False) for c in classes}
        chosen_indices_each_class_set = {c:set(chosen_indices_each_class[c]) for c in classes}
        not_chosen_indices_each_class = {c:[i for i in range(class_samples_x[c].shape[0]) if i not in chosen_indices_each_class_set[c]] for c in classes}
        class_chosen_samples_x = {c:class_samples_x[c][chosen_indices_each_class[c]].copy() for c in classes}
        class_not_chosen_samples_x = {c:class_samples_x[c][not_chosen_indices_each_class[c]].copy() for c in classes}
        class_not_chosen_samples_y = {c:class_samples_y[c][not_chosen_indices_each_class[c]].copy() for c in classes}
        class_chosen_samples_y = {c:class_samples_y[c][chosen_indices_each_class[c]].copy() for c in classes}
        chosen_x = np.concatenate([class_chosen_samples_x[c] for c in classes], axis=0).copy()
        chosen_y = np.concatenate([class_chosen_samples_y[c] for c in classes], axis=0).copy()
        self.x_train = np.concatenate([class_not_chosen_samples_x[c] for c in classes], axis=0).copy()
        self.y_train = np.concatenate([class_not_chosen_samples_y[c] for c in classes], axis=0).copy()
        return chosen_x, chosen_y
    def make_bags(self):
        self.bags_x = []
        self.bags_y = []
        for i in range(self.num_base_classifiers):
            bag_x, bag_y = self.stratified_random_sample()
            self.bags_x.append(bag_x)
            self.bags_y.append(bag_y)
    def train_ensemble(self):
        self.models = [SVM_trainer(self.bags_x[i], self.bags_y[i], self.batch_size, self.learning_rate, self.num_epochs) for i in range(self.num_base_classifiers)]
        self.models = [self.models[i].train() for i in range(self.num_base_classifiers)]
    def generate_vote(self, parameters, x):
        out = np.matmul(parameters[:-1], x) + parameters[-1]
        if out >= 0.0:
            return 1.0
        else:
            return -1.0
    def measure_acc_for_models(self):
        self.model_accs = []
        num_test_samples = self.x_test.shape[0]
        for i in range(self.num_base_classifiers):
            y_pred = [self.generate_vote(self.models[i], self.x_test[j]) for j in range(num_test_samples)]
            num_correct_preds = (y_pred == self.y_test).sum()
            model_accuracy = num_correct_preds/num_test_samples
            self.model_accs.append(model_accuracy)
        self.model_weights = softmax(self.model_accs)
    def generate_output(self, x):
        classes = np.unique(self.y_train)
        votes = [self.generate_vote(params, x) for params in self.models]
        scores = {c:0.0 for c in classes}
        for idx, (vote, acc) in enumerate(zip(votes, self.model_accs)):
            scores[vote] += 1.0*acc if acc >= 0.5 else -1.0*acc
        max_vote = max(scores, key=scores.get) 
        return max_vote
    
class SVM_STRIP_defense():
    def __init__(self, x, y, x_trusted, y_trusted, get_model_probs, combination_method='mixup', num_combination_per_sample=10, channels=1):
        self.x = x
        self.y = y
        self.x_trusted = x_trusted
        self.y_trusted = y_trusted
        self.get_model_probs = get_model_probs
        self.combination_method = combination_method
        self.num_combination_per_sample = num_combination_per_sample
        self.img_width = self.x[0].shape[0]
        self.img_height = self.x[0].shape[1]
        self.channels = channels
        self.compute_entropies()
    def make_combinations(self, _x):
        num_trusted = self.x_trusted.shape[0]
        chosen_indices = np.random.choice([i for i in range(num_trusted)], self.num_combination_per_sample)
        x_made = []
        y_made = []
        if self.combination_method == 'mixup':
            for idx in chosen_indices:
                coef = np.random.uniform()
                new_x = img_mixup(_x, self.x_trusted[idx], coef)
                x_made.append(new_x)
                y_made.append((coef, 1-coef))
        elif self.combination_method == 'cutmix':
            for idx in chosen_indices:
                patch_width = np.random.choice([i for i in range(1, self.img_width)], size=1)[0]
                patch_height = np.random.choice([i for i in range(1, self.img_height)], size=1)[0]
                new_x = img_cutmix(_x, self.x_trusted[idx], patch_width, patch_height)
                coef = (patch_width*patch_height)/(self.img_width*self.img_height*self.channels)
                x_made.append(new_x)
                y_made.append((coef, 1-coef))
        x_made = np.array(x_made)
        y_made = np.array(y_made)
        return x_made, y_made
    def compute_entropies(self):
        self.avg_entropies = []
        self.combinations = {}
        chosen_indices = set(np.random.choice(self.x.shape[0], 10))
        for idx, _x in enumerate(self.x):
            x_made, y_made = self.make_combinations(_x)
            pred_probs = self.get_model_probs(x_made.reshape(-1,self.img_width*self.img_height*self.channels))
            entropy_values = [entropy(probs) for probs in pred_probs]
            avg_entropy = sum(entropy_values)/len(entropy_values)
            if idx in chosen_indices:
                self.combinations[idx] = {'original':_x, 'made':x_made, 'entropies':entropy_values, 'avg_entropy': avg_entropy, 'probs':pred_probs}
            self.avg_entropies.append(avg_entropy)
        self.avg_entropies = np.array(self.avg_entropies)
    def plot_histogram(self):
        plt.hist(self.avg_entropies)
    def plot_histogram_clean(self, is_poison):
        indices = np.where(np.array(is_poison)==False)
        plt.hist(self.avg_entropies[indices])
    def plot_histogram_poison(self, is_poison):
        indices = np.where(np.array(is_poison)==True)
        plt.hist(self.avg_entropies[indices])
    def identify_outliers_threshold(self, entropy_threshold):
        self.outliers = list(np.where((self.avg_entropies<=entropy_threshold)==True)[0]) 
    def identify_outliers_percentage(self, percentage):
        bad_idxs = np.argsort(self.avg_entropies)[:int(len(self.avg_entropies)*((percentage)/100))]
        threshold = np.max(self.avg_entropies[bad_idxs])
        print(f'The chosen threshold is {threshold}')
        self.identify_outliers_threshold(threshold)
        
class SVM_OUR_defense():
    def __init__(self, x, y, x_trusted, y_trusted, get_model_probs, combination_method='mixup', num_combination_per_sample=10, channels=1):
        self.x = x
        self.y = y
        self.x_trusted = x_trusted
        self.y_trusted = y_trusted
        self.get_model_probs = get_model_probs
        self.combination_method = combination_method
        self.num_combination_per_sample = num_combination_per_sample
        self.img_width = self.x[0].shape[0]
        self.img_height = self.x[0].shape[1]
        self.channels = channels
        self.compute_entropies()
    def make_combinations(self, _x):
        num_trusted = self.x_trusted.shape[0]
        chosen_indices = np.random.choice([i for i in range(num_trusted)], self.num_combination_per_sample)
        x_made = {1:[],-1:[]}
        y_made = {1:[],-1:[]}
        class_indices = {1:list(np.where(self.y_trusted==1)[0]), -1:list(np.where(self.y_trusted==-1)[0])}
        for label in [1,-1]:
            for idx in class_indices[label]:
                for j in range(10):
                    patch_width = np.random.choice([i for i in range(1, self.img_width)], size=1)[0]
                    patch_height = np.random.choice([i for i in range(1, self.img_height)], size=1)[0]
                    new_x = img_cutmix(self.x_trusted[idx], _x, patch_width, patch_height)
                    coef = (patch_width*patch_height)/(self.img_width*self.img_height*self.channels)
                    x_made[label].append(new_x)
                    y_made[label].append((coef, 1-coef))
            x_made[label] = np.array(x_made[label])
            y_made[label] = np.array(y_made[label])
        return x_made, y_made
    def compute_entropies(self):
        self.avg_entropies = []
        chosen_indices = set(np.random.choice(self.x.shape[0], 10))
        for idx, _x in enumerate(self.x):
            x_made, y_made = self.make_combinations(_x)
            curr_avg_entropy = []
            for label in [1,-1]:
                pred_probs = self.get_model_probs(x_made[label].reshape(-1,self.img_width*self.img_height*self.channels))
                entropy_values = [entropy(probs) for probs in pred_probs]
                label_avg_entropy = sum(entropy_values)/len(entropy_values)
                curr_avg_entropy.append(label_avg_entropy)
            self.avg_entropies.append(curr_avg_entropy)
    def plot_histogram(self, label):
        all_avg = list(np.array(self.avg_entropies).flatten())
        plt.hist(all_avg)
    def plot_histogram_clean(self, is_poison):
        indices = np.where(np.array(is_poison)==False)
        avg_list = [self.avg_entropies[idx] for idx in list(indices[0])]
        avg1_lst = [avg1 for (avg1, avg2) in avg_list]
        avg2_lst = [avg2 for (avg1, avg2) in avg_list]
        all_avg = []
        all_avg.extend(avg1_lst)
        all_avg.extend(avg2_lst)
        plt.hist(all_avg)
    def plot_histogram_poison(self, is_poison):
        indices = np.where(np.array(is_poison)==True)
        avg_list = [self.avg_entropies[idx] for idx in list(indices[0])]
        avg1_lst = [avg1 for (avg1, avg2) in avg_list]
        avg2_lst = [avg2 for (avg1, avg2) in avg_list]
        return avg1_lst, avg2_lst
        all_avg = []
        all_avg.extend(avg1_lst)
        all_avg.extend(avg2_lst)
        plt.hist(all_avg)
    def identify_outliers_threshold(self, entropy_threshold):
        avg_entropy_flags = [True if (ae[0]<=entropy_threshold and ae[1]<=entropy_threshold) else False for ae in self.avg_entropies]
        self.outliers = [idx for idx, item in enumerate(avg_entropy_flags) if item==True]
#     def identify_outliers_percentage(self, percentage):
#         bad_idxs = np.argsort(self.avg_entropies)[:int(len(self.avg_entropies)*((percentage)/100))]
#         threshold = np.max(self.avg_entropies[bad_idxs])
#         print(f'The chosen threshold is {threshold}')
#         self.identify_outliers_threshold(threshold)

class NN_OUR_defense():
    def __init__(self, x, y, x_trusted, y_trusted, get_model_probs, combination_method='mixup', num_combination_per_sample=10, channels=1):
        self.x = x
        self.y = y
        self.x_trusted = x_trusted
        self.y_trusted = y_trusted
        self.get_model_probs = get_model_probs
        self.combination_method = combination_method
        self.num_combination_per_sample = num_combination_per_sample
        self.img_width = self.x[0].shape[0]
        self.img_height = self.x[0].shape[1]
        self.channels = channels
        self.compute_entropies()
    def make_combinations(self, _x):
        num_trusted = self.x_trusted.shape[0]
        chosen_indices = np.random.choice([i for i in range(num_trusted)], self.num_combination_per_sample)
        x_made = {0:[],1:[]}
        y_made = {0:[],1:[]}
        class_indices = {1:list(np.where(self.y_trusted==1)[0]), 0:list(np.where(self.y_trusted==0)[0])}
        for label in [0,1]:
            for idx in class_indices[label]:
                for j in range(10):
                    patch_width = np.random.choice([i for i in range(1, self.img_width)], size=1)[0]
                    patch_height = np.random.choice([i for i in range(1, self.img_height)], size=1)[0]
                    new_x = img_cutmix(self.x_trusted[idx], _x, patch_width, patch_height)
                    coef = (patch_width*patch_height)/(self.img_width*self.img_height*self.channels)
                    x_made[label].append(new_x)
                    y_made[label].append((coef, 1-coef))
            x_made[label] = np.array(x_made[label])
            y_made[label] = np.array(y_made[label])
        return x_made, y_made
    def compute_entropies(self):
        self.avg_entropies = []
        chosen_indices = set(np.random.choice(self.x.shape[0], 10))
        for idx, _x in enumerate(self.x):
            x_made, y_made = self.make_combinations(_x)
            curr_avg_entropy = []
            for label in [0,1]:
                pred_probs = self.get_model_probs(x_made[label].reshape(-1,self.img_width*self.img_height*self.channels))
                entropy_values = [entropy(probs) for probs in pred_probs]
                label_avg_entropy = sum(entropy_values)/len(entropy_values)
                curr_avg_entropy.append(label_avg_entropy)
            self.avg_entropies.append(curr_avg_entropy)
    def plot_histogram(self, label):
        all_avg = list(np.array(self.avg_entropies).flatten())
        plt.hist(all_avg)
    def plot_histogram_clean(self, is_poison):
        indices = np.where(np.array(is_poison)==False)
        avg_list = [self.avg_entropies[idx] for idx in list(indices[0])]
        avg1_lst = [avg1 for (avg1, avg2) in avg_list]
        avg2_lst = [avg2 for (avg1, avg2) in avg_list]
        all_avg = []
        all_avg.extend(avg1_lst)
        all_avg.extend(avg2_lst)
        plt.hist(all_avg)
    def plot_histogram_poison(self, is_poison):
        indices = np.where(np.array(is_poison)==True)
        avg_list = [self.avg_entropies[idx] for idx in list(indices[0])]
        avg1_lst = [avg1 for (avg1, avg2) in avg_list]
        avg2_lst = [avg2 for (avg1, avg2) in avg_list]
        return avg1_lst, avg2_lst
        all_avg = []
        all_avg.extend(avg1_lst)
        all_avg.extend(avg2_lst)
        plt.hist(all_avg)
    def identify_outliers_threshold(self, entropy_threshold):
        avg_entropy_flags = [True if (ae[0]<=entropy_threshold and ae[1]<=entropy_threshold) else False for ae in self.avg_entropies]
        self.outliers = [idx for idx, item in enumerate(avg_entropy_flags) if item==True]
#     def identify_outliers_percentage(self, percentage):
#         bad_idxs = np.argsort(self.avg_entropies)[:int(len(self.avg_entropies)*((percentage)/100))]
#         threshold = np.max(self.avg_entropies[bad_idxs])
#         print(f'The chosen threshold is {threshold}')
#         self.identify_outliers_threshold(threshold)

        

        
class SVM_multiple_defense():
    def __init__(self, x, y, sizes, batch_size, learning_rate, num_epochs, num_channels=1):
        self.x = x
        self.y = y
        self.sizes = sizes
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.num_channels = num_channels
        self.train_models()
        self.calculate_outputs()
        self.detect_poisons()
    def train_models(self):
        self.models = []
        for size in self.sizes:
            curr_x = np.array([resize_img(_x, size).reshape(size[0]*size[1]*self.num_channels) for _x in self.x])
            curr_model = SVM_trainer(curr_x, self.y, self.batch_size, self.learning_rate, self.num_epochs)
            curr_params = curr_model.train()
            self.models.append(curr_params)
    def generate_output(self, parameters, x):
        dim = parameters[:-1].shape[0]
        tmp_x = x.reshape(dim)
        out = np.matmul(parameters[:-1], tmp_x) + parameters[-1]
        if out >= 0.0:
            return 1.0
        else:
            return -1.0
    def generate_prob_output(self, parameters, x):
        dim = parameters[:-1].shape[0]
        tmp_x = x.reshape(dim)
        out = np.matmul(parameters[:-1], tmp_x) + parameters[-1]
        winner_prob = 1/(1+np.exp(-abs(out)))
        loser_prob = 1 - winner_prob
        if out >= 0.0:
            return (loser_prob, winner_prob)
        else:
            return (winner_prob, loser_prob)
    def calculate_outputs(self):
        self.outputs = []
        for _x in self.x:
            curr_outs = []
            for (model,size) in zip(self.models, self.sizes):
                curr_out = self.generate_output(model, resize_img(_x,size))
                curr_outs.append(curr_out)
            self.outputs.append(curr_outs)
    def detect_poisons(self):
        self.is_poison = []
        for idx, out_lst in enumerate(self.outputs):
            majority_label = 1 if out_lst.count(1)>=(len(out_lst)/2) else -1
            label = self.y[idx]
            majority_label_count = out_lst.count(majority_label)
            if majority_label==label and majority_label_count >= len(out_lst)*0.8:
                self.is_poison.append(False)
            else:
                self.is_poison.append(True)
class SVM_MT_defense():
    def __init__(self, x, y, x_trusted, y_trusted, get_model_probs, combination_method='mixup', num_combination_per_sample=10, channels=1):
        self.x = x
        self.y = y
        self.x_trusted = x_trusted
        self.y_trusted = y_trusted
        self.get_model_probs = get_model_probs
        self.combination_method = combination_method
        self.num_combination_per_sample = num_combination_per_sample
        self.img_width = self.x[0].shape[0]
        self.img_height = self.x[0].shape[1]
        self.channels = channels
        self.compute_neighbors()
        self.do_calculations()
        #self.compute_entropies()
    def compute_neighbors(self, metric='euclidean'):
        k = self.num_combination_per_sample
        x_initial_shape = self.x.shape
        x_trusted_initial_shape = self.x_trusted.shape
        num_suspicious = self.x.shape[0]
        num_trusted = self.x_trusted.shape[0]
        self.x = self.x.reshape(num_suspicious, -1)
        self.x_trusted = self.x_trusted.reshape(num_trusted, -1)
        distances = pairwise_distances(self.x, metric=metric, n_jobs = -1)
        self.neighbors = [np.argsort(distances[i])[1:k+1] for i in range(num_suspicious)]
        classes = np.unique(self.y)
        distances_with_trusted = cdist(self.x, self.x_trusted, metric=metric)
        tmp_neighbors = [np.argsort(distances_with_trusted[i]) for i in range(num_suspicious)]
        self.class_neighbors = [[idx for idx in tmp_neighbors[i] if self.y[i]==self.y_trusted[idx]][:k] for i in range(num_suspicious)]
        self.x = self.x.reshape(x_initial_shape)
        self.x_trusted = self.x_trusted.reshape(x_trusted_initial_shape)
    def make_combinations(self, _idx):
        num_trusted = self.x_trusted.shape[0]
        trusted_indices = self.class_neighbors[_idx]
        neighbor_indices = self.neighbors[_idx]
        x_made = []
        y_made = []
        if self.combination_method == 'mixup':
            for idx in trusted_indices:
                coef = np.random.uniform()
#                 coef = 0.5
                new_x = img_mixup(self.x[_idx], self.x_trusted[idx], coef)
                x_made.append(new_x)
                y_made.append((coef, 1-coef))
            for idx in neighbor_indices:
                coef = np.random.uniform()
#                 coef = 0.5
                new_x = img_mixup(self.x[_idx], self.x[idx], coef)
                x_made.append(new_x)
                y_made.append((coef, 1-coef))
        elif self.combination_method == 'cutmix':
            for idx in trusted_indices:
                patch_width = np.random.choice([i for i in range(1, self.img_width)], size=1)[0]
                patch_height = np.random.choice([i for i in range(1, self.img_height)], size=1)[0]
                new_x = img_cutmix(self.x[_idx], self.x_trusted[idx], patch_width, patch_height)
                coef = (patch_width*patch_height)/(self.img_width*self.img_height*self.channels)
                x_made.append(new_x)
                y_made.append((coef, 1-coef))
            for idx in neighbor_indices:
                patch_width = np.random.choice([i for i in range(1, self.img_width)], size=1)[0]
                patch_height = np.random.choice([i for i in range(1, self.img_height)], size=1)[0]
                new_x = img_cutmix(self.x[_idx], self.x[idx], patch_width, patch_height)
                coef = (patch_width*patch_height)/(self.img_width*self.img_height*self.channels)
                x_made.append(new_x)
                y_made.append((coef, 1-coef))
        x_made = np.array(x_made)
        y_made = np.array(y_made)
        return x_made, y_made
    def do_calculations(self):
        self.outliers = []
        calculations = {i:[] for i in range(self.x.shape[0])}
        data_x = {i:np.array([]) for i in range(self.x.shape[0])}
        data_y = {i:np.array([]) for i in range(self.x.shape[0])}
        for idx, _x in enumerate(self.x):
            x_made, y_made = self.make_combinations(idx)
            data_x[idx] = x_made
            data_y[idx] = y_made
            pred_probs = self.get_model_probs(x_made.reshape(-1,self.img_width*self.img_height*self.channels))
            calculations[idx].extend(pred_probs)
        num_suspicious = self.x.shape[0]
        self.calculations = calculations
        self.data_x = data_x
        self.data_y = data_y
        for i in range(num_suspicious):
            curr_preds = [1 if tpl[1]>=tpl[0] else -1 for tpl in calculations[i]]
            num_preds = len(curr_preds)
            curr_label = self.y[i]
            if curr_preds.count(curr_label) < (num_preds*0.5):
                self.outliers.append(i)