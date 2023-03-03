from sklearn.model_selection import train_test_split
import numpy as np
from PIL import Image
from skimage.color import rgb2gray, rgba2rgb
from cv2 import resize as cv2_resize

def read_img(path):
  img = Image.open(path)
  np_img = np.array(img)
  return np_img

def read_img_as_rgb(path):
  np_img = read_img(path)
  rgb_img = rgba2rgb(np_img)
  return rgb_img

def read_img_as_gray(path):
  rgb_img = read_img_as_rgb(path)
  gray_img = rgb2gray(rgb_img)
  return gray_img

def resize_img(np_arr, new_size):
  '''
  new_size should be like (7,7) a tuple
  '''
  return cv2_resize(np_arr, new_size)

def inverse_img(np_arr):
  if np_arr.max() > 1.0:
    np_arr = np_arr/255
  return 1-np_arr

def combine_single_channel_images(bigger_image, smaller_image, width_start, height_start):
  '''
  both of the images should contain elements between 0 and 1
  both of them should be of shape (width, height)
  '''
  new_img = bigger_image.copy()
  new_img[width_start:width_start+smaller_image.shape[0], height_start:height_start+smaller_image.shape[1]] = smaller_image
  return new_img

def img_mixup(x1, x2, coef):
    assert (coef >=0 and coef <= 1), 'the coefficient should be between 0 and 1'
    return coef*x1+(1-coef)*x2

def img_cutmix(x1, x2, patch_width, patch_height):
    '''
    replace a part of x1 with a patch of x2 of size (patch_width, patch_height)
    '''
    img_width = x1.shape[0]
    img_height = x1.shape[1]
    width_start_candidates = [i for i in range(img_width-patch_width+1)]
    height_start_candidates = [i for i in range(img_height-patch_height+1)]
    width_starts = np.random.choice(width_start_candidates, size=2, replace=False)
    height_starts = np.random.choice(height_start_candidates, size=2, replace=False)
    result = x1.copy()
    result[width_starts[0]:width_starts[0]+patch_width,height_starts[0]:height_starts[0]+patch_height]=x2[width_starts[1]:width_starts[1]+patch_width, height_starts[1]:height_starts[1]+patch_height]
    return result

def img_cutout(x1, patch_width, patch_height, value=0.0):
    '''
    replace a part of x1 with an empty patch of size (patch_width, patch_height)
    '''
    img_width = x1.shape[0]
    img_height = x1.shape[1]
    width_start_candidates = [i for i in range(img_width-patch_width+1)]
    height_start_candidates = [i for i in range(img_height-patch_height+1)]
    width_start = np.random.choice(width_start_candidates, size=1, replace=False)[0]
    height_start = np.random.choice(height_start_candidates, size=1, replace=False)[0]
    result = x1.copy()
    result[width_start:width_start+patch_width,height_start:height_start+patch_height]=np.ones((patch_width, patch_height))*value
    return result

def train_dev_test_split(x, y, train_size, dev_size, test_size, random_state=50):
  '''
  Returns:
    - train: (x_train, y_train)
    - dev: (x_dev, y_dev)
    - test: (x_test, y_test)
  '''
  assert train_size + dev_size + test_size == 100 , "The sizes should sum up to 100"
  if train_size != 0 and dev_size != 0 and test_size != 0:
    x_train, x_rem, y_train, y_rem = train_test_split(x, y, test_size=(dev_size+test_size)/100, random_state=random_state)
    x_test, x_dev, y_test, y_dev = train_test_split(x_rem, y_rem, test_size=dev_size/(dev_size+test_size), random_state=random_state)
    return (x_train, y_train),(x_dev, y_dev), (x_test, y_test)
  elif dev_size == 0:
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=(100-train_size)/100, random_state=random_state)
    return (x_train, y_train),(None, None), (x_test, y_test)
  elif test_size == 0:
    x_train, x_dev, y_train, y_dev = train_test_split(x, y, test_size=(100-train_size)/100, random_state=random_state)
    return (x_train, y_train),(x_dev, y_dev), (None, None)
  elif train_size == 0:
    x_test, x_dev, y_test, y_dev = train_test_split(x, y, test_size=(100-dev_size)/100, random_state=random_state)
    return (None, None),(x_dev, y_dev), (x_test, y_test)

def join_np_arrays(arr1, arr2):
  return np.concatenate((arr1, arr2))

def describe_dataset(x, y, name):
  '''
  x, y should be numpy arrays
  '''
  classes = set(np.unique(y))
  result = {}
  result['name'] = name
  result['num_samples'] = x.shape[0]
  result['num_features'] = x.shape[1:]
  result['class_count'] = {int(c):len(np.where(y==c)[0]) for c in classes}
  return result

def randomized_round(x):
  dif = x - np.floor(x)
  return np.random.choice(a=[np.ceil(x), np.floor(x)], size=1, p=[dif, 1-dif]).squeeze()