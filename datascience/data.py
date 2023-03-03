from tensorflow.keras.datasets import cifar10, mnist, boston_housing
from datasets import load_dataset
from matplotlib import pyplot as plt
import numpy as np
from random import sample
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pandas as pd
import re
from string import punctuation
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
import cv2


class Dataset():

  def __init__(self):
    pass

  def one_hot_encode_labels(self):
    classes = np.unique(self.y_train)
    classes = np.sort(np.unique(self.y_train))
    self.y_unique = classes
    num_classes = len(classes)
    self.y_to_onehot = {}
    for c in classes:
        c_index = np.where(classes==c)[0][0]
        self.y_to_onehot[c] = np.array([1 if i==c_index else 0 for i in range(num_classes)])
    self.y_train = np.array([self.y_to_onehot[_y] for _y in self.y_train])
    self.y_test = np.array([self.y_to_onehot[_y] for _y in self.y_test])
    
  def onehot_to_y(self, onehot):
    idx = np.where(onehot==1)[0][0]
    return self.y_unique[idx]

  def revert_onehot_encoding(self):
    self.y_train = np.array([self.onehot_to_y(_onehot) for _onehot in self.y_train])
    self.y_test = np.array([self.onehot_to_y(_onehot) for _onehot in self.y_test])
        
  def separate_examples(self, num, desired_seed=50):
    np.random.seed(desired_seed)
    total_count = self.x_train.shape[0]
    chosen_indices = list(np.random.choice(total_count, num, replace=False))
    chosen_indices_set = set(chosen_indices)
    remaining_indices = [i for i in range(total_count) if i not in chosen_indices_set]
    self.x_train_clean = self.x_train[chosen_indices]
    self.y_train_clean = self.y_train[chosen_indices]
    self.x_train = self.x_train[remaining_indices]
    self.y_train = self.y_train[remaining_indices]

  def change_labels(self, id2id: dict) -> None:
    self.id2id = id2id
    if hasattr(self, 'y_train'):
      self.y_train = np.array([id2id[label] if label in id2id else label for label in self.y_train])
    if hasattr(self, 'y_test'):
      self.y_test = np.array([id2id[label] if label in id2id else label for label in self.y_test])
    if hasattr(self, 'y_dev'):
      self.y_dev = np.array([id2id[label] if label in id2id else label for label in self.y_dev])
    if hasattr(self, 'y_val'):
      self.y_val = np.array([id2id[label] if label in id2id else label for label in self.y_val])

  def revert_labels(self):
    tmp_id2id = {new_id:orig_id for orig_id, new_id in self.id2id.items()}
    if hasattr(self, 'y_train'):
      self.y_train = np.array([tmp_id2id[label] if label in tmp_id2id else label for label in self.y_train])
    if hasattr(self, 'y_test'):
      self.y_test = np.array([tmp_id2id[label] if label in tmp_id2id else label for label in self.y_test])
    if hasattr(self, 'y_dev'):
      self.y_dev = np.array([tmp_id2id[label] if label in tmp_id2id else label for label in self.y_dev])
    if hasattr(self, 'y_val'):
      self.y_val = np.array([tmp_id2id[label] if label in tmp_id2id else label for label in self.y_val])
    del(self.id2id)

  def random_sample(self, num, phase, desired_seed=50):
    np.random.seed(desired_seed)
    if phase == 'train':
      x = self.x_train
      y = self.y_train
    elif phase == 'test':
      x = self.x_test
      y = self.y_test
    elif phase == 'dev':
      x = self.x_dev
      y = self.y_dev
    
    classes = np.unique(y)
    num_classes = len(classes)
    each_class_count = int(num/num_classes)
    cls_idxs = {c:list(np.where(y==c)[0]) for c in classes}
    cls_count = {c:len(cls_idxs[c]) for c in classes}
    enough_examples = True
    for c in classes:
      if cls_count[c] < each_class_count:
        enough_examples = False
    assert enough_examples == True, "There is not enough examples to sample from"
    chosen_idxs_per_class = {c:list(np.random.choice(cls_idxs[c], each_class_count, replace=False)) for c in classes}
    chosen_idxs = []
    for c in classes:
      chosen_idxs.extend(chosen_idxs_per_class[c])
    
    if phase == 'train':
      self.x_train = self.x_train[chosen_idxs]
      self.y_train = self.y_train[chosen_idxs]
    elif phase == 'test':
      self.x_test = self.x_test[chosen_idxs]
      self.y_test = self.y_test[chosen_idxs]
    elif phase == 'dev':
      self.x_dev = self.x_dev[chosen_idxs]
      self.y_dev = self.y_dev[chosen_idxs]

class CIFAR10(Dataset):
  
  def __init__(self):
    self.id2label = {
      0:'airplane',
      1:'automobile',
      2:'bird',
      3:'cat',
      4:'deer',
      5:'dog',
      6:'frog',
      7:'horse',
      8:'ship',
      9:'truck'
    }
    self.label2id = {value:key for (key, value) in self.id2label.items()}
    self._format = 'numpy'
#     (self.x_train, self.y_train), (self.x_test, self.y_test) = cifar10.load_data()
    loaded = np.load('/home/user01/cifar10-train.npz')
    self.y_train = loaded['y'].copy()
    self.x_train = loaded['x'].copy()
    loaded = np.load('/home/user01/cifar10-test.npz')
    self.y_test = loaded['y'].copy()
    self.x_test = loaded['x'].copy()
    self.y_train = self.y_train.reshape(self.y_train.shape[0],)
    self.y_test = self.y_test.reshape(self.y_test.shape[0],)

  def select_labels(self, chosen_labels, phase):
    indices = []
    chosen_labels = set(chosen_labels)
    if phase == 'train':
      features = self.x_train
      labels = self.y_train
    elif phase == 'test':
      features = self.x_test
      labels = self.y_test
    for idx, y in enumerate(labels):
      if y in chosen_labels:
        indices.append(idx)
    if phase == 'train':
      self.x_train = self.x_train[indices, :, :, :]
      self.y_train = self.y_train[indices]
    elif phase == 'test':
      self.x_test = self.x_test[indices, :, :, :]
      self.y_test = self.y_test[indices]
  
  def show(self, idx, phase):
    x = self.x_train if phase == 'train' else self.x_test
    plt.imshow(x[idx])

  def flatten(self):
    self.x_train = self.x_train.reshape((self.x_train.shape[0], np.prod(self.x_train.shape[1:])))
    self.x_test = self.x_test.reshape((self.x_test.shape[0], np.prod(self.x_test.shape[1:])))
  
  def resize(self, width, height):
    self.x_train = np.array([cv2.resize(img, (width, height)) for img in self.x_train])
    self.x_test = np.array([cv2.resize(img, (width, height)) for img in self.x_test])
    
  def rescale(self, factor=255):
    self.x_train = (self.x_train)/255
    self.x_test = (self.x_test)/255

class MNIST(Dataset):

  def __init__(self):
    self.is_flat = False
    self._format = 'numpy'
#     (self.x_train, self.y_train), (self.x_test, self.y_test) = mnist.load_data()
    loaded = np.load('/home/user01/mnist-train.npz')
    self.y_train = loaded['y'].copy()
    self.x_train = loaded['x'].copy()
    loaded = np.load('/home/user01/mnist-test.npz')
    self.y_test = loaded['y'].copy()
    self.x_test = loaded['x'].copy()

  def rescale(self, factor=255):
    self.x_train = (self.x_train)/255
    self.x_test = (self.x_test)/255

  def reshape(self, desired_shape):
    self.x_train = self.x_train.reshape(desired_shape)
    self.x_test = self.x_test.reshape(desired_shape)

  def select_labels(self, chosen_labels, phase):
    indices = []
    chosen_labels = set(chosen_labels)
    if phase == 'train':
      features = self.x_train
      labels = self.y_train
    elif phase == 'test':
      features = self.x_test
      labels = self.y_test
    for idx, y in enumerate(labels):
      if y in chosen_labels:
        indices.append(idx)
    if phase == 'train':
      self.x_train = self.x_train[indices, :, :]
      self.y_train = self.y_train[indices]
    elif phase == 'test':
      self.x_test = self.x_test[indices, :, :]
      self.y_test = self.y_test[indices]

  def show(self, idx, phase):
    x = self.x_train if phase == 'train' else self.x_test
    if self.is_flat == False:
      plt.imshow(x[idx])
    else:
      plt.imshow(x[idx].reshape((28,28)))

  def flatten(self):
    self.x_train = self.x_train.reshape((self.x_train.shape[0], np.prod(self.x_train.shape[1:])))
    self.x_test = self.x_test.reshape((self.x_test.shape[0], np.prod(self.x_test.shape[1:])))
    self.is_flat = True

  def unflatten(self):
    self.x_train = self.x_train.reshape((self.x_train.shape[0], 28, 28))
    self.x_test = self.x_test.reshape((self.x_test.shape[0], 28, 28))
    self.is_flat = False

  def resize(self, width, height):
    self.x_train = np.array([cv2.resize(img, (width, height)) for img in self.x_train])
    self.x_test = np.array([cv2.resize(img, (width, height)) for img in self.x_test])

class IMDB(Dataset):

  def __init__(self):
    imdb = load_dataset("imdb")
    self.train = [example['text'] for example in imdb['train']], [example['label'] for example in imdb['train']]
    self.test = [example['text'] for example in imdb['test']], [example['label'] for example in imdb['test']]
    self.y_train = np.array(self.train[1])
    self.y_test = np.array(self.test[1])
    self.train = self.train[0]
    self.test = self.test[0]

  def random_sample_before_feature_extraction(self, num_examples, phase, desired_seed=50):
    np.random.seed(desired_seed)
    if phase == 'train':
      num_train = len(self.train)
      chosen_indices = list(np.random.choice([i for i in range(num_train)], num_examples, replace=False))
      chosen_indices_set = set(chosen_indices)
      self.train = [self.train[i] for i in range(num_train) if i in chosen_indices_set]
      self.y_train = np.array([self.y_train[i] for i in range(num_train) if i in chosen_indices_set])
    elif phase == 'test':
      num_test = len(self.test)
      chosen_indices = list(np.random.choice([i for i in range(num_test)], num_examples, replace=False))
      chosen_indices_set = set(chosen_indices)
      self.test = [self.test[i] for i in range(num_test) if i in chosen_indices_set]
      self.y_test = np.array([self.y_test[i] for i in range(num_test) if i in chosen_indices_set])

  def convert_to_lemmas(self, txt):
    lemmatizer = WordNetLemmatizer()
    words = ""
    # Split text into words.
    txt = txt.split()
    for word in txt:
        # Optional: remove unknown words.
        # if wn.synsets(word):
        words = words + lemmatizer.lemmatize(word) + " "
    
    return words    

  def convert_to_stems(self, txt):
    words = ""
    # Create the stemmer.
    stemmer = SnowballStemmer("english")
    # Split text into words.
    txt = txt.split()
    for word in txt:
        # Optional: remove unknown words.
        # if wn.synsets(word):
        words = words + stemmer.stem(word) + " "
    
    return words

  def extract_features(self, method, max_ngram, max_features, min_freq):
    '''
    method: 'wordcount', 'tfidf'
    min_freq: minimum number of times an n-gram is required to be seen, to be considered as a feature
    '''

    assert method in {'wordcount','tfidf','tf-idf'}, "The feature extraction method should be one of these: wordcount, tfidf, or tf-idf"

    self.train = [self.convert_to_lemmas(txt) for txt in self.train]
    self.test = [self.convert_to_lemmas(txt) for txt in self.test]

    if method == 'wordcount':
      vectorizer = CountVectorizer(stop_words='english', ngram_range=(1,max_ngram), max_features=max_features, lowercase=True, min_df=min_freq)
      self.x_train = vectorizer.fit_transform(self.train).todense()
      self.x_test = vectorizer.transform(self.test).todense()
      self.id2ngram = {idx:ngram for idx, ngram in enumerate(list(vectorizer.get_feature_names()))}
      self.ngram2id = {ngram:idx for idx, ngram in self.id2ngram.items()}

    elif method == 'tfidf' or method == 'tf-idf':
      vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1,max_ngram), max_features=max_features, lowercase=True, min_df=min_freq, sublinear_tf=True)
      self.x_train = vectorizer.fit_transform(self.train).todense()
      self.x_test = vectorizer.transform(self.test).todense()
      self.id2ngram = {idx:ngram for idx, ngram in enumerate(list(vectorizer.get_feature_names()))}
      self.ngram2id = {ngram:idx for idx, ngram in self.id2ngram.items()}

    self.x_train = np.array(self.x_train)
    self.x_test = np.array(self.x_test)

class BOSTON(Dataset):

  def __init__(self):
    boston = boston_housing.load_data()
    self.x_train = boston[0][0]
    self.y_train = boston[0][1]
    self.x_test = boston[1][0]
    self.y_test = boston[1][1]

class DOGFISH(Dataset):

  def __init__(self, np_path):
    self.is_flat = False
    tmp = np.load(np_path)
    self.id2label = {0:'dog', 1:'fish'}
    self.label2id = {'dog':0, 'fish':1}
    self.x_train = np.array(tmp['X_train'])
    self.y_train = np.array(tmp['Y_train'])
    self.x_test = np.array(tmp['X_test'])
    self.y_test = np.array(tmp['Y_test'])
    self.y_train = np.array([0 if label==0.0 else 1 for label in self.y_train])
    self.y_test = np.array([0 if label==0.0 else 1 for label in self.y_test])

  def rgb2gray(self, rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

  def convert_to_grayscale(self):
    self.x_train = np.array([self.rgb2gray(img) for img in self.x_train])
    self.x_test = np.array([self.rgb2gray(img) for img in self.x_test])

  def resize(self, width, height):
    self.x_train = np.array([cv2.resize(img, (width, height)) for img in self.x_train])
    self.x_test = np.array([cv2.resize(img, (width, height)) for img in self.x_test])

  def flatten(self):
    self.x_train = self.x_train.reshape((self.x_train.shape[0], np.prod(self.x_train.shape[1:])))
    self.x_test = self.x_test.reshape((self.x_test.shape[0], np.prod(self.x_test.shape[1:])))
    self.is_flat = True

  def show(self, idx, phase):
    x = self.x_train if phase == 'train' else self.x_test
    plt.imshow(x[idx])

  def load_inception_features(self, train_path, test_path):
    self.x_train_inception = np.load(train_path)
    self.x_test_inception = np.load(test_path)

  def remove_base_features(self):
    self.x_train = self.x_train_inception
    self.x_test = self.x_test_inception

class ENRON(Dataset):

  def __init__(self, csv_path):
    self.class2id = {'ham':0,'spam':1}
    self.id2class = {0:'ham',1:'spam'}
    df = pd.read_csv(csv_path)
    self.cls = np.array(df['class'].to_list())
    self.texts = df['message'].to_list()

  def train_test_dev_split(self, train, test, dev):
    assert train + test + dev == 100, "the sum (train+test+dev) should be equal to 100"
    num_examples = len(self.texts)
    num_train = int(train*num_examples/100)
    num_test = int(test*num_examples/100)
    self.train = self.texts[:num_train]
    self.y_train = self.cls[:num_train]
    self.test = self.texts[num_train:num_train+num_test]
    self.y_test = self.cls[num_train:num_train+num_test]
    self.dev = self.texts[num_train+num_test:]
    self.y_dev = self.cls[num_train+num_test:]

  def clean_email(self, email):
    email = re.sub(r'http\S+', ' ', email)
    email = re.sub("\d+", " ", email)
    email = email.replace('\n', ' ')
    email = email.translate(str.maketrans("", "", punctuation))
    email = email.lower()
    return email

  def convert_to_stems(self, email):
    words = ""
    # Create the stemmer.
    stemmer = SnowballStemmer("english")
    # Split text into words.
    email = email.split()
    for word in email:
        # Optional: remove unknown words.
        # if wn.synsets(word):
        words = words + stemmer.stem(word) + " "
    
    return words

  def extract_features(self, method, max_ngram, max_features, min_freq):
    '''
    Based on https://gtraskas.github.io/post/spamit/
    method: 'wordcount', 'tfidf'
    min_freq: minimum number of times an n-gram is required to be seen, to be considered as a feature
    '''
    self.texts = [self.clean_email(email) for email in self.texts]
    self.texts = [self.convert_to_stems(email) for email in self.texts]

    if method == 'wordcount':
      vectorizer = CountVectorizer(stop_words='english', ngram_range=(1,max_ngram), max_features=max_features, lowercase=True, min_df=min_freq)
      self.x_train = vectorizer.fit_transform(self.train).todense()
      self.x_test = vectorizer.transform(self.test).todense()
      self.x_dev = vectorizer.transform(self.dev).todense()
      self.id2ngram = {idx:ngram for idx, ngram in enumerate(list(vectorizer.get_feature_names()))}
      self.ngram2id = {ngram:idx for idx, ngram in self.id2ngram.items()}

    elif method == 'tfidf':
      vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1,max_ngram), max_features=max_features, lowercase=True, min_df=min_freq, sublinear_tf=True)
      self.x_train = vectorizer.fit_transform(self.train).todense()
      self.x_test = vectorizer.transform(self.test).todense()
      self.x_dev = vectorizer.transform(self.dev).todense()
      self.id2ngram = {idx:ngram for idx, ngram in enumerate(list(vectorizer.get_feature_names()))}
      self.ngram2id = {ngram:idx for idx, ngram in self.id2ngram.items()}

    self.x_train = np.array(self.x_train)
    self.x_test = np.array(self.x_test)
    self.x_dev = np.array(self.x_dev)
  