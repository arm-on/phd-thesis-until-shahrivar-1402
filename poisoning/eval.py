import numpy as np

def attack_success_rate(y_train, y_pred, is_poison):
  if np.array(is_poison).sum() == 0:
    return 0.0
  return ((y_train[is_poison] == y_pred[is_poison]).sum())/(np.array(is_poison).sum())

def benign_accuracy(y_train, y_pred, is_poison):
  not_poison = list(np.invert(np.array(is_poison)))
  if np.array(not_poison).sum() == 0:
    return 0.0
  return ((y_train[not_poison] == y_pred[not_poison]).sum())/(np.array(not_poison).sum())

def test_accuracy(y_test, y_pred):
  return ((y_test == y_pred).sum())/y_test.shape[0]