import importlib

def _reload(fn):
  return importlib.reload(fn)