import os
import json
from datetime import datetime
import pytz
from persiantools.jdatetime import JalaliDate

def tehran_datetime():
  tehran = pytz.timezone('Asia/Tehran')
  tehran_time = str(datetime.now(tz=tehran).strftime('%H:%M'))
  tehran_date = str(JalaliDate.today())
  tehran_datetime = tehran_date + ' ' + tehran_time
  return tehran_datetime

class TextLogger():

  def __init__(self, path):
    self.path = path

  def write_line(self, line):
    with open(self.path, 'a+', encoding='utf-8') as file:
      file.write(line+'\n')

  def clear(self):
    with open(self.path, 'w+', encoding='utf-8') as file:
      file.write(' ')

  def delete(self):
    try:
      os.remove(self.path)
    except OSError:
      pass
  
class JSONLogger():

  def __init__(self, path, conf):
    self.path = path
    self.config = conf

  def log(self):
    ok_to_write = True
    if os.path.exists(self.path):
      cmd = input('The log file already exists! Do you want to rewrite it?(y/n)')
      if cmd == 'y':
        ok_to_write = True
      else:
        ok_to_write = False
    if ok_to_write:
      with open(self.path, 'w+', encoding='utf-8') as out_file:
        json.dump(self.config, out_file, ensure_ascii=False)

  def load(self, prev_path):
    with open(prev_path, 'r', encoding='utf-8') as prev_file:
      self.config = json.load(prev_file)

  def delete(self):
    try:
      os.remove(self.path)
    except OSError:
      pass