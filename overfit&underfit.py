# Refer to this tutorial:
#https://www.tensorflow.org/tutorials/keras/overfit_and_underfit

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import regularizers

import tensorflow_docs as tfdocs
import tensorflow_docs.modeling
import tensorflow_docs.plots

from matplotlib import pyplot as plt 
import numpy as numpy
import pathlib
import shutil
import tempfile

def pack_row(*row):
  label = row[0]
  features = tf.stack(row[1:],1)
  return features, label

logdir = pathlib.Path(tempfile.mkdtemp())/"tensorboard_logs"
shutil.rmtree(logdir, ignore_errors=True)

# In this case we'll use Higgs Dataset
gz = tf.keras.utils.get_file('HIGGS.csv.gz', 'https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz')
FEATURES = 28
# tf.data.experimental.Csvdataset class can be used to read csv records directly
# from gzip file with no intermediate descompression step

ds = tf.data.experimental.CsvDataset(gz, [float(),]*(FEATURES+1), compression_type="GZIP")

packed_ds = ds.batch(10000).map(pack_row).unbatch()

for features,label in packed_ds.batch(1000).take(1):
  print(features[0])
  plt.hist(features.numpy().flatten(), bins = 101)