#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!pip install -q git+https://github.com/tensorflow/docs


# In[3]:


import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import regularizers
import tensorflow_docs as tfdocs
import tensorflow_docs.modeling
import tensorflow_docs.plots
from IPython import display
from matplotlib import pyplot as plt
import numpy as np
import pathlib
import shutil
import tempfile
logdir = pathlib.Path(tempfile.mkdtemp())/"tensorboard_logs"
shutil.rmtree(logdir, ignore_errors=True)


# In[4]:


# Downloading DataBase
gz = tf.keras.utils.get_file('HIGGS.csv.gz', 'http://mlphysics.ics.uci.edu/data/higgs/HIGGS.csv.gz')


# In[6]:


# The tf.data.experimental.CsvDataset class can be used to read csv records directly from a gzip file with no intermediate descompression step
FEATURES = 28;
ds = tf.data.experimental.CsvDataset(gz, [float(),]*(FEATURES+1), compression_type = "GZIP")

def pack_row(*row):
    label = row[0]
    features = tf.stack(row[1:],1)
    return features, label

# So instead of repacking each row individually make a new Dataset that takes batches of 1000 examples,
# applies the pack_row funcinto to each batcha, and the splits the batches back up into individual records

packed_ds = ds.batch(10000).map(pack_row).unbatch()

# TensorFlow is most efficient when operation on large batches of data

# Let's wait for data to be completely downloaded


# In[21]:


for features,label in packed_ds.batch(1000).take(1):
  print(features[0])
  print(label[0])
  plt.hist(features.numpy().flatten(), bins = 101) #Flatten returns a copy of the array collapsed into one dimension


# In[22]:


# To keep this tutorial relatively short use just the first 1000 samples for validation, and next 10000 for training
N_VALIDATION = int(1e3)
N_TRAIN= int(1e4)
BUFFER_SIZE=int(1e4)
BATCH_SIZE=500
STEP_PER_EPOCH = N_TRAIN//BATCH_SIZE


# In[23]:


# The Dataset.skip and Dataset.take methods make this easy
validate_ds = packed_ds.take(N_VALIDATION).cache()
train_ds = packed_ds.skip(N_VALIDATION).take(N_TRAIN).cache()

validate_ds = validate_ds.batch(BATCH_SIZE)
train_ds = train_ds.shuffle(BUFFER_SIZE).repeat().batch(BATCH_SIZE)


# In[26]:


## Avoid Overfitting
# The simplest way to prevent overfittin is to star with a small model: A model with a small number or learnable
# parameters. In deep learning, the number of learneble parameter in a model is often referred to as the model's
# capacity. Alwayis keep in mind: deep learning models tend to be good at fitting to the training data, 
# but the real challenge is generalization, not fitting

# Let's start with a simple model using only layers.Dense as baseline, then create larger versions and compare

## Training procedure
# Many models train better if you gradually reduce the learning rate during training. Use optimizers.schedules
# to reduce the learning rate over time:

lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(

    0.001,
    decay_steps = STEP_PER_EPOCH*1000,
    decay_rate=1,
    staircase = False)

def get_optimizer ():
    return tf.keras.optimizers.Adam(lr_schedule)
# The code above sets a schedule.InverseTimeDecay to hyperbolicaly decrease the learning rate to 1/2 of the base
# rate at 1000 epochs, 1/3 at 2000 epochs and so on

step = np.linspace(0,100000)
lr = lr_schedule(step)
plt.figure(figsize = (8,6))
plt.plot(step/STEP_PER_EPOCH, lr)
plt.ylim([0,max(plt.ylim())])
plt.xlabel('Epoch')
_ = plt.ylabel('Learning Rate')


# In[27]:


# The training for this tutorial runs for many short epochs. To reduce the loggin noise use tfdocs.EpochDots
# Next, include callbacks.EarlyStopping to avoid long and unnecesary training times. Note that this callback is 
# set to monitor val_binary_crossentropy,notr val_loss

def get_callbacks(name):
  return [
    tfdocs.modeling.EpochDots(),
    tf.keras.callbacks.EarlyStopping(monitor='val_binary_crossentropy', patience=200),
    tf.keras.callbacks.TensorBoard(logdir/name),
  ]


# In[30]:


# Similarly each model will use the same Model.compile and Model.fit settings:

def compile_and_fit(model, name, optimizer=None, max_epochs=10000):
  if optimizer is None:
    optimizer = get_optimizer()
  model.compile(optimizer=optimizer,
                loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                metrics=[
                  tf.keras.losses.BinaryCrossentropy(
                      from_logits=True, name='binary_crossentropy'),
                  'accuracy'])

  model.summary()

  history = model.fit(
    train_ds,
    steps_per_epoch = STEP_PER_EPOCH,
    epochs=max_epochs,
    validation_data=validate_ds,
    callbacks=get_callbacks(name),
    verbose=0)
  return history


# In[31]:


# Tiny Model

tiny_model = tf.keras.Sequential([
    layers.Dense(16, activation='elu', input_shape=(FEATURES,)),
    layers.Dense(1)
])

size_histories = {}

size_histories['Tiny'] = compile_and_fit(tiny_model, 'sizes/Tiny')


# In[32]:


plotter = tfdocs.plots.HistoryPlotter(metric = 'binary_crossentropy', smoothing_std=10)
plotter.plot(size_histories)
plt.ylim([0.5, 0.7])


# In[33]:


# Small Model
small_model = tf.keras.Sequential([
    # `input_shape` is only required here so that `.summary` works.
    layers.Dense(16, activation='elu', input_shape=(FEATURES,)),
    layers.Dense(16, activation='elu'),
    layers.Dense(1)
])
size_histories['Small'] = compile_and_fit(small_model, 'sizes/Small')


# In[34]:


## Medium Model
medium_model = tf.keras.Sequential([
    layers.Dense(64, activation='elu', input_shape=(FEATURES,)),
    layers.Dense(64, activation='elu'),
    layers.Dense(64, activation='elu'),
    layers.Dense(1)
])

size_histories['Medium']  = compile_and_fit(medium_model, "sizes/Medium")

# Large Model
large_model = tf.keras.Sequential([
    layers.Dense(512, activation='elu', input_shape=(FEATURES,)),
    layers.Dense(512, activation='elu'),
    layers.Dense(512, activation='elu'),
    layers.Dense(512, activation='elu'),
    layers.Dense(1)
])

size_histories['large'] = compile_and_fit(large_model, "sizes/large")

plotter.plot(size_histories)
a = plt.xscale('log')
plt.xlim([5, max(plt.xlim())])
plt.ylim([0.5, 0.7])
plt.xlabel("Epochs [Log Scale]")


# In[35]:


# As we can see Tiny model is the only one which is able to not overfit the model. Larger ones tend to overfit 
# the training set and then, the cause more wrong decissions face to unseen data


# In[36]:


# Strategies to prevent Overfitting
# Before getting into the content of this section copy the training logos from the Tiny model above to use as a baseline for comparison

shutil.rmtree(logdir/'regularizers/Tiny', ignore_errors=True)
shutil.copytree(logdir/'sizes/Tiny', logdir/'regularizers/Tiny')


# In[37]:


regularizer_histories = {}
regularizer_histories['Tiny'] = size_histories['Tiny']


# In[38]:


## Add weigth regularization
# A simple model in this contet is a model where the distribution of parameter values has less entropy. 
# Thus a common way to mitigate overfitting is to put constraints (límites) on the complexity of a network
# by forcing its weigths only to take small values, which makes the distribution of weights values more regular.
# This is called weight regularization, and it's done by addiing to the loss function of the network a cost associated
# with having large weigths.

l2_model = tf.keras.Sequential([
    layers.Dense(512, activation='elu',
                 kernel_regularizer=regularizers.l2(0.001),
                 input_shape=(FEATURES,)),
    layers.Dense(512, activation='elu',
                 kernel_regularizer=regularizers.l2(0.001)),
    layers.Dense(512, activation='elu',
                 kernel_regularizer=regularizers.l2(0.001)),
    layers.Dense(512, activation='elu',
                 kernel_regularizer=regularizers.l2(0.001)),
    layers.Dense(1)
])

regularizer_histories['l2'] = compile_and_fit(l2_model, "regularizers/l2")

plotter.plot(regularizer_histories)
plt.ylim([0.5, 0.7])


# In[39]:


# Another way to avoid overfitting is to use Dropout. Let's use L2 and Dropout together
combined_model = tf.keras.Sequential([
    layers.Dense(512, kernel_regularizer=regularizers.l2(0.0001),
                 activation='elu', input_shape=(FEATURES,)),
    layers.Dropout(0.5),
    layers.Dense(512, kernel_regularizer=regularizers.l2(0.0001),
                 activation='elu'),
    layers.Dropout(0.5),
    layers.Dense(512, kernel_regularizer=regularizers.l2(0.0001),
                 activation='elu'),
    layers.Dropout(0.5),
    layers.Dense(512, kernel_regularizer=regularizers.l2(0.0001),
                 activation='elu'),
    layers.Dropout(0.5),
    layers.Dense(1)
])

regularizer_histories['combined'] = compile_and_fit(combined_model, "regularizers/combined")

plotter.plot(regularizer_histories)
plt.ylim([0.5, 0.7])


# In[ ]:


# This model with the "Combined" regularization is obviously the best one so far.

