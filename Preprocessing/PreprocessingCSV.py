#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Load CSV Data

# This tutorial provides and example of how to load CSV data from a file into a tf.data.Dataset
# The data used in this tutorial are taken from the Titanic passenger list.
# The model will predict the likelihood a passenger survided based
# on characteristics like age, gender, ticket class and wheter the person
# was travelling alone

import functools
import numpy as np
import tensorflow as tf

TRAIN_DATA_URL = "https://storage.googleapis.com/tf-datasets/titanic/train.csv"
TEST_DATA_URL = "https://storage.googleapis.com/tf-datasets/titanic/eval.csv"

train_file_path = tf.keras.utils.get_file("train.csv", TRAIN_DATA_URL)
test_file_path = tf.keras.utils.get_file("eval.csv", TEST_DATA_URL)


# In[3]:


# Load data
get_ipython().system('head {train_file_path}')


# In[7]:


# You can load this using pandas, and pass the Numpy arrays to Tensorflow.

LABEL_COLUMN = 'survived'
LABELS= [0,1]

# Now read the CSV data from the file and create a dataset

def get_dataset(file_path, **kwargs):
    dataset = tf.data.experimental.make_csv_dataset(
        file_path,
        batch_size=5,
        label_name=LABEL_COLUMN,
        na_value='?',
        num_epochs=1,
        ignore_errors= True,
        **kwargs)
    return dataset

raw_train_data = get_dataset(train_file_path)
raw_test_data = get_dataset(test_file_path)

def show_batch(dataset):
  for batch, label in dataset.take(1):
    for key, value in batch.items():
      print("{:20s}: {}".format(key,value.numpy()))

show_batch(raw_train_data)


# In[8]:


# As you can see, the columns in the CSV are named. The dataset
# constructor will pick these names up automatically.

SELECT_COLUMNS = ['survived', 'age', 'n_siblings_spouses', 'class', 'deck', 'alone']

temp_dataset = get_dataset(train_file_path, select_columns=SELECT_COLUMNS)

show_batch(temp_dataset)


# In[9]:


# Data Preprocessing
# A CSV file can contain a variety of data types. Typically you
# want to convert those mixed types to a fixed length vector before
# feeding the data into your model

# Tensorflow has a built-in system for describing common input
# conversions: tf.features_columns

# Continuous data: If your data is already ina appropiate numeric,
# format, you can pack the data into a vector before passing it of the model

SELECT_COLUMNS = ['survived', 'age', 'n_siblings_spouses', 'parch', 'fare']
DEFAULTS = [0, 0.0, 0.0, 0.0, 0.0]
temp_dataset = get_dataset(train_file_path, 
                           select_columns=SELECT_COLUMNS,
                           column_defaults = DEFAULTS)

show_batch(temp_dataset)


# In[20]:


example_batch, labels_batch = next(iter(temp_dataset)) 
def pack(features, label):
  return tf.stack(list(features.values()), axis=-1), label

packed_dataset = temp_dataset.map(pack)

for features, labels in packed_dataset.take(1):
  print(features.numpy())
  print()
  print(labels.numpy())

show_batch(raw_train_data)


# In[21]:



example_batch, labels_batch = next(iter(temp_dataset))
# So define a more general preprocessor that selects a list of numeric 
# features and pack them into a single column

class PackNumericFeatures(object):
  def __init__(self, names):
    self.names = names

  def __call__(self, features, labels):
    numeric_features = [features.pop(name) for name in self.names]
    numeric_features = [tf.cast(feat, tf.float32) for feat in numeric_features]
    numeric_features = tf.stack(numeric_features, axis=-1)
    features['numeric'] = numeric_features

    return features, labels

NUMERIC_FEATURES = ['age','n_siblings_spouses','parch', 'fare']

packed_train_data = raw_train_data.map(
    PackNumericFeatures(NUMERIC_FEATURES))

packed_test_data = raw_test_data.map(
    PackNumericFeatures(NUMERIC_FEATURES))

show_batch(packed_train_data)


# In[25]:


# Data Normalization
# Continuous data should always be normalized
example_batch, labels_batch = next(iter(packed_train_data)) 
import pandas as pd
desc = pd.read_csv(train_file_path)[NUMERIC_FEATURES].describe()
desc


# In[26]:


MEAN = np.array(desc.T['mean'])
STD = np.array(desc.T['std'])

def normalize_numeric_data(data, mean, std):
    #Center data
    return (data-mean)/std

# See what you just created.
normalizer = functools.partial(normalize_numeric_data, mean=MEAN, std=STD)
numeric_column = tf.feature_column.numeric_column('numeric', normalizer_fn=normalizer, shape=[len(NUMERIC_FEATURES)])
numeric_columns = [numeric_column]
numeric_column


# In[27]:


example_batch['numeric']


# In[28]:


numeric_layer = tf.keras.layers.DenseFeatures(numeric_columns)
numeric_layer(example_batch).numpy()


# In[29]:


# Tensorflow tutorial is extremely confuse. A good explanation 
# of this example is: https://medium.com/@a.ydobon/tensorflow-2-0-load-csv-to-tensorflow-2634f7089651

# Categorical Data
# Some of the columns in the CSV data are categorical columns. That is,
# the content should be one of a limited set of options

# Use the tf.feature_column API to create a collection with a 
# tf.feature_column.indicator_column for each categorical column

CATEGORIES = {
    'sex': ['male', 'female'],
    'class' : ['First', 'Second', 'Third'],
    'deck' : ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'],
    'embark_town' : ['Cherbourg', 'Southhampton', 'Queenstown'],
    'alone' : ['y', 'n']
}

categorical_columns = []
for feature, vocab in CATEGORIES.items():
  cat_col = tf.feature_column.categorical_column_with_vocabulary_list(
        key=feature, vocabulary_list=vocab)
  categorical_columns.append(tf.feature_column.indicator_column(cat_col))
    
# See what you just created. # It indicates possibilies in our 
# dataset for each feature
categorical_columns


# In[35]:


categorical_layer = tf.keras.layers.DenseFeatures(categorical_columns)
print(categorical_layer(example_batch).numpy()[0])


# In[36]:


# Combined preprocessing layer
# Add the two features column collections and pass them to an 
# tf.keras.layers.DenseFeatures to create an input layer that
# will extract and preprocess both input layers

preprocessing_layer = tf.keras.layers.DenseFeatures(categorical_columns+numeric_columns)
print(preprocessing_layer(example_batch).numpy()[0])


# In[38]:


# Build the model
# Build a tf.keras.Sequential starting with preprocessing_layer

model = tf.keras.Sequential([
    preprocessing_layer,
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dense(1)
])

model.compile(
    loss= tf.keras.losses.BinaryCrossentropy(from_logits=True),
    optimizer = 'adam',
    metrics = ['accuracy']
)


# In[39]:


# Train, evaluate and predict
train_data = packed_train_data.shuffle(500)
test_data = packed_test_data

model.fit(train_data,epochs=20)


# In[42]:


predictions = model.predict(test_data)

# Show some results
for prediction, survived in zip(predictions[:10], list(test_data)[0][1][:10]):
  prediction = tf.sigmoid(prediction).numpy()
  print("Predicted survival: {:.2%}".format(prediction[0]),
        " | Actual outcome: ",
        ("SURVIVED" if bool(survived) else "DIED"))


# In[ ]:


# A better explanation of all this process is in the link bellow:
# https://medium.com/@a.ydobon/tensorflow-2-0-load-csv-to-tensorflow-2634f7089651


# In[44]:


show_batch(train_data)


# In[ ]:




