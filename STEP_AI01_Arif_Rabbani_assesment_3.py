#%%
#1. Import packages
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers,optimizers,losses,metrics,callbacks,applications
import numpy as np
import os
import matplotlib.pyplot as plt
import shutil

# %%
#2. Load the data
_URL = 'https://prod-dcd-datasets-public-files-eu-west-1.s3.eu-west-1.amazonaws.com/b60351e5-29e8-49a6-a83f-1e275aea6622'
path_to_zip = tf.keras.utils.get_file('Concrete Crack Images for Classification.rar', origin=_URL, extract=True)
PATH = os.path.join(os.path.dirname(path_to_zip), 'Concrete Crack Images for Classification')
positive_class = os.path.join('Concrete Crack Images for Classification','Positive')
negative_class = os.path.join('Concrete Crack Images for Classification','Negative')
os.makedirs(PATH +'\\train')
os.makedirs(PATH +'\\train')
os.makedirs(PATH +'\\validation')
os.makedirs(PATH +'\\validation')

# %%
#3. Data preparation
#(A) Define the path to the train and validation data folder
train_path = os.path.join(PATH,'train')
val_path = os.path.join(PATH,'validation')

#(B) Define the batch size and image size
BATCH_SIZE = 32
IMG_SIZE = (160,160)

# %%
#(C) Load the data into tensorflow dataset using the specific method
train_dataset = keras.utils.image_dataset_from_directory(train_path,shuffle=True,batch_size=BATCH_SIZE,image_size=IMG_SIZE)
val_dataset = keras.utils.image_dataset_from_directory(val_path,shuffle=True,batch_size=BATCH_SIZE,image_size=IMG_SIZE)
