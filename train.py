from sklearn.model_selection import train_test_split
import numpy as np
from models.CNN_2 import model
import datetime
from variables import Variables
from reader import Data
import os
from keras.callbacks import TensorBoard
from keras.optimizers import Adam

V = Variables()

days_list = V.days_list
days_list_train, days_list_test, _, _ = train_test_split(days_list, days_list, test_size=0.2, random_state=666)
days_list_test, days_list_valid, _, _ = train_test_split(days_list_test, days_list_test, test_size=0.5,
                                                         random_state=666)

expand_dims = False       # si RNN ou CNN
data_train = Data('./data/processed/data.csv', days_list_train, expand_dims=expand_dims)
data_test = Data('./data/processed/data.csv', days_list_test, expand_dims=expand_dims)
data_valid = Data('./data/processed/data.csv', days_list_valid, expand_dims=expand_dims)

now = datetime.datetime.now().replace(microsecond=0)
name = datetime.date.today().isoformat() + '-' + now.strftime("%H-%M-%S")
os.makedirs('./experiments/' + name)

learning_rate = 1e-3
loss = 'categorical_crossentropy'
batch_size = 4
epoch = 22

gen_train = data_train.generator(batch_size)
gen_validation = data_train.generator(len(days_list_valid))
valid = gen_validation.__next__()

model = model()

model.compile(loss='mean_squared_error', optimizer='Adam')

model.fit_generator(gen_train,
                      epochs=epoch,
                      validation_data=valid,
                      steps_per_epoch=50)