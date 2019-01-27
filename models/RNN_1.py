import keras
from keras import backend as K
from keras import layers
from keras.models import Model, Sequential
from reader import Data

from variables import Variables

V = Variables()


def model(sequence_length=24, nb_buildings=10):
    """
    A first model without recurrence
    :param sequence_length: number of hours in the day, 24 then
    :param nb_buildings: number of buildings, 10 then
    :return: a keras model with mono-dimensional convolutions and dense layers on the channel dimension
    """

    model = Sequential()

    model.add(layers.Bidirectional(layers.LSTM(5, return_sequences=True, dropout=0.1), name="LSTM0",
                                   input_shape=(sequence_length, nb_buildings)))
    model.add(layers.Bidirectional(layers.LSTM(5, return_sequences=True, dropout=0.1), name="LSTM1"))
    model.add(layers.Bidirectional(layers.LSTM(5, return_sequences=True, dropout=0.1), name="LSTM2"))

    return model


if __name__ == '__main__':
    data = Data('../data/processed/data.csv', V.days_list, expand_dims=False)
    gen = data.generator(11)

    values, labels = gen.__next__()

    model = model()

    output = model.predict(values)
    print(output.shape - labels.shape)

    pass
