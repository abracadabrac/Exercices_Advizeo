import pandas as pd
import random
import numpy as np
from variables import Variables

V = Variables()


class Data:
    """
    class Data permits to manage the data. I particular it provides the generator.
    """
    def __init__(self, csv_path, days_list, expand_dims=True):
        self.df = pd.read_csv(csv_path, index_col=0, parse_dates=[0])
        self.days_list = days_list

        self.expand_dims = expand_dims

    def generator(self, batch_size):
        """
        the generator provides data ready for training and testing
        :param batch_size:
        :return: the consumption on sequences of 1 day for all buildings
        """
        while True:
            days_list = self.days_list[:]
            random.shuffle(days_list)

            while len(days_list) > batch_size - 1:
                try:
                    batch_values = []
                    batch_dayOff = []
                    batch_labels = []
                    for day in days_list[:batch_size]:
                        day_values = [self.df[self.df['Bat'] == num][day]['value'] for num in range(1,11)]
                        day_values = pd.concat(day_values, axis=1)
                        batch_values.append(np.array(day_values))

                        day_labels = [self.df[self.df['Bat'] == num][day]['labels'] for num in range(1, 11)]
                        day_labels = pd.concat(day_labels, axis=1)
                        batch_labels.append(np.array(day_labels))

                        day_off = self.df[day]['day_off'][0]
                        batch_dayOff.append(day_off)

                    del days_list[:batch_size]

                    batch_values = np.array(batch_values)         # [batch_size, sequence_length (=24), nb_building (=10)]
                    batch_labels = np.array(batch_labels)         # [batch_size, sequence_length (=24), nb_building (=10)]
                    batch_dayOff = np.array(batch_dayOff)         # [batch_size]

                    assert (batch_values.shape == (batch_size, 24, 10)) & \
                           (batch_labels.shape == (batch_size, 24, 10)) & \
                           (batch_dayOff.shape == (batch_size,))

                    if self.expand_dims:
                        batch_values = np.expand_dims(batch_values, axis=-1)
                        batch_labels = np.expand_dims(batch_labels, axis=-1)

                    yield batch_values, batch_labels #, batch_dayOff

                except AssertionError:
                    print('batch of anormal size')


if __name__ == '__main__':
    data = Data('./data/processed/data.csv', V.days_list)
    gen = data.generator(4)

    values, labels = gen.__next__()
    pass