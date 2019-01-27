import matplotlib.pyplot as plt
from reader import Data

from variables import Variables

V = Variables()


def plot_value_labels(value, label):
    plt.figure()

    plt.subplot(1, 2, 1)
    plt.imshow(value)
    plt.title('value')

    plt.subplot(1, 2, 2)
    plt.imshow(label)
    plt.title('label')

    plt.show()


def visualise_batch_input():

    data = Data('./data/processed/data.csv', V.days_list)
    gen = data.generator(6)

    values, _ = gen.__next__()

    plt.figure(figsize=(10, 8))
    plt.title('example of a batch of size 6')

    for i in range(0,3):

        plt.subplot(1, 3, i+1)
        plt.title('day %d' % i)
        plt.imshow(values[i,:,:,0])
        plt.ylabel('hour')
        plt.xlabel('bat')

    plt.show()



if __name__ == '__main__':
    visualise_batch_input()

