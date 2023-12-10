import numpy as np
from numpy.random import Generator, PCG64

def split_list(x_input_list, y_input_list, ratio):
    list_length = len(x_input_list)
    if list_length == 0:
        return ([], [])

    rng = Generator(PCG64(12345))
    # Shuffle the input_list indices
    indices = np.arange(list_length)
    rng.shuffle(indices)

    # Split the shuffled indices into X and 1-X portions
    split_index = int(list_length * ratio)
    indices_train = indices[:split_index]
    indices_test = indices[split_index:]

    # Use the indices to get the actual elements from the input list
    x_train_list = [x_input_list[i] for i in indices_train]
    x_test_list = [x_input_list[i] for i in indices_test]

    y_train_list = [y_input_list[i] for i in indices_train]
    y_test_list = [y_input_list[i] for i in indices_test]

    return x_train_list, x_test_list, y_train_list, y_test_list