import numpy as np
import timeit
import pickle

from copy import deepcopy


def calc_input_shape_with_padding(input_shape, padding):
    """
    [Flow-Function]
        1. Shape after padded are:
            row      -> row + padding[0] + padding[1]
            col      -> col + padding[2] + padding[3]
            channels -> channels

    [Params]
        input_shape (Tuple(row, col, channels)) -> Matrix shape input data
        padding (Tuple(top, bot, left, right))  -> Padding direction

    [Return]
        out_shape (Tuple(row, col))
    """
    return (
        input_shape[0] + padding[0] + padding[1],
        input_shape[1] + padding[2] + padding[3],
        input_shape[2],
    )


def pad2D(data, pad=(1, 1, 1, 1), constant_values=0):
    """
    [Flow-Function]
        1. Use np.pad for padding

    [Params]
        data (Array(row, col))                -> Matrix will be padded
        pad (Tuple(top, bottom, left, right)) -> Padding direction

    [Return]
        padded (Array(row, col))
    """
    pad = ((pad[0], pad[1]), (pad[2], pad[3]))
    padded = np.pad(data, pad, "constant", constant_values=constant_values)
    return padded


def calc_convoluted_shape(input_shape, kernel_shape, stride):
    """
    [Flow-Function]
        1. Row after convoluted = ((row - kernel_row) // stride_row) + 1
        2. Col after convoluted = ((col - kernel_col) // stride_col) + 1
        3. Channels are not affected

    [Params]
        input_shape (Tuple(row, col, channels))  -> Matrix shape input data
        kernel_shape (Tuple(row, col))           -> Kernel used for convolution
        stride (Tuple(row, col))                 -> Stride when convolution

    [Return]
        convoluted_shape (Tuple(row, col))
    """
    n_row, n_col, n_channel = input_shape
    n_kernel_row, n_kernel_col = kernel_shape
    n_stride_row, n_stride_col = stride

    convoluted_shape = (
        ((n_row - n_kernel_row) // n_stride_row) + 1,
        ((n_col - n_kernel_col) // n_stride_col) + 1,
        n_channel,
    )
    return convoluted_shape


def generate_strided_matrix2d(matrix, kernel_shape, stride):
    """
    [Flow-Function]
        1. Get num of row and columns after convoluted
        2. Using as_strided function on numpy to generate strided matrix

    [Params]
        matrix (Array(row, col)) -> Base matrix for sub matrix
        kernel_shape (Tuple(row, col)) -> kernel used
        stride (Tuple(row, col)) -> stride used

    [Return]
        output (Array(sub_matrix, row, col))
    """
    matrix_shape = matrix.shape
    adapter = (matrix_shape[0], matrix_shape[1], 1)
    num_row, num_col, _ = calc_convoluted_shape(adapter, kernel_shape, stride)

    stride_row, stride_col = matrix.strides
    shapes = (num_row, num_col, kernel_shape[0], kernel_shape[1])
    strides = (stride_row * stride[0], stride_col * stride[1], stride_row, stride_col)
    sub_matrices = np.lib.stride_tricks.as_strided(matrix, shapes, strides)

    return sub_matrices.reshape(-1, kernel_shape[0], kernel_shape[1])


def convolve2D(data, kernel, stride=(1, 1)):
    """
    [Flow-Function]
        1. Calculate output shape
        2. Create strided matrix with numpy
        3. Use vectorized matrix multiplication to get convoluted matrix

    [Params]
        data (Array(row, col))   -> Matrix data as input
        kernel (Array(row, col)) -> Matrix kernel as convolution kernel
        stride (Tuple(row, col)) -> Movement stride, where first element is striding towards row,
                                    and second element is striding toward col
    [Return]
        convoluted (Array(row, col))
    """
    n_row, n_col = data.shape
    n_kernel_row, n_kernel_col = kernel.shape

    adapter = (n_row, n_col, 1)
    convoluted_shape = calc_convoluted_shape(adapter, kernel.shape, stride)
    convoluted_shape = (convoluted_shape[0], convoluted_shape[1])

    kernel1d = kernel.reshape(
        -1,
    )

    strided_matrix = generate_strided_matrix2d(data, kernel.shape, stride)
    vectorized_col = n_kernel_row * n_kernel_col
    vectorized = strided_matrix.reshape(-1, vectorized_col)

    temp = vectorized * kernel1d
    summed = np.sum(temp, axis=-1)
    summed = summed.reshape(convoluted_shape)
    return summed


def dilate(matrix, stride):
    """
    [Flow-Method]
        1. Insert zeros first for rows
        2. Handle insert zeros for columns

    (matrix)
        [[0, 1, 2],
         [3, 4, 5],
         [6, 7, 8]]

    (stride)
        (2, 2)

    (return)
        [[0, 0, 1, 0, 2],
         [0, 0, 0, 0, 0],
         [3, 0, 4, 0, 5],
         [0, 0, 0, 0, 0],
         [6, 0, 7, 0, 8]]

    [Params]
        matrix (Array(row, col))
        stride (Tuple(row, col))

    [Return]
        result (Array(row, col))
    """
    stride_row, stride_col = stride
    _, matrix_col = matrix.shape

    # Handle rows
    if stride_row == 1:
        result_row = matrix
    else:
        result_row = []
        zeros = np.zeros(matrix_col)
        for row in matrix:
            result_row.append(row)
            for _ in range(stride_row - 1):
                result_row.append(zeros)
        result_row = np.array(result_row[: -(stride_row - 1)])

    # Handle cols
    if stride_col == 1:
        result = result_row
    else:
        result = []
        matrix_row = result_row.shape[0]
        zeros = np.zeros(matrix_row)
        for col in np.rollaxis(result_row, 1):
            result.append(col)
            for _ in range(stride_col - 1):
                result.append(zeros)
        result = np.array(result[: -(stride_col - 1)]).T

    return result


def normalize_result(pred):
    """
    [Flow-Function]
        1. Get all index from each highest value of the sequence
        ps: Assumption for this function, index <=> class (for classification label)
        (representation)
    (Class)    0     1     2
            ?????????????????????????????????????????????????????????
            ??? 0.2 ??? 0.3 ??? 0.5 ??? ??? 2
            ?????????????????????????????????????????????????????????
            ??? 0.1 ??? 0.1 ??? 0.8 ??? ??? 2
            ?????????????????????????????????????????????????????????
            ??? 0.2 ??? 0.1 ??? 0.7 ??? ??? 2
            ?????????????????????????????????????????????????????????
            ??? 0.5 ??? 0.1 ??? 0.4 ??? ??? 0
            ?????????????????????????????????????????????????????????
            ??? 0.4 ??? 0.4 ??? 0.2 ??? ??? 0 (first)
            ?????????????????????????????????????????????????????????

        [Params]
            pred Array(batch, predictions) -> multiple sequence (batch) from softmax result

        [Return]
            output Array(float)
    """
    return np.argmax(pred, axis=1)


def pooling2D(data, stride, size, shape, type):
    """
    [Flow-Function]
        1. Create new feature map with output shape
        2. Slice the matrix based on its stride and filter size
        3. Get its max/average value from sliced matrix
        4. Fill feature map with its max/average value (from every sliced matrix)

    [Params]
        data (Array(row, col))   -> Matrix data as input
        stride (Tuple(row, col)) -> Movement stride, where first element is striding towards row,
                                    and second element is striding toward col
        size (Tuple(row, col)) -> Size of the filter
        shape(Tuple(row, col, depth)) -> Size of convoluted shape
        type(String)    -> Type of the pooling

    [Return]
        pooled2D (Array(row, col))
    """
    cols = size[0] * size[1]

    strided_matrix = generate_strided_matrix2d(data, size, stride)
    max_index = []
    if type == "max":
        pooled2D = np.max(strided_matrix.reshape(-1, cols), axis=-1)
        for matrix in strided_matrix:
            max_index.append(np.unravel_index(matrix.argmax(), matrix.shape))
        max_index = np.array(max_index).reshape(shape[:2] + (2,))
    else:
        pooled2D = np.mean(strided_matrix.reshape(-1, cols), axis=-1)

    pooled2D = pooled2D.reshape(shape[:2])
    return pooled2D, max_index


def split_batch(data, batch_size):
    """
    [Flow-Function]
        1. Count total data in a batch
        2. Split data into batches
        3. Return array of tuple (x, y)

    [Params]
        data (Array(batch, row, col, channel))   -> Train data
        labels (Array(row, col)) -> Labels for train data
        batch_size (Integer) -> batch size

    [Return]
        array (Array(batches, batch, row, col, channels))
    """
    batches = []
    n_data = data.shape[0]
    n_batch = n_data // batch_size

    for i in range(n_batch):
        start = batch_size * i
        if i == n_batch - 1:
            mini_batch = data[start:]
        else:
            end = start + batch_size
            mini_batch = data[start:end]
        batches.append(mini_batch)

    array = np.array(batches, dtype=object)
    return array


def calc_params_conv(filterLength, filterWidth, filterDepth=1, totalFilter=1):
    """
    [Flow-Function]
        1. Calculate total params

    [Params]
        filterLength (Integer) -> Length of filter
        filterWidth (Integer) -> Width of filter
        filterDepth (Integer) -> Depth of filter
        totalFilter (Integer) -> Total of filter

    [Return]
        output (Integer)
    """
    return totalFilter * ((filterLength * filterWidth * filterDepth) + 1)


def calc_params_dense(input_shape, unit):
    """
    [Flow-Function]
        1. Calculate with input_shape * unit + unit (for bias)

    [Params]
        input_shape (int) -> input shape for dense layer
        unit (int) -> num of unit in dense layer

    [Return]
        output (int)
    """
    return (input_shape * unit) + unit


def save_model(model, path):
    """
    Function to save model

    [Params]
        model (Obj)
        path (str)
    """
    pickle.dump(model, open(path, "wb"))


def load_model(path):
    """
    Function to load model

    [Params]
        path (str)
    """
    return pickle.load(open(path, "rb"))
