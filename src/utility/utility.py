import numpy as np
import timeit

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


def convolve2D(data, kernel, stride=(1, 1)):
    """
    [Flow-Function]
        1. Calculate output shape
        2. Create matrix ones for output later
        3. Use double for loop for convolution

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
    n_stride_row, n_stride_col = stride

    adapter = (n_row, n_col, 1)
    convoluted_shape = calc_convoluted_shape(adapter, kernel.shape, stride)
    convoluted_shape = (convoluted_shape[0], convoluted_shape[1])

    vectorized = []
    kernel1d = kernel.reshape(-1,)

    # TODO: Create this vectorized matrix faster without loop
    start = timeit.default_timer()
    for row in range(0, n_row - n_kernel_row + 1, n_stride_row):
        for col in range(0, n_col - n_kernel_col + 1, n_stride_col):
            sliced_mat = data[row : row + n_kernel_row, col : col + n_kernel_col]
            vectorized.append(sliced_mat.reshape(-1,))
    vectorized = np.stack(vectorized, axis=0)

    # print(f"Create Vector Time: {timeit.default_timer() - start:4}")

    start = timeit.default_timer()
    temp = vectorized * kernel1d
    summed = np.sum(temp, axis=-1)
    summed = summed.reshape(convoluted_shape)
    # print(f"Multiplication Time: {timeit.default_timer() - start:.4f}")
    # print()
    return summed


def normalize_result(pred):
    """
    [Flow-Function]
        1. Get all index from each highest value of the sequence
        ps: Assumption for this function, index <=> class (for classification label)
        (representation)
    (Class)    0     1     2
            ╔═════╦═════╦═════╗
            ║ 0.2 ║ 0.3 ║ 0.5 ║ → 2
            ╠═════╬═════╬═════╣
            ║ 0.1 ║ 0.1 ║ 0.8 ║ → 2
            ╠═════╬═════╬═════╣
            ║ 0.2 ║ 0.1 ║ 0.7 ║ → 2
            ╠═════╬═════╬═════╣
            ║ 0.5 ║ 0.1 ║ 0.4 ║ → 0
            ╠═════╬═════╬═════╣
            ║ 0.4 ║ 0.4 ║ 0.2 ║ → 0 (first)
            ╚═════╩═════╩═════╝

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
    n_row, n_col = data.shape
    pooled2D = np.ones(shape[:2])

    for i_row_pool, i_row in enumerate(range(0, n_row - size[0] + 1, stride[0])):
        for i_col_pool, i_col in enumerate(range(0, n_col - size[1] + 1, stride[1])):
            sliced = data[
                i_row : i_row + size[0],
                i_col : i_col + size[1],
            ]
            pooled2D[i_row_pool][i_col_pool] = (
                type == "max" and sliced.max() or sliced.mean()
            )

    return pooled2D


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
        array (Array(batches, batch, row, col, channels ))
    """
    batches = []
    n_data = data.shape[0]
    n_batch = n_data // batch_size

    for i in range(batch_size):
        start = n_batch * i
        if i == batch_size - 1:
            mini_batch = data[start:]
        else:
            end = start + n_batch
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
