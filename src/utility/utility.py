import numpy as np


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
        n_channel
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

    convoluted = np.ones(convoluted_shape)
    for i_row, row in enumerate(range(0, n_row - n_kernel_row + 1, n_stride_row)):
        for i_col, col in enumerate(range(0, n_col - n_kernel_col + 1, n_stride_col)):
            sliced_mat = data[row : row + n_kernel_row, col : col + n_kernel_col]
            mult_two_mat = kernel * sliced_mat
            convoluted[i_row][i_col] = np.sum(mult_two_mat)

    return convoluted


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
