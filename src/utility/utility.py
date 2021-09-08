import numpy as np

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
    padded = np.pad(data, pad, 'constant', constant_values=constant_values)
    return padded

def convolve2D(data, kernel, stride=(1, 1)):
    """
    [Flow-Function]
        1. Calculate output shape
        2. Create matrix ones for output later
        3. Use double for loop for convolution

    [Params]
        data (Array(row, col))   -> Matrix data as input
        kernel (Array(row, col)) -> Matrix kernel as convolution kernel
        stride (Tuple(int, int)) -> Movement stride, where first element is striding towards row,
                                    and second element is striding toward col
    [Return]
        convoluted (Array(row, col))
    """
    n_row, n_col = data.shape
    n_kernel_row, n_kernel_col = kernel.shape
    n_stride_row, n_stride_col = stride

    convoluted_shape = (
        ((n_row - n_kernel_row) // n_stride_row) + 1,
        ((n_col - n_kernel_col) // n_stride_col) + 1,
    )

    convoluted = np.ones(convoluted_shape)
    for i_row, row in enumerate(range(0, n_row - n_kernel_row + 1, n_stride_row)):
        for i_col, col in enumerate(range(0, n_col - n_kernel_col + 1, n_stride_col)):
            sliced_mat = data[row : row + n_kernel_row, col : col + n_kernel_col]
            mult_two_mat = kernel * sliced_mat
            convoluted[i_row][i_col] = np.sum(mult_two_mat)

    return convoluted


if __name__ == "__main__":
    # import cv2

    # img = cv2.imread('images/Desktop - 1.png')
    # kernel = np.array(
    #     [
    #         np.array([-1, -1, -1]),
    #         np.array([-1, 8, -1]),
    #         np.array([-1, -1, -1])
    #     ]
    # )
    # resB = convolve2D(img[:,:,0], kernel)
    # resG = convolve2D(img[:,:,1], kernel)
    # resR = convolve2D(img[:,:,2], kernel)

    # res = np.stack([resB, resG, resR], axis=-1)
    # print(res.shape)
    # cv2.imshow('convolved', res)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    mat = np.array(
        [
            np.array([1, 2, 3, 4, 5]),
            np.array([1, 2, 3, 4, 5]),
            np.array([1, 2, 3, 4, 5]),
            np.array([1, 2, 3, 4, 5]),
            np.array([1, 2, 3, 4, 5]),
        ]
    )
    ret = pad2D(mat, pad=(1, 1 ,1, 1))
    print(ret)