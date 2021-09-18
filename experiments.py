# import numpy as np
# from mlxtend.data import loadlocal_mnist
# from tqdm import tqdm
# from sklearn.preprocessing import OneHotEncoder

# from src.layer import MaxPooling2D, Flatten, Dense
# from src.layer import Conv2D, ReLU, Softmax, Sigmoid
# from src.utility import normalize_result, split_batch

# from src.neuron import NeuronConv2D, NeuronDense
# from src.sequential import Sequential

# def data():
#     matrix = np.array(
#         [
#             np.array([1, 2, 3, 2, 3]),
#             np.array([8, 3, 7, 4, 9]),
#             np.array([9, 1, 4, 7, 2]),
#             np.array([2, 6, 5, 3, 9]),
#         ]
#     )

#     data = np.stack(
#         [
#             matrix.copy(),
#             matrix.copy(),
#             matrix.copy(),
#         ],
#         axis=-1,
#     )

#     batch = np.array(
#         [
#             data.copy(),
#             data.copy(),
#             data.copy(),
#             data.copy(),
#         ]
#     )

#     return batch


# def college_data():
#     c1 = np.array([[16, 24, 32], [47, 18, 26], [68, 12, 9]])
#     c2 = np.array([[26, 57, 43], [24, 21, 12], [2, 11, 19]])
#     c3 = np.array([[18, 47, 21], [4, 6, 12], [81, 22, 13]])
#     data = np.stack([c1, c2, c3], axis=-1)
#     return np.array([data])


# def kernel():
#     return np.array(
#         [
#             np.array([1, 3]),
#             np.array([0, -1]),
#         ]
#     )


# def kernel_college():
#     return np.array(
#         [
#             [[0, -1], [1, 0]],
#             [[5, 4], [3, 2]],
#             [[16, 24], [68, -2]],
#             [[60, 22], [32, 18]],
#             [[35, 46], [7, 23]],
#             [[78, 81], [20, 42]],
#         ]
#     )

# def data2():
#     return np.array(
#         [
#             np.array([1, 0.05, 0.1, 0.25]),
#             np.array([1, 0.05, 0.1, 0.25]),
#             np.array([1, 0.05, 0.1, 0.25]),
#         ]
#     )

# x, y = loadlocal_mnist(
#     images_path="images/train-images.idx3-ubyte",
#     labels_path="images/train-labels.idx1-ubyte"
# )

# row, col = 28, 28
# x = np.expand_dims(x.reshape(-1, row, col), axis=-1)
# print(x.shape)
# encoder = OneHotEncoder()
# y = encoder.fit_transform(np.array(y).reshape(-1, 1)).toarray()
# print(y.shape)

# splitted = split_batch(x, 500)

# model = Sequential()
# model.add(Conv2D(1, (5,5), (1,1)))
# model.add(Sigmoid())
# model.add(Flatten())
# model.add(Dense(y.shape[1]))
# model.add(Softmax())
# shape = (x.shape[1], x.shape[2], x.shape[3])

# print("Building")
# model.build(shape)
# model.summary()

# # print(model.layers[-2]._neurons[0]._weights.shape)

# outs = []
# for split_data in tqdm(splitted):
#     out = model.predict(split_data)
#     norm_res = normalize_result(out)
#     outs.append((out, norm_res))

# print(outs[0][1].shape)


# # print("Predicting")
# # out = model.predict(x)
# # norm_res = normalize_result(out)
# # print(out)
# # print(norm_res)

# # layer = Dense(5)
# # layer.build(4)

# # out = layer.forward_propagation(data2())

# # neuron = NeuronDense(4)
# # print(neuron.compute(data2()))


# # params = (2, 3)
# # batch = data2()
# # print(batch.shape)
# # dense_layer = Dense(*params)
# # out = dense_layer.forward_propagation(batch)
# # print(out.shape)

# # layer = Flatten()
# # data_shape = data().shape
# # layer.build((data_shape[1], data_shape[2], data_shape[3]))
# # out = layer.forward_propagation(data())

# # dense = Dense(10, None)
# # dense.build(layer.output_shape)

# # print(out.shape)
# # print(out.shape)
# # out = np.rollaxis(out, axis=1)
# # print(out.shape)
# # out2 = dense.forward_propagation(out)
# # print(out2.shape)

# # model = Sequential()
# # model.add(Conv2D(10, (3,3), (1,1)))
# # model.add(Conv2D(20, (5,5), (1,1)))
# # model.build((32, 32, 3))
# # model.summary()
# # print(model.summary())


# # layer = Conv2D(2, (2, 2), (1, 1), input_shape=(3, 3, 3))
# # kernel = kernel_college()
# # print(layer._neurons[0]._kernels.shape)
# # # print(layer._neurons[0]._kernels[:, :, 0])
# # layer._neurons[0]._kernels[:, :, 0] = kernel[0]
# # layer._neurons[0]._kernels[:, :, 1] = kernel[1]
# # layer._neurons[0]._kernels[:, :, 2] = kernel[2]

# # layer._neurons[1]._kernels[:, :, 0] = kernel[3]
# # layer._neurons[1]._kernels[:, :, 1] = kernel[4]
# # layer._neurons[1]._kernels[:, :, 2] = kernel[5]

# # layer._neurons[0]._bias = 0
# # layer._neurons[1]._bias = 0

# # out = layer.forward_propagation(college_data())
# # print(out)

# # expected_output = np.array(
# #     [
# #         np.stack(
# #             [
# #                 [
# #                     [2171.0, 2170.0],
# #                     [5954.0, 2064.0],
# #                 ],
# #                 [[13042.0, 13575.0], [11023.0, 6425.0]],
# #             ],
# #             axis=-1,
# #         )
# #     ]
# # )
# # print(expected_output)

# # neuron = NeuronConv2D((2,2), (1,1), (4,5,3))
# # x = data()
# # print(neuron._kernels)
# # print(neuron._bias)
# # neuron.compute(data())


# # params = ((3,3), (1,1), (0,0,0,0))
# # layer = MaxPooling2D(*params)
# # print(data().shape)
# # print(layer.propagate(data())[0,:,:,0])


# # def binary_cross_entropy(y_true, y_pred, deriv=False):
# #     if deriv:
# #         return -1 * (y_true / y_pred - ((1 - y_true) / (1 - y_pred)))
# #     return (-1 * (y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))).mean()


# # from math import log
# # from numpy import mean


# # def cross_entropy(y_true, y_pred):
# #     return -sum([y_true[i] * log(y_pred[i]) for i in range(len(y_true))])


# # y_true = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
# # y_pred = np.array([0.8, 0.9, 0.9, 0.6, 0.8, 0.1, 0.4, 0.2, 0.1, 0.3])

# # print(binary_cross_entropy(y_true, y_pred, deriv=True))

# # cross = []
# # for a, b in zip(y_true, y_pred):
# #     if a == 0:
# #         cross.append(cross_entropy([1], [1 - b]))
# #     else:
# #         cross.append(cross_entropy([1], [b]))
# # print(sum(cross)/len(cross))


# # def categorical_cross_entropy(y_true, y_pred, deriv=False, epsilon=1e-12):
# #     y_pred = np.clip(y_pred, epsilon, 1.0 - epsilon)
# #     if deriv:
# #         return np.array([binary_cross_entropy(true, pred, deriv=True) for true, pred in zip(y_true, y_pred)])

# #     N = y_pred.shape[0]
# #     ce = -np.sum(y_true * np.log(y_pred)) / N
# #     return ce

# # import tensorflow as tf


# # y_true = np.array([[1, 0, 0], [0, 1, 0], [1, 0, 0], [0, 1, 0]])
# # y_pred = np.array([[0.8, 0.1, 0.1], [0.2, 0.5, 0.3], [0.4, 0.3, 0.3], [0.1, 0.9, 0.0]])

# # from src.activation import Activation
# # x = np.array([[1, 4, 2], [2, 1, 0]])
# # y_true = np.array([[1, 0, 0], [0, 1, 0]])
# # y_pred = np.array([Activation.softmax(x[0]), Activation.softmax(x[1])])


# # loss = categorical_cross_entropy(y_true, y_pred)
# # d_loss = categorical_cross_entropy(y_true, y_pred, deriv=True)
# # d_act = np.array([Activation.softmax(x[0], deriv=True), Activation.softmax(x[1], deriv=True)])

# # print('ACTIVATION\n', y_pred)
# # print('D ACTIVATION\n', d_act)
# # print('LOSS\n', loss)
# # print('D LOSS\n', d_loss)
# # print('DEDNET\n', (d_loss * d_act))

# # y_true = np.array([[0,0,0,1], [0,0,0,1]])
# # y_pred = np.array([[0.25,0.25,0.25,0.25], [0.01, 0.01, 0.01, 0.96]])

# # print(categorical_cross_entropy(y_true, y_pred, deriv=True))

# # import tensorflow as tf
# # from tensorflow.keras import Sequential
# # from tensorflow.keras.layers import Conv2D, Flatten, Dense

# # model = Sequential()
# # model.add(Conv2D(10, (3,3), activation='relu', input_shape=(32,32,1)))
# # model.add(Conv2D(20, (3,3), activation='relu'))
# # model.add(Conv2D(30, (3,3), activation='relu'))
# # model.add(Flatten())
# # model.add(Dense(10, activation='softmax'))
# # print(model.summary())


import numpy as np

a = np.array(
    [
        [0, 1, 2, 3, 4],
        [5, 6, 7, 8, 9],
        [10, 11, 12, 13, 14],
        [15, 16, 17, 18, 19],
        [20, 21, 22, 23, 24],
    ]
)

# print(a.shape)
# sub_shape = (3, 3)
# view_shape = (3, 3, 3)
# strides = a.strides
# print(view_shape)

# sub_matrices = np.lib.stride_tricks.as_strided(a, view_shape, strides)
# print(sub_matrices.shape)

# x = np.asarray([0, 1, 10, 11, 20, 21, 30, 31, 40, 41, 50, 51], np.int8).reshape(6, 2)
# x = np.asarray(range(1,26), np.int8).reshape(5,5)
# print(x.strides)
# print(a.strides)
# print(a.shape)
# from src.utility import calc_convoluted_shape

# def sub_stride_mat(matrix, kernel_shape, stride):
#     matrix_shape = matrix.shape
#     adapter = (matrix_shape[0], matrix_shape[1], 1)
#     num_row, num_col, _ = calc_convoluted_shape(adapter, kernel_shape, stride)

#     stride_row, stride_col = matrix.strides
#     shapes = (num_row, num_col, kernel_shape[0], kernel_shape[1])
#     strides = (stride_row*stride[0], stride_col*stride[1], stride_row, stride_col)
#     sub_matrices = np.lib.stride_tricks.as_strided(matrix, shapes, strides)

#     return sub_matrices.reshape(-1, kernel_shape[0], kernel_shape[1])

# kernel_shape = (2, 2)

# sub = sub_stride_mat(a, kernel_shape, (1, 2))
# print(sub.reshape(-1, 4))

# from src.utility import pooling2D

# data = np.array(
#     [
#         [1, 2, 3, 4],
#         [5, 6, 7, 8],
#         [9, 10, 11, 12],
#         [13, 14, 15, 16],
#     ]
# )

# stride = (2, 2)
# size = (2, 2)
# pooled_size = (2, 2)
# mode = "max"

# p = pooling2D(data, stride, size, pooled_size, mode)
# # print(p)

from src.neuron import NeuronDense

def data():
    return np.array([[1, 0.95, 0.1, 0.25], [0.75, 0.4, 0.3, 0.2]])
def weights():
    return np.array([1, 2, 3, 4])

dense = NeuronDense(4)
dense._weights = weights()
dense._bias = 1
out = dense.compute(data())
print(out)

