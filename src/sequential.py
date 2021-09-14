from typing import Callable

from src.utility import split_batch


class Sequential:
    """
    [Description]
        This call is main Deep Learning model.
        It consists of many Layer such as Dense, Conv2D, MaxPooling2D, ReLU, etc.

    [Attributes]

    [Method]

    TODO:
        - Create add layer method
        - Create summary method
        - Create predict method
        - Create convolution computation (fit)
    """

    def __init__(self, name='sequential'):
        """
        [Params]
            name (String) -> model name
        """
        self.name = name
        self.layers = []

    def add(self, layer):
        """
        [Flow-Method]
            Adding layer by appending them to self.layers

        [Params]
            layer (Layer)
        """
        self.layers.append(layer)

    def compile(self, loss: Callable, metrics: Callable):
        """
        [Flow-Method]
            Compiling data into the model

        [Params]
            loss (Callable)
            metrics (Callable)
        """
        self.loss = loss
        self.metrics = metrics

    def forward_propagation(self, data):
        """
        [Flow-Method]
            1. Iterate through self.layers
            2. Propagate the layer
            3. Use the layer's output as next layer's  input

        [Params]
            data (Array(batch, row, columns, channels))

        [Return]
            out (Array(batch, row, columns, channels))
        """
        if len(self.layers) == 0:
            raise Exception("No Layer")

        out = self.layers[0].propagate(data)
        for layer in self.layers[1:]:
            out = layer.propagate(out)
        return out


    def summary(self):
        """
        [Flow-Method]
            Iterate through layer and print out its output shape and params

        """
        params = 0
        print('Model: ' +str(self.name))
        print('----------------------------------------------------------------------')
        print('Layer (type)                     Output Shape                Param #')
        print('======================================================================')
        for idx, val in enumerate(self.layers):
            print("%-32s %-27s %-25s\n" % ((str(val.name) + '(' + str(val.__class__.__name__) + ')'), val.output_shape, val.total_params))
            params += val.total_params
            if idx != (len(self.layers) - 1):
                print('----------------------------------------------------------------------')
        print('======================================================================')
        print('Total params: {}'.format(params))
        print('Trainable params: {}'.format(params))
        print('Non-trainable params: {}'.format(params))

    def fit(self,
        x_train, 
        y_train,
        batch_size,
        learning_rate,
        epochs,
        validation_data,
        verbose,
    ):
        pass
    
    def predict(self, x):
        return self.forward_propagation(x)


