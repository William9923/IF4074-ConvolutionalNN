from src.layer.detector import ReLU
import numpy as np

if __name__ == "__main__":
    input_shape = ((5),)
    batch = np.array([[-20, -1.0, 0.0, 1.0, 20]])
    layer = ReLU()
    layer.build(input_shape)

    errors = 1
    layer.forward_propagation(batch)
    propagate_error = layer.backward_propagation(errors)

    print(propagate_error)
