import numpy as np


def expectation_value(density_matrix, operater):
    expectation_val = np.trace(np.dot(density_matrix, operater))

    if expectation_val.imag > 0.01:
        raise ValueError("Imaginary Expectation")
        sys.exit()

    else:
        return expectation_val.real
