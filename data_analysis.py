import numpy as np
from scipy.linalg import expm
from scipy.linalg import logm


def expectation_value(density_matrix, operator):
    expectation_val = np.trace(np.dot(density_matrix, operator))

    if expectation_val.imag > 0.01:
        raise ValueError("Imaginary Expectation")
        sys.exit()

    else:
        return expectation_val.real


def has_sublists(my_list):
    return any(isinstance(item, list) for item in my_list)


def von_nuemann_entropy(density_matrix):
    vne = -np.trace(np.dot(density_matrix, logm(density_matrix) / np.log(2)))
    if vne.imag > 0.01:
        raise ValueError("Imaginary Entanglement")
        sys.exit()

    else:
        return vne.real


def quantum_relative_entropy(density_matrix_1, density_matrix_2):
    """
    Calculates the quantum relative entropy of density matrix 1 with respect to density matrix 2
    :param denstiy_matrix_1:
    :param density_matrix_2:
    :return:
    """
    qre = np.trace(np.dot(density_matrix_1, logm(density_matrix_1) / np.log(2))) - np.trace(
        np.dot(density_matrix_1, logm(density_matrix_2) / np.log(2)))

    if qre.imag > 0.01:
        raise ValueError("Imaginary Entanglement")
        sys.exit()

    else:
        return qre.real

def state_prob(v,ψ):
    p = (abs(np.dot(v, ψ)) ** 2)[0][0]

    return p


if __name__ == "__main__":
    pass
