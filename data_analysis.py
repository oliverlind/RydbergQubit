import numpy as np
from scipy.linalg import expm
from scipy.linalg import logm

import vector_manipluation_tools as vm


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
    vne = -np.trace(np.dot(density_matrix, logm(density_matrix))) #/ np.log(2)
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
    p = (abs(np.dot(v.conj().T, ψ)) ** 2)[0][0]

    return p

def get_state_fidelities(q_states, state_to_test):

    v_state_to_test = vm.initial_state(state_to_test)
    state_fidelities = []

    for j in range(0, len(q_states)):
        state_fidelity = state_prob(v_state_to_test, q_states[j])
        state_fidelities += [state_fidelity]

    return state_fidelities







if __name__ == "__main__":
    pass
