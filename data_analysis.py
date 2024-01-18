import numpy as np
from scipy.linalg import expm
from scipy.linalg import logm
from scipy.linalg import svd

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
    vne = -np.trace(np.dot(density_matrix, logm(density_matrix)) / np.log(2))
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

def q_mutual_info(rdm_i, rdm_j, rdm_ij):

    qmi = -np.trace(np.dot(rdm_i, logm(rdm_i)/np.log(2))) - np.trace(np.dot(rdm_j, logm(rdm_j)/np.log(2))) + np.trace(np.dot(rdm_ij, logm(rdm_ij) / np.log(2)))

    if qmi.imag > 0.01:
        raise ValueError("Imaginary Entanglement")
        sys.exit()

    else:
        return qmi.real

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

def croberg_density_density_correlation_map():
    pass

def schmidt_coeffients_red(state, n, n_A):
    # Reshape the state vector to a matrix
    state_matrix = state.reshape(2**n_A, 2**int(n-n_A))

    print(state_matrix)

    # Perform singular value decomposition (SVD)
    s = svd(state_matrix, compute_uv=False)

    # Extract Schmidt coefficients and states
    schmidt_coefficients = s
    # schmidt_states_A = U[:, :len(schmidt_coefficients)]
    # schmidt_states_B = Vdag[:len(schmidt_coefficients), :]

    # Construct the reduced density matrix for subsystem A
    rho_A = np.dot(state_matrix, np.conjugate(state_matrix.T))

    return schmidt_coefficients, rho_A

def correlation_funtction(rho_ij, rho_i, rho_j):

    n_i = np.array([[0, 0], [0, 1]])
    n_ij = np.array([[0, 0, 0, 0],
                     [0, 0, 0, 0],
                     [0, 0, 0, 0],
                     [0, 0, 0, 1]])

    expec_n_i = expectation_value(rho_i, n_i)
    expec_n_j = expectation_value(rho_j, n_i)
    expec_n_ij = expectation_value(rho_ij, n_ij)

    g_ij = expec_n_ij - (expec_n_i * expec_n_j)

    return g_ij


















if __name__ == "__main__":
    n_qubits = 2
    state_vector = np.array([0, 1, 0, 0]) #/ np.sqrt(2)  # Example state: (|00⟩ + |11⟩) / sqrt(2)
    subsystem_A_indices = 1  # Qubits in subsystem A

    schmidt_coeffs, reduced_density_matrix_A = schmidt_decomposition(state_vector,
                                                                                                         2,
                                                                                                         1)

    # Print the results
    print("Schmidt Coefficients:", schmidt_coeffs)
    print("Reduced Density Matrix for Subsystem A:\n", reduced_density_matrix_A)
