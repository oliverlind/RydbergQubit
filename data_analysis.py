import numpy as np
import scipy.linalg
from scipy.linalg import expm
from scipy.linalg import logm
from scipy.linalg import svd
from scipy.optimize import curve_fit
import pandas as pd
from functools import reduce
from scipy.stats import norm

import vector_manipluation_tools as vm



def expectation_value(density_matrix, operator):
    expectation_val = np.trace(np.dot(density_matrix, operator))

    if expectation_val.imag > 0.01:
        raise ValueError("Imaginary Expectation")
        sys.exit()

    else:
        return expectation_val.real

def std_observable(density_matrix, operator):
    expec_A = np.trace(np.dot(density_matrix, operator))
    expec_AA  = np.trace(np.dot(np.dot(density_matrix, operator), operator))

    if expec_A.imag > 0.01 or expec_AA.imag > 0.01:
        raise ValueError("Imaginary Expectation")
        sys.exit()

    expec_A = np.abs(expec_A)
    expec_AA = np.abs(expec_AA)


    std = np.sqrt(expec_AA-(expec_A)**2)

    if std.imag > 0.01:
        raise ValueError("Imaginary Expectation")
        sys.exit()

    else:
        return std.real

def energy_spread(expec_vals, std_vals):

    expec_vals = np.array(expec_vals)
    std_vals = np.array(std_vals)

    energy_spread_frac = np.abs(std_vals/expec_vals)

    return energy_spread_frac
def has_sublists(my_list):
    return any(isinstance(item, list) for item in my_list)


def von_nuemann_entropy(density_matrix):
    vne = -np.trace(np.dot(density_matrix, logm(density_matrix)))
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
    qre = np.trace(np.dot(density_matrix_1, logm(density_matrix_1))) - np.trace(
        np.dot(density_matrix_1, logm(density_matrix_2)))

    if qre.imag > 0.01:
        raise ValueError("Imaginary Entanglement")
        sys.exit()

    else:
        return qre.real

def q_mutual_info(rdm_i, rdm_j, rdm_ij):

    qmi = -np.trace(np.dot(rdm_i, logm(rdm_i))) - np.trace(np.dot(rdm_j, logm(rdm_j))) + np.trace(np.dot(rdm_ij, logm(rdm_ij)))

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
    n_ij = np.array([[1, 0, 0, 0],
                     [0, -1, 0, 0],
                     [0, 0, -1, 0],
                     [0, 0, 0, 1]])

    expec_n_i = expectation_value(rho_i, n_i)
    expec_n_j = expectation_value(rho_j, n_i)
    expec_n_ij = expectation_value(rho_ij, n_ij)

    g_ij = expec_n_ij #- (expec_n_i * expec_n_j)

    return g_ij

def correlation_length(n, g_r):

    g_r = np.array(g_r)

    g_r = np.abs(g_r)
    x_data = np.arange(1, n, 1)

    params, covariance = curve_fit(exponential_function, x_data, g_r)

    # Extract the optimized parameters
    a_opt, b_opt = params

    # Calculate the characteristic length
    corr_length = 1 / abs(b_opt)

    return corr_length

def concurrence(rho):

    rho = rho.astype(np.complex128)
    #
    # rho_sqrt = np.sqrt(rho)
    # rho_conj = np.conj(rho)
    #
    # Y = np.array([[0,-1j],[1j,0]])
    # Y_prod = np.kron(Y,Y)
    #
    # R = np.linalg.multi_dot([rho, Y_prod, rho_conj, Y_prod])
    #
    # eigenvalues = np.sqrt(np.linalg.eigvals(R))
    # eigenvalues = np.sort(eigenvalues)[::-1]
    #
    # C = max(0,eigenvalues[0]-eigenvalues[1]-eigenvalues[2]-eigenvalues[3])

    # Pauli Y matrix
    sigma_y = np.array([[0, -1j], [1j, 0]])

    # Calculate the spin-flipped density matrix
    rho_star = np.conjugate(rho)
    spin_flipped_rho = np.kron(sigma_y, sigma_y).dot(rho_star).dot(np.kron(sigma_y, sigma_y))

    # Calculate R matrix
    sqrt_rho = scipy.linalg.sqrtm(rho)
    R = scipy.linalg.sqrtm(sqrt_rho.dot(spin_flipped_rho).dot(sqrt_rho))


    R = R.astype(np.complex128)

    # Eigenvalues of R, sorted in decreasing order
    eigenvalues = np.sort(np.real(np.linalg.eigvals(R)))[::-1]

    # Calculate concurrence
    C = max(0, eigenvalues[0] - eigenvalues[1] - eigenvalues[2] - eigenvalues[3])

    if C.imag > 0.01:
        raise ValueError("Imaginary Concurrence")
        sys.exit()

    else:
        return round(C.real,5)


def exponential_function(x, a, b):
    return a * np.exp(b * x)

def thermalization_matrix(h, obs_type='r excitations', eigenvectors_table=False):

    if obs_type == 'r excitations':

        d = h.shape[0]
        n = int(np.log2(d))

        A = np.zeros((d, d))
        id = np.eye(2)
        n_i = np.array([[0, 0],
                       [0, 1]])

        for i in range(0, n):
            id_list = [id] * n
            id_list[i] = n_i

            A_i = reduce(np.kron, id_list)

            A = A + A_i

        print(A)

        eigenvalues, eigenvectors_h = np.linalg.eig(h)

        # Sort eigenvectors based on eigenvalues
        sorted_indices = np.argsort(eigenvalues)
        sorted_eigenvalues = eigenvalues[sorted_indices]
        eigenvectors_h = eigenvectors_h[:, sorted_indices]
        print(sorted_eigenvalues)

        eigenvectors_hconj = eigenvectors_h.conj().T

        A = np.dot(eigenvectors_hconj, np.dot(A, eigenvectors_h))

        if eigenvectors_table:
            eigenvectors_df = pd.DataFrame(eigenvectors_h, columns=eigenvalues)
            print(eigenvectors_df)
            return A, eigenvectors_h

        else:
            return A


def rydberg_number_expectations(density_matrices):

    ryd_num_expec = []

    d = density_matrices[0].shape[0]
    n = int(np.log2(d))

    A = np.zeros((d, d))
    id = np.eye(2)
    n_i = np.array([[0, 0],
                    [0, 1]])

    for i in range(0, n):
        id_list = [id] * n
        id_list[i] = n_i

        A_i = reduce(np.kron, id_list)

        A = A + A_i

    for density_matrix in density_matrices:

        ryd_num_expec += [expectation_value(density_matrix,A)]

    return ryd_num_expec

def lb_bound(n, H, dt):

    id = np.eye(2**(n-1))
    n_i = np.array([[0, 0], [0, 1]])

    n_1 = np.kron(id, n_i)
    n_N = np.kron(n_i, id)


    n_1_dt = np.dot(expm(1j * H * dt), np.dot(n_1, expm(-1j * H * dt)))

    print(n_1_dt)

    com = np.dot(n_1_dt, n_N)-np.dot(n_N, n_1_dt)

    # Compute the singular values of com
    singular_values = np.linalg.svd(com, compute_uv=False)

    # The norm of the operator is the largest singular value
    com_norm = np.max(singular_values)

    return com_norm

def gaussian_distribtion(x, mu, sigma):
    print(x)
    pdf_values = norm.pdf(x, mu, sigma)

    return pdf_values











if __name__ == "__main__":

    y = gaussian_distribtion(98.1,96.7,2)
    print(y)



    # n_qubits = 2
    # state_vector = np.array([[1], [0], [0], [1]])/ np.sqrt(2)  # Example state: (|00⟩ + |11⟩) / sqrt(2)
    # rho = np.dot(state_vector, state_vector.conj().T)
    # rho1= np.eye(2)/2
    # # rho = 0.5 * np.array([[1, 0, 0, 0],
    # #                      [0, 0, 0, 0],
    # #                      [0, 0, 0, 0],
    # #                      [0, 0, 0, 1]])
    #
    # print(rho)
    #
    # print(concurrence(rho))

