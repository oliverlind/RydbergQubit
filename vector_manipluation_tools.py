import numpy as np
import math


def initial_state(state_list):
    state_list = list(reversed(state_list))
    v = col_basis_vectors(2)[state_list[0]]  # self.bel_psi_minus(

    for i in state_list[1:]:
        u = col_basis_vectors(2)[i]
        v = np.kron(v, u)

    return v


def col_basis_vectors(n):
    return [np.array([np.eye(n)[i]]).astype(int).T for i in range(n)]


def comp_basis_vector_to_qubit_states(basis_vector):

    rows = np.shape(basis_vector)[0]
    cols = np.shape(basis_vector)[1]

    # Case for bra vector
    if cols > rows:
        n = int(np.log2(cols))

        # Convert the basis vector to binary representation
        basis_int = int(math.log2(int(''.join(map(str, reversed(basis_vector[0]))), 2)))
        binary_rep = format(basis_int, f'0{n}b')

        # Initialize a list to store individual qubit states
        qubit_states = []

        # Split the binary representation into n parts
        for i in range(n):
            qubit_binary = binary_rep[i]  # Get the binary value for the i-th qubit
            qubit_states.append(np.array([1 - int(qubit_binary), int(qubit_binary)]))  # Convert to 2D state vector

    # Case for ket vector
    if rows > cols:
        n = int(np.log2(rows))
        print(n)

        # Convert the basis vector to binary representation


        print(int(''.join(map(str, reversed(basis_vector[:, 0]))), 2))

        basis_int = int(math.log2(int(''.join(map(str, reversed(basis_vector[:, 0]))), 2)))
        binary_rep = format(basis_int, f'0{n}b')

        # Initialize a list to store individual qubit states
        qubit_states = []

        # Split the binary representation into n parts
        for i in range(n):
            qubit_binary = binary_rep[i]  # Get the binary value for the i-th qubit
            qubit_states += [np.array([[1 - int(qubit_binary), int(qubit_binary)]]).T]  # Convert to 2D state vector

    return qubit_states

if __name__ == "__main__":

    v = col_basis_vectors(2**9)
    # v[64][63] = 1
    # print(2**4)
    # print(np.shape(v))
    # print(v[64][63])


    #print(comp_basis_vector_to_qubit_states(v[511]))
    # print(9223372036854775808 -
    #       18446744073709551616)
    #
    # print(math.log2(18446744073709551616))
    #
    # print(9223372036854775808 -
    #       36893488147419103232)

