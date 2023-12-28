import numpy as np


def initial_state(state_list):
    state_list = list(reversed(state_list))
    v = col_basis_vectors(2)[state_list[0]]  # self.bel_psi_minus(

    for i in state_list[1:]:
        u = col_basis_vectors(2)[i]
        v = np.kron(v, u)

    return v


def col_basis_vectors(n):
    return [np.array([np.eye(n)[i]]).astype(int).T for i in range(n)]


if __name__ == "__main__":

    print(initial_state([1, 0, 0]))