import sys
import numpy as np
import scipy as sc
from scipy import sparse
from scipy.linalg import expm
from scipy.linalg import logm
from matplotlib import pyplot as plt
from rydberg_hamiltonian_1d import RydbergHamiltonian1D

import detuning_regimes
import data_analysis as da


class AdiabaticEvolution(RydbergHamiltonian1D):
    def __init__(self, n, t, dt, δ_start, δ_end, rabi_osc=False, no_int=False, detuning_type=None,
                 single_addressing_list=None):
        super().__init__(n)
        self.t = t
        self.dt = dt
        self.steps = int(t / dt)
        self.δ_start = δ_start
        self.δ_end = δ_end
        self.times = np.linspace(0, t, self.steps)
        self.reduced_den_initial = np.zeros((2, 2))
        self.two_π = 2 * np.pi

        if rabi_osc:
            self.detunning = np.zeros(self.steps)

        if no_int:
            self.C_6 = 0

        self.detunning = detuning_regimes.global_detuning(t, dt, δ_start, δ_end, detuning_type)

        if single_addressing_list is not None:
            self.detunning = detuning_regimes.single_addressing(self.t, self.dt, self.δ_start, self.δ_end,
                                                                single_addressing_list)

    def ground_state(self):
        g = np.zeros((2 ** self.n, 1))
        g[0, 0] = 1

        return g

    def time_evolve(self, density_matrix=False, rydberg_fidelity=False, bell_fidelity=False, bell_fidelity_types=None,
                    states_list=False, reduced_density_matrix_pair=False):
        ψ = self.ground_state()
        j = self.row_basis_vectors(2 ** (self.n - 1))
        rydberg_fidelity_list = [[] for _ in range(self.n)]
        density_matrices = []
        bell_fidelity_list = [[] for _ in range(self.n - 1)]
        bell_fidelity_dict = {'psi plus': [[] for _ in range(self.n - 1)], 'psi minus': [[] for _ in range(self.n - 1)],
                              'phi plus': [[] for _ in range(self.n - 1)], 'phi minus': [[] for _ in range(self.n - 1)]}
        states = []
        rdms_pairs_list = [[] for _ in range(self.n - 1)]

        for k in range(0, self.steps):

            ψ = np.dot(expm(-1j * self.hamiltonian_matrix(self.detunning[:, k]) * self.dt), ψ)

            if density_matrix:
                density_m = np.dot(ψ, ψ.conj().T)
                density_matrices += [density_m]

            if rydberg_fidelity:
                self.rydberg_fidelity(rydberg_fidelity_list, ψ)

            if states_list:
                states += [ψ]

            if bell_fidelity:
                self.bell_state_fidelity(bell_fidelity_list, ψ)

            if bell_fidelity_types is not None:
                for i in bell_fidelity_types:
                    self.bell_state_fidelity(bell_fidelity_dict[i], ψ, bell_type=i)

            if reduced_density_matrix_pair:
                for i in range(1, self.n):
                    rdm = self.reduced_density_matrix_pair(ψ, i, i + 1)
                    rdms_pairs_list[i - 1] += [rdm]

        if rydberg_fidelity and states_list:
            return rydberg_fidelity_list, states

        elif bell_fidelity_types and rydberg_fidelity:
            return rydberg_fidelity_list, bell_fidelity_dict

        elif rydberg_fidelity and bell_fidelity:
            return rydberg_fidelity_list, bell_fidelity_list

        elif rydberg_fidelity:
            return rydberg_fidelity_list

        elif density_matrix:
            return density_matrices

        elif states_list:
            return states

        elif bell_fidelity:
            return bell_fidelity_list

        elif bell_fidelity_types:
            return bell_fidelity_dict

        elif reduced_density_matrix_pair:
            return rdms_pairs_list

        else:
            return ψ

    def expectation_vals(self, eval, evec):
        expec_vals = []
        for i in range(0, self.dimension):
            v = self.row_basis_vectors(self.dimension)[i]
            expec_val = 0
            for j in range(0, self.dimension):
                expec_val = expec_val + (eval[j] * np.dot(v, evec[:, j]))[0]

            expec_vals += [expec_val]

        return expec_vals

    def eigenvalue_evolve(self, show=False):
        eigenvalues = []
        expectation_vals = []
        # self.linear_step_detunning()

        for k in range(0, self.steps):
            h_m = self.hamiltonian_matrix(self.detunning[k])
            eval, evec = np.linalg.eigh(h_m)
            expectation_val = self.expectation_vals(eval, evec)
            # eigenvalue = np.sort(eigenvalue)
            # eigenvalues += [eval]
            expectation_vals += [expectation_val]

        # g = np.array([1,0])
        # expec_val = eval[0]*np.dot(g,evec[:,0]) + eval[1]*np.dot(g,evec[:,1])
        # print(expec_val)
        #
        # print(self.expectation_vals(eval,evec))

        expectation_vals = np.array(expectation_vals)

        if show:

            if self.n == 1:

                for i in range(0, self.dimension):
                    if i == 0:
                        plt.plot(self.times, expectation_vals[:, i], label=f'|g⟩')

                    else:
                        plt.plot(self.times, expectation_vals[:, i], label=f'|r⟩')

                plt.xlabel('Δ (2πxMHz)')
                plt.ylabel('Energy Eigenvalue')

                plt.legend()

                plt.show()

            if n == 2:
                states = ['|g⟩']
                for i in range(0, self.dimension):
                    plt.plot(self.detunning, eigenvalues[:, i], label=f'{i}')

                plt.title(f'Rabi Oscillations ( {"$R_{b}$"}={round(self.r_b, 2)}μm, a={self.a}μm)')
                plt.xlabel('Time')  # 'Δ (2πxMHz)')
                plt.ylabel('Energy Eigenvalue')

                # plt.legend()

                plt.show()

    def linear_detunning(self):
        start_steps = int(0.1 * self.steps)

        start_detuning = [self.δ_start] * start_steps

        sweep = np.linspace(self.δ_start, self.δ_end, self.steps - 2 * start_steps)

        end_detuning = [self.δ_end] * start_steps

        detunning = np.hstack((start_detuning, sweep, end_detuning))

        self.detunning = detunning

        # x = np.arange(0,self.steps)
        #
        # plt.plot(x, detunning)
        #
        # plt.show()

    def linear_step_detunning(self, show=False):

        steps = int(self.steps / 3)
        remainder = self.steps - (3 * steps)

        sweep_1 = np.linspace(self.δ_start, 0, steps)

        flat = [0] * steps

        sweep_2 = np.linspace(0, self.δ_end, steps + remainder)

        detunning = np.hstack((sweep_1, flat, sweep_2))

        self.detunning = detunning

        if show:
            x = np.arange(0, self.steps)

            plt.plot(x, detunning)

            plt.show()

    def linear_detunning_quench(self, show=False):

        linear_steps = self.steps // 2

        sweep = np.linspace(self.δ_start, self.δ_end, linear_steps)

        quench = np.linspace(0, 0, self.steps - linear_steps)

        detunning = np.hstack((sweep, quench))

        if show:
            x = np.arange(0, self.steps)

            plt.plot(x, detunning)

            plt.show()

        return detunning

    def linear_detunning_flat(self, show=False):

        linear_steps = self.steps // 2

        sweep = np.linspace(self.δ_start, self.δ_end, linear_steps)

        quench = np.linspace(self.δ_end, self.δ_end, self.steps - linear_steps)

        detunning = np.hstack((sweep, quench))

        if show:
            x = np.arange(0, self.steps)

            plt.plot(x, detunning)

            plt.show()

        return detunning

    def cubic_detunning(self):
        start_steps = int(0.1 * self.steps)

        start_detuning = [self.δ_start] * start_steps

        sweep = np.linspace(self.δ_start, self.δ_end, self.steps - 2 * start_steps)

        end_detuning = [self.δ_end] * start_steps

        detunning = np.hstack((start_detuning, sweep, end_detuning))

        x = np.arange(0, self.steps - 2)

        plt.plot(x, detunning)

        plt.show()

    def two_atom_evolve(self, density_matrix, j, rydberg_fidelity={}, n_1list=[], n_2list=[]):

        # Initialise reduced density matrices to 0
        density_matrix_one = self.reduced_den_initial
        density_matrix_two = self.reduced_den_initial

        # Calculate reduced density matrices for each atom
        for i in range(0, 2 ** (self.n - 1)):
            j_bra = j[i]
            j_ket = j[i].conj().T

            # Atom 1
            m_left = np.kron(self.id, j_bra)
            m_right = np.kron(self.id, j_ket)
            density_matrix_one = density_matrix_one + np.dot(m_left, np.dot(density_matrix, m_right))

            # Atom 2
            m_left = np.kron(j_bra, self.id)
            m_right = np.kron(j_ket, self.id)
            density_matrix_two = density_matrix_two + np.dot(m_left, np.dot(density_matrix, m_right))

        # Rydberg Fidelity Atom 1
        n_1 = abs(np.trace(np.dot(density_matrix_one, self.ni_op)))
        n_1list += [n_1]

        # Rydberg Fidelity Atom 2
        n_2 = abs(np.trace(np.dot(density_matrix_two, self.ni_op)))
        n_2list += [n_2]

        rydberg_fidelity['Atom 1'] = n_1list
        rydberg_fidelity['Atom 2'] = n_2list

        return rydberg_fidelity

    def three_atom_evolve(self, density_matrix, j, rydberg_fidelity={}, n_1list=[], n_2list=[], n_3list=[]):

        # Initialise reduced density matrices to 0
        density_matrix_one = self.reduced_den_initial
        density_matrix_two = self.reduced_den_initial
        density_matrix_three = self.reduced_den_initial

        # Calculate reduced density matrices for each atom
        for i in range(0, 2 ** (self.n - 1)):
            j_bra = j[i]
            j_ket = j[i].conj().T

            # Atom one
            m_left = np.kron(self.id, j_bra)
            m_right = np.kron(self.id, j_ket)
            density_matrix_one = density_matrix_one + np.dot(m_left, np.dot(density_matrix, m_right))

            # Atom two
            self.comp_basis_vector_to_qubit_states(j_bra)

            # Atom three
            m_left = np.kron(j_bra, self.id)
            m_right = np.kron(j_ket, self.id)
            density_matrix_two = density_matrix_two + np.dot(m_left, np.dot(density_matrix, m_right))

        # Rydberg Fidelity Atom 1
        n_1 = abs(np.trace(np.dot(density_matrix_one, self.ni_op)))
        n_1list += [n_1]

        # Rydberg Fidelity Atom 2
        n_2 = abs(np.trace(np.dot(density_matrix_two, self.ni_op)))
        n_2list += [n_2]

        rydberg_fidelity['Atom 1'] = n_1list
        rydberg_fidelity['Atom 2'] = n_2list

        return rydberg_fidelity

    @staticmethod
    def comp_basis_vector_to_qubit_states(basis_vector):

        rows = np.shape(basis_vector)[0]
        cols = np.shape(basis_vector)[1]

        # Case for bra vector
        if cols > rows:
            n = int(np.log2(cols))

            # Convert the basis vector to binary representation
            basis_int = int(np.log2(int(''.join(map(str, reversed(basis_vector[0]))), 2)))
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

            # Convert the basis vector to binary representation
            basis_int = int(np.log2(int(''.join(map(str, reversed(basis_vector[:, 0]))), 2)))
            binary_rep = format(basis_int, f'0{n}b')

            # Initialize a list to store individual qubit states
            qubit_states = []

            # Split the binary representation into n parts
            for i in range(n):
                qubit_binary = binary_rep[i]  # Get the binary value for the i-th qubit
                qubit_states += [np.array([[1 - int(qubit_binary), int(qubit_binary)]]).T]  # Convert to 2D state vector

        return qubit_states

    @staticmethod
    def row_basis_vectors(n):
        return [np.array([np.eye(n)[i]]).astype(int) for i in range(n)]

    @staticmethod
    def col_basis_vectors(n):
        return [np.array([np.eye(n)[i]]).astype(int).T for i in range(n)]

    def reduced_density_matrix(self, ψ, i):

        if self.n == 1:
            reduced_density_matrix = np.dot(ψ, ψ.conj().T)

        else:
            dim_sub = 2 ** (self.n - 1)

            m_left_list = []
            m_right_list = []

            # Produce list of matrices on left side of sum
            row_basis_vectors = self.row_basis_vectors(dim_sub)
            for row_vector in row_basis_vectors:
                bra_vectors = self.comp_basis_vector_to_qubit_states(row_vector)
                bra_vectors.insert(i - 1, self.id)

                m_left = bra_vectors[0]  # initialise left matrix

                for bra in bra_vectors[1:]:
                    m_left = np.kron(m_left, bra)  # taking tensor product left to right

                m_left_list += [m_left]

            # Produce list of matrices on right side of sum
            col_basis_vectors = self.col_basis_vectors(dim_sub)
            for col_vector in col_basis_vectors:
                ket_vectors = self.comp_basis_vector_to_qubit_states(col_vector)
                ket_vectors.insert(i - 1, self.id)

                m_right = ket_vectors[0]  # initialise right matrix

                for ket in ket_vectors[1:]:
                    m_right = np.kron(m_right, ket)  # taking tensor product left to right

                m_right_list += [m_right]

            reduced_density_matrix = np.zeros((2, 2))

            for j in range(0, dim_sub):
                m_left = m_left_list[j]
                m_right = m_right_list[j]
                reduced_density_matrix = reduced_density_matrix + np.dot(np.dot(m_left, ψ), np.dot(ψ.conj().T, m_right))

        return reduced_density_matrix

    def reduced_density_matrix_pair(self, ψ, i, j):

        if self.n == 1:
            sys.exit()

        dim_sub = 2 ** (self.n - 2)

        m_left_list = []
        m_right_list = []

        # Produce list of matrices on left side of sum
        row_basis_vectors = self.row_basis_vectors(dim_sub)
        for row_vector in row_basis_vectors:
            bra_vectors = self.comp_basis_vector_to_qubit_states(row_vector)
            bra_vectors.insert(i - 1, self.id)
            bra_vectors.insert(j - 1, self.id)

            m_left = bra_vectors[0]  # initialise left matrix

            for bra in bra_vectors[1:]:
                m_left = np.kron(m_left, bra)  # taking tensor product left to right

            m_left_list += [m_left]

        # Produce list of matrices on right side of sum
        col_basis_vectors = self.col_basis_vectors(dim_sub)
        for col_vector in col_basis_vectors:
            ket_vectors = self.comp_basis_vector_to_qubit_states(col_vector)
            ket_vectors.insert(i - 1, self.id)
            ket_vectors.insert(j - 1, self.id)

            m_right = ket_vectors[0]  # initialise right matrix

            for ket in ket_vectors[1:]:
                m_right = np.kron(m_right, ket)  # taking tensor product left to right

            m_right_list += [m_right]

        reduced_density_matrix = np.zeros((4, 4))

        for j in range(0, dim_sub):
            m_left = m_left_list[j]
            m_right = m_right_list[j]
            reduced_density_matrix = reduced_density_matrix + np.dot(np.dot(m_left, ψ), np.dot(ψ.conj().T, m_right))

        return reduced_density_matrix

    def rydberg_fidelity(self, rydberg_fidelity_list, ψ):

        for i in range(1, self.n + 1):
            rdm = self.reduced_density_matrix(ψ, i)
            rf = da.expectation_value(rdm, self.ni_op)
            rydberg_fidelity_list[i - 1] += [rf]

    def bell_state_fidelity(self, bell_fidelity_list, ψ, bell_type='psi minus'):

        if bell_type == 'psi plus':
            bell_state = 0.5 * np.array([[0, 0, 0, 0],
                                         [0, 1, 1, 0],
                                         [0, 1, 1, 0],
                                         [0, 0, 0, 0]])

        elif bell_type == 'psi minus':
            bell_state = 0.5 * np.array([[0, 0, 0, 0],
                                         [0, 1, -1, 0],
                                         [0, -1, 1, 0],
                                         [0, 0, 0, 0]])

        elif bell_type == 'phi plus':
            bell_state = 0.5 * np.array([[1, 0, 0, 1],
                                         [0, 0, 0, 0],
                                         [0, 0, 0, 0],
                                         [1, 0, 0, 1]])

        elif bell_type == 'phi minus':
            bell_state = 0.5 * np.array([[1, 0, 0, -1],
                                         [0, 0, 0, 0],
                                         [0, 0, 0, 0],
                                         [-1, 0, 0, 1]])

        else:
            sys.exit()

        for i in range(1, self.n):
            rdm = self.reduced_density_matrix_pair(ψ, i, i + 1)
            bf = da.expectation_value(rdm, bell_state)
            bell_fidelity_list[i - 1] += [bf]


if __name__ == "__main__":
    t = 5
    dt = 0.01
    n = 3
    δ_start = -200
    δ_end = 200

    evol = AdiabaticEvolution(n, t, dt, δ_start, δ_end)

    rdms = evol.time_evolve(reduced_density_matrix_pair=True)
