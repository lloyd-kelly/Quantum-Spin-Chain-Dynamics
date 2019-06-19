import numpy as np
from scipy.linalg import expm, sinm, cosm
from initialise import *

def dot_product_spin_operators(chain_length, qubit_position_1, qubit_position_2):
    dot_product = np.zeros((2**chain_length, 2**chain_length))
    dot_product = dot_product.astype('complex128')
    for Spin_Operator in range(3):
        Spin_Operator_1 = Spin_Operator_List[qubit_position_1][Spin_Operator]
        Spin_Operator_2 = Spin_Operator_List[qubit_position_2][Spin_Operator]
        dot_product += np.matmul(Spin_Operator_1, Spin_Operator_2)
    return dot_product

def dot_product_spin_operators_XY(chain_length, qubit_position_1, qubit_position_2):
    dot_product = np.zeros((2**chain_length, 2**chain_length))
    dot_product = dot_product.astype('complex128')
    for Spin_Operator in range(2):
            Spin_Operator_1 = Spin_Operator_List[qubit_position_1][Spin_Operator]
            Spin_Operator_2 = Spin_Operator_List[qubit_position_2][Spin_Operator]
            dot_product += np.matmul(Spin_Operator_1, Spin_Operator_2)
    return dot_product

def Heisenberg_Hamiltonian_Constructor(chain_length, couplings):
    Heisenberg = np.zeros((2**chain_length, 2**chain_length))
    Heisenberg = Heisenberg.astype('complex128')
    for qubit_position, J in enumerate(couplings):
        Heisenberg = Heisenberg + (J * dot_product_spin_operators(chain_length, qubit_position, qubit_position+1))
    return Heisenberg

def time_evolution(heisenberg_matrix, time):
    time_scaled_matrix = -1j * time * heisenberg_matrix
    time_evol = expm(time_scaled_matrix)
    return time_evol