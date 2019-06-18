import numpy as np
from initialise import *

chain_length = 3
initial_state = create_initial_final_states(chain_length, True)
final_state = create_initial_final_states(chain_length, False)
Spin_Operator_List = Spin_List_Creator(chain_length)

def dot_product_spin_operators(qubit_position_1, qubit_position_2):
    dot_product = np.zeros((2**chain_length, 2**chain_length))
    for i in range(3):
        dot_product += Spin_Operator_List[qubit_position_1][i].dot(
                       Spin_Operator_List[qubit_position_2][i])
    return dot_product



def Heisenberg_Hamiltonian_Constructor(chain_length, couplings):
    Heisenberg = np.zeros((2**chain_length, 2**chain_length))
    print(Heisenberg)
    for i, J in enumerate(couplings):
        Heisenberg += J * dot_product_spin_operators(i, i+1)
    return Heisenberg

couplings = [1, 1]
X = Heisenberg_Hamiltonian_Constructor(chain_length, couplings)