import numpy as np
from initialise import *
from Hamiltonian import *

total_chain_length = 5
initial_state = create_initial_final_states(total_chain_length, True)
final_state = create_initial_final_states(total_chain_length, False)
Spin_Operator_List = Spin_List_Creator(total_chain_length)
couplings = [1, 2, 3, 4]
final_state = final_state.transpose()

def Calculate_Fidelity(couplings, time):
    Temp_Hamiltonian = Heisenberg_Hamiltonian_Constructor(total_chain_length, couplings)
    time_evolved_matrix = time_evolution(Temp_Hamiltonian, time)
    fidelity = np.matmul(final_state, np.matmul(time_evolved_matrix, initial_state))
    fidelity_value = fidelity.item()
    probability = fidelity_value * np.conj(fidelity_value)
    return probability

F = Calculate_Fidelity(couplings, 100)
print(F)