import numpy as np
from matplotlib import pyplot as plt
import math
from initialise import *
from Hamiltonian import *

total_chain_length = 4
initial_state = create_initial_final_states(total_chain_length, True)
final_state = create_initial_final_states(total_chain_length, False)
final_state = final_state.transpose()
Spin_Operator_List = Spin_List_Creator(total_chain_length)
couplings_initialise = [0, 0]


def Calculate_Fidelity(couplings, time, XY_HAM = False):
    """
    Calculate fidelity for given couplings over a given time t. Outputs the probability of achieving 
    that coupling
    """
    Temp_Hamiltonian = Heisenberg_Hamiltonian_Constructor(total_chain_length, couplings, XY_HAM)
    time_evolved_matrix = time_evolution(Temp_Hamiltonian, time)
    fidelity = np.matmul(final_state, np.matmul(time_evolved_matrix, initial_state))
    fidelity_value = fidelity.item()
    probability = fidelity_value * np.conj(fidelity_value)
    return probability

def plot_fidelity_overtime(couplings, total_time):
    """
    Plots the fidelity over a given time with specified coupling
    """
    x = np.arange(0, total_time, 0.1)
    y = np.zeros(x.size)
    for index, value in np.ndenumerate(x):
        y[index] = Calculate_Fidelity(couplings, value, XY_HAM = False)
    plt.ylabel("Probability") 
    plt.xlabel("Time") 
    plt.title("Probability of perfect fidelity") 
    plt.plot(x,y) 
    plt.show()

#plot_fidelity_overtime([1, math.sqrt(2), 1], 20)

def find_optimal_fidelity(couplings, total_time, XY_HAM = False):
    """
    Finds the MAXIMUM fidelity achieved by the evolution when evolved over total_time
    """
    times = np.arange(0, total_time, 0.1)
    fidelities = np.zeros(times.size)
    for index, value in np.ndenumerate(times):
        fidelities[index] = Calculate_Fidelity(couplings, value, XY_HAM)
    return np.amax(fidelities)


