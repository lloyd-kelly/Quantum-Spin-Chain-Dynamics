from Fidelity import *
import math
import numpy as np


def plot_changing_fidelity(evolution_time):
    """
    #Try changing J2 to see what happens to the maximum fidelity achieved
    """
    j2s = np.arange(0, 10, 0.1)
    fidelities = np.zeros(j2s.size)
    for index, j2_value in np.ndenumerate(j2s):
        coupling_trial = [1, j2_value, 1]
        fidelities[index] = find_optimal_fidelity(coupling_trial, evolution_time, XY_HAM = False)
    best_fid = np.amax(fidelities)
    first_j2_optimal = j2s[np.where(fidelities == best_fid)[0][0]]
    print(best_fid)
    print(first_j2_optimal)
    print(initial_state)
    print(final_state)
    plt.ylabel("Fidelity") 
    plt.xlabel("j2")
    plt.title("Best fidelity found for different j2s") 
    plt.plot(j2s,fidelities)
    plt.axvline(x=math.sqrt(2), color='r', linestyle='-') 
    plt.show()
    
#plot_changing_fidelity(10)

def cost_function(couplings, total_time, XY_HAM = False):
    """
    #Calculate the cost function in terms of fidelity for a given coupling over a given time, using the maximum achieved fidelity
    """
    return (1 - find_optimal_fidelity(couplings, total_time, XY_HAM))

def return_fidelities(evolution_time, chain_size):
    initial_state = create_initial_final_states(chain_size, True)
    final_state = create_initial_final_states(chain_size, False)
    final_state = final_state.transpose()
    Spin_Operator_List = Spin_List_Creator(chain_size)
    number_couplings = chain_size - 1
    couplings = np.zeros(number_couplings)
    if chain_size % 2 == 1: #even no. of couplings
        number_variable_couplings = int(number_couplings / 2)
        coupling_values = np.full(number_variable_couplings, 1) #half the size
        indicies = [i for i in range(number_variable_couplings)] + [number_couplings - 1 - i for i in range(number_variable_couplings)]
        np.put(couplings, indicies, coupling_values)
    print(cost_function(couplings, evolution_time))

return_fidelities(1, 11)
