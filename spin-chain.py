import numpy as np
from scipy.linalg import expm, sinm, cosm
from matplotlib import pyplot as plt
import math
from scipy.optimize import minimize
from psopy import minimize as minimize_pso
from psopy import init_feasible
import argparse
import warnings
warnings.filterwarnings('ignore')

#pauli matrices
Sx = np.array([[0, 1], [1, 0]])
Sy = np.array([[0, -1j], [1j, 0]])
Sz = np.array([[1, 0], [0, -1]])

#up and down states
up = np.array([[1], 
               [0]])
down = np.array([[0],
                 [1]])

qubit_identity = np.identity(2)

def tensor_product_initialiser(number_particles, state, iterated_state, first):
    """
    Creates a tensor product over a certain number of particles, with an initial state, and an iterated 
    state that attaches however many times. If first is true, the initial state remains as the first state,
    otherwise it moves to last.
    """
    states_created = 1
    while states_created < number_particles:
        if first == True:
            state = np.kron(state, iterated_state)
        else:
            state = np.kron(iterated_state, state)
        states_created += 1
    return state

def create_initial_final_states(chain_length, initial):
    """
    Creates the initial and final states of the chain, initial being a boolean which we can set to True for
    the initial, False for the final.
    """
    return tensor_product_initialiser(chain_length, up, down, initial)

def chain_operator_constructor(chain_length, chain_position, qubit_operator):
    """
    Takes an operator acting on a single qubit, in a given position of the chain. Outputs the operator on the entire 
    chain.
    """
    operator_left = tensor_product_initialiser(chain_position, qubit_operator, qubit_identity, False)
    operator = tensor_product_initialiser(chain_length - chain_position + 1, operator_left, qubit_identity, True)
    return operator


def Spin_List_Creator(chain_length):
    """
    Creates list of size chain_length of the Spin operators for each site, where S = [Sx, Sy, Sz]. Indexed from 0.
    """
    Spin_List = []
    for qubit_chain_position in range(chain_length):
        Qubit_Spin_Array = np.array([chain_operator_constructor(chain_length, qubit_chain_position + 1, Sx), 
                                      chain_operator_constructor(chain_length, qubit_chain_position + 1, Sy),
                                      chain_operator_constructor(chain_length, qubit_chain_position + 1, Sz)])
        Spin_List.append(Qubit_Spin_Array)
    return Spin_List



def dot_product_spin_operators(chain_length, Spin_Operator_List, qubit_position_1, qubit_position_2):
    """
    perform dot product of 2 spin operators (Sx, Sy, Sz) in 2 postions
    """
    dot_product = np.zeros((2**chain_length, 2**chain_length))
    dot_product = dot_product.astype('complex128')
    for Spin_Operator in range(3):
        Spin_Operator_1 = Spin_Operator_List[qubit_position_1][Spin_Operator]
        Spin_Operator_2 = Spin_Operator_List[qubit_position_2][Spin_Operator]
        dot_product += np.matmul(Spin_Operator_1, Spin_Operator_2)
    return dot_product

def dot_product_spin_operators_XY(chain_length, Spin_Operator_List, qubit_position_1, qubit_position_2):
    """
    Perform dot product of 2 spin operators (Sx, Sy) in 2 postions
    """
    dot_product = np.zeros((2**chain_length, 2**chain_length))
    dot_product = dot_product.astype('complex128')
    for Spin_Operator in range(2):
            Spin_Operator_1 = Spin_Operator_List[qubit_position_1][Spin_Operator]
            Spin_Operator_2 = Spin_Operator_List[qubit_position_2][Spin_Operator]
            dot_product += np.matmul(Spin_Operator_1, Spin_Operator_2)
    return dot_product

def Heisenberg_Hamiltonian_Constructor(chain_length, Spin_Operator_List, couplings, XY_HAM = False):
    """
    Create Heisenberg Hamiltonian with specific couplings and chain length. If XY_HAM is True, will use the XY
    Hamiltonian instead of the Heisenberg.
    """
    #Spin_Operator_List = Spin_List_Creator(chain_length)
    Heisenberg = np.zeros((2**chain_length, 2**chain_length))
    Heisenberg = Heisenberg.astype('complex128')
    for qubit_position, J in enumerate(couplings):
        if XY_HAM == False:
            Heisenberg = Heisenberg + (J * dot_product_spin_operators(chain_length, Spin_Operator_List, qubit_position, qubit_position+1))
        else:
            Heisenberg = Heisenberg + (J * dot_product_spin_operators_XY(chain_length, Spin_Operator_List, qubit_position, qubit_position+1))
    return Heisenberg

def time_evolution(hamiltonian_matrix, time):
    """
    Evolve the Hamiltonian for specific time interval
    """
    time_scaled_matrix = -1j * time * hamiltonian_matrix
    time_evol = expm(time_scaled_matrix)
    return time_evol


def Calculate_Fidelity(chain_length, Spin_Operator_List, initial_state, final_state, couplings, time, XY_HAM = False):
    """
    Calculate fidelity for given couplings over a given time. Outputs the probability of achieving 
    that coupling
    """
    Temp_Hamiltonian = Heisenberg_Hamiltonian_Constructor(chain_length, Spin_Operator_List, couplings, XY_HAM)
    time_evolved_matrix = time_evolution(Temp_Hamiltonian, time)
    fidelity = np.matmul(final_state, np.matmul(time_evolved_matrix, initial_state))
    fidelity_value = fidelity.item()
    probability = fidelity_value * np.conj(fidelity_value)
    return probability

def plot_fidelity_overtime(chain_size, couplings, total_time):
    """
    Plots the fidelity over a given time with specified coupling
    """
    initial_state = create_initial_final_states(chain_size, True)
    final_state = create_initial_final_states(chain_size, False)
    final_state = final_state.transpose()
    Spin_Operator_List = Spin_List_Creator(chain_size)
    x = np.arange(0, total_time, 0.1)
    y = np.zeros(x.size)
    for index, value in np.ndenumerate(x):
        y[index] = Calculate_Fidelity(chain_size, Spin_Operator_List, initial_state, final_state, couplings, value, XY_HAM = False)
    plt.ylabel("Fidelity") 
    plt.xlabel("Time · J") 
    plt.title("Changing fidelity of state transfer for different times") 
    plt.plot(x,y) 
    plt.show()
    plt.hold

#plot_fidelity_overtime(4, [0.9813, 1.9624, 0.9813], 10)
plot_fidelity_overtime(4, [1,1,1], 20)

def find_optimal_fidelity(chain_length, couplings, Spin_Operator_List, initial_state, final_state, total_time, XY_HAM = False):
    """
    Finds the MAXIMUM fidelity achieved by the evolution when evolved over total_time
    """
    times = np.arange(0, total_time, 0.1)
    fidelities = np.zeros(times.size)
    for index, value in np.ndenumerate(times):
        fidelities[index] = float(Calculate_Fidelity(chain_length, Spin_Operator_List, initial_state, final_state, couplings, value, XY_HAM))
    return np.amax(fidelities)


def plot_changing_fidelity(chain_size, evolution_time):
    """
    #Try changing J2 to see what happens to the maximum fidelity achieved
    """
    initial_state = create_initial_final_states(chain_size, True)
    final_state = create_initial_final_states(chain_size, False)
    final_state = final_state.transpose()
    Spin_Operator_List = Spin_List_Creator(chain_size)
    j2s = np.arange(0, 10, 0.1)
    fidelities = np.zeros(j2s.size)
    for index, j2_value in np.ndenumerate(j2s):
        coupling_trial = [1, j2_value, 1]
        fidelities[index] = find_optimal_fidelity(chain_size, coupling_trial, Spin_Operator_List, initial_state, final_state, evolution_time, XY_HAM = False)
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
    
#plot_changing_fidelity(4, 10)

def cost_function(couplings, chain_length, Spin_Operator_List, initial_state, final_state, total_time, XY_HAM = False):
    """
    #Calculate the cost function in terms of fidelity for a given coupling over a given time, using the maximum achieved fidelity
    """
    return (1 - find_optimal_fidelity(chain_length, couplings, Spin_Operator_List, initial_state, final_state, total_time, XY_HAM))

def symmetric_chain_cost_function(couplings_values, chain_size, Spin_Operator_List, initial_state, final_state, XY_HAM = False):
    number_couplings = chain_size - 1
    new_couplings = np.zeros(number_couplings)
    if number_couplings % 2 == 0:
        number_variable_couplings = int(number_couplings / 2) #half the size
        indicies = [i for i in range(number_variable_couplings)] + [number_couplings - 1 - i for i in range(number_variable_couplings)]
        np.put(new_couplings, indicies, couplings_values)
    if number_couplings % 2 == 1:
        number_variable_couplings = int((number_couplings + 1) / 2)
        indicies = [i for i in range(number_variable_couplings)] + [number_couplings - 1 - i for i in range(number_variable_couplings)]
        np.put(new_couplings, indicies, couplings_values)
    total_time = (10 * chain_size) / np.average(new_couplings)
    return (1 - find_optimal_fidelity(chain_size, new_couplings, Spin_Operator_List, initial_state, final_state, total_time, XY_HAM))


def return_fidelities(chain_size, couplings_0, consts):
    """
    Final fidelity calculator. Should build the chain, calculate the fidelity over the chain and find the
    optimisation.
    Works only for chains of size > 3
    """
    initial_state = create_initial_final_states(chain_size, True)
    final_state = create_initial_final_states(chain_size, False)
    final_state = final_state.transpose()
    Spin_Operator_List = Spin_List_Creator(chain_size)

    res = minimize_pso(symmetric_chain_cost_function, couplings_0, args = (chain_size, Spin_Operator_List, initial_state, final_state), constraints= consts, options={'stable_iter' : 20, 'max_velocity' : 1, 'verbose' : True})
    print(f"Best final fidelity: {1 - symmetric_chain_cost_function(res.x, chain_size, Spin_Operator_List, initial_state, final_state, XY_HAM = False)}")
    print(f"Final Couplings: {res.x}")
        
    #result = minimize(symmetric_chain_cost_function, couplings_0, args = (chain_size, Spin_Operator_List, initial_state, final_state, evolution_time), method='nelder-mead', options={'xtol': 1e-8, 'disp': True})
    #print(result.x)

#couplings_0 = np.random.uniform(1, 2, (2, 2))
#return_fidelities(4, 1, couplings_0, cons)
"""
def Main():
    parser = argparse.ArgumentParser()
    parser.add_argument("chain_size", help = "The number of quantum states in your chain", type = int)
    parser.add_argument("particles", help = "The number of particles created in swarm", type = int)
    args = parser.parse_args()
    number_couplings = args.chain_size - 1
    
    cons = []
    if number_couplings % 2 == 0:
        number_variable_couplings = int(number_couplings / 2)
    if number_couplings % 2 == 1:
        number_variable_couplings = int((number_couplings + 1) / 2)
    for i in range(number_variable_couplings):
            cons.append({'type' : 'ineq', 'fun' : lambda x: x[i]})
    cons = tuple(cons)
    print(cons)
     
     #cons = ({'type' : 'ineq', 'fun' : lambda x: x[0]},
     #        {'type' : 'ineq', 'fun' : lambda x: x[1]})

    
    initial_couplings = init_feasible(cons, low=1., high=2., shape=(args.particles, number_variable_couplings))
    return return_fidelities(args.chain_size, initial_couplings, cons)

if __name__ == '__main__':
    Main()
"""