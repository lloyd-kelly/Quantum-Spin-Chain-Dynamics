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

def create_basis_state(chain_size, site):
    vector = np.zeros((chain_size, 1))
    vector[site - 1] = 1.
    return vector

def Hamiltonian_Constructor(chain_size, couplings):
    couplings_sum = 0.
    N = chain_size
    for coupling in couplings:
        couplings_sum += coupling
    H = np.zeros((N, N))
    H[0,0] = couplings_sum - 2 * couplings[0]
    H[N-1,N-1] = couplings_sum - 2 * couplings[N-2]
    for i in range(1, N-1):
        H[i,i] = couplings_sum - 2 * couplings[i-1] - 2 * couplings[i]
    for i in range(N-1):
        H[i,i+1] = 2 * couplings[i]
        H[i+1,i] = 2 * couplings[i]
    H = np.divide(H, np.average(couplings))
    return H

def time_evolution2(hamiltonian_matrix, time):
    """
    Evolve the Hamiltonian for specific time interval
    """
    time_scaled_matrix = -1j * time * hamiltonian_matrix
    time_evol = expm(time_scaled_matrix)
    return time_evol

def Calculate_Fidelity2(chain_size, initial_state, final_state, couplings, time):
    """
    Calculate fidelity for given couplings over a given time. Outputs the probability of achieving 
    that coupling
    """
    Temp_Hamiltonian = Hamiltonian_Constructor(chain_size, couplings)
    time_evolved_matrix = time_evolution2(Temp_Hamiltonian, time)
    fidelity = np.matmul(final_state, np.matmul(time_evolved_matrix, initial_state))
    fidelity_value = fidelity.item()
    probability = fidelity_value * np.conj(fidelity_value)
    return probability

def plot_fidelity_overtime2(chain_size, initial_state, final_state, couplings, total_time, colour, line_title):
    """
    Plots the fidelity over a given time with specified coupling
    """
    x = np.arange(0, total_time, 0.1)
    y = np.zeros(x.size)
    for index, value in np.ndenumerate(x):
        y[index] = Calculate_Fidelity2(chain_length, initial, final, couplings, value)
    plt.ylabel("Fidelity") 
    plt.xlabel("Time • J") 
    plt.plot(x,y, colour, label = line_title)

def find_optimal_fidelity2(chain_size, initial_state, final_state, couplings, total_time):
    """
    Finds the MAXIMUM fidelity achieved by the evolution when evolved over total_time
    """
    times = np.arange(0, total_time, 0.1)
    fidelities = np.zeros(times.size)
    for index, value in np.ndenumerate(times):
        fidelities[index] = float(Calculate_Fidelity2(chain_length, initial, final, couplings, value))
    return np.amax(fidelities)

def turn_symmetric_couplings(couplings_values, chain_size, initial_state, final_state):
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
    return new_couplings

def symmetric_chain_cost_function2(couplings_values, chain_size, initial_state, final_state):
    new_couplings = turn_symmetric_couplings(couplings_values, chain_size, initial_state, final_state)
    total_time = (10 * chain_size) / np.average(new_couplings)
    return (1 - find_optimal_fidelity2(chain_size, initial_state, final_state, new_couplings, total_time))

def return_fidelities2(chain_size, initial_state, final_state, couplings_initial, consts):
    """
    # Final fidelity calculator. Should build the chain, calculate the fidelity over the chain and find the
    # optimisation.
    # Works only for chains of size > 3
    """
    res = minimize_pso(symmetric_chain_cost_function2, couplings_initial, args = (chain_size, initial_state, final_state), constraints= None, options={'stable_iter' : 5, 'max_velocity' : 1, 'verbose' : True})
    print(f"Best final fidelity: {1 - symmetric_chain_cost_function2(res.x, chain_size, initial_state, final_state)}")
    print(f"Final Couplings: {res.x}")
    return res
    #result = minimize(symmetric_chain_cost_function, couplings_0, args = (chain_size, Spin_Operator_List, initial_state, final_state, evolution_time), method='nelder-mead', options={'xtol': 1e-8, 'disp': True})
    #print(result.x)


def plot_couplings(chain_size, couplings):
    x = range(1, chain_size)
    y = couplings
    chain_size_str = str(chain_size)
    plt.ylabel("Coupling value") 
    plt.xlabel("Site Position") 
    plt.title("Optimised couplings between sites for an " + chain_size_str + "-site chain") 
    plt.plot(x,y)
    plt.savefig('Couplings for ' + chain_size_str + '-site chain.png')
    plt.close()

chain_length = 4
chain_lenth_str = str(chain_length)
number_couplings = chain_length - 1
initial = create_basis_state(chain_length, 1)
final = create_basis_state(chain_length, chain_length).transpose()
basic_couplings = [1] * (chain_length - 1)
if number_couplings % 2 == 0:
    number_variable_couplings = int(number_couplings / 2)
if number_couplings % 2 == 1:
    number_variable_couplings = int((number_couplings + 1) / 2)
cons = []
for i in range(number_variable_couplings):
    cons.append({'type' : 'ineq', 'fun' : lambda x: x[i]})
cons = tuple(cons)
initial_couplings = init_feasible(cons, low=1., high=5., shape=(100, number_variable_couplings))
result = return_fidelities2(chain_length, initial, final, initial_couplings, cons)
final_couplings = turn_symmetric_couplings(result.x, chain_length, initial, final)
normaliser = final_couplings[0]
final_couplings = np.divide(final_couplings, final_couplings[0])
print(final_couplings)
plot_fidelity_overtime2(chain_length, initial, final, basic_couplings, (10 * chain_length) / np.average(initial_couplings), 'b', 'Unoptimised Couplings')
plot_fidelity_overtime2(chain_length, initial, final, final_couplings, (10 * chain_length) / np.average(final_couplings), 'r', 'Optimised Couplings')
plt.legend(loc='best')
plt.title('Fidelity over time for ' + chain_lenth_str + '-site spin chain')
plt.savefig('Fidelity over time for ' + chain_lenth_str + '-site spin chain.png')
plt.close()
plot_couplings(chain_length, final_couplings)

# """

# def plot_changing_fidelity(chain_size, evolution_time):
#     """
#     #Try changing J2 to see what happens to the maximum fidelity achieved
#     """
#     initial_state = create_initial_final_states(chain_size, True)
#     final_state = create_initial_final_states(chain_size, False)
#     final_state = final_state.transpose()
#     Spin_Operator_List = Spin_List_Creator(chain_size)
#     j2s = np.arange(0, 10, 0.1)
#     fidelities = np.zeros(j2s.size)
#     for index, j2_value in np.ndenumerate(j2s):
#         coupling_trial = [1, j2_value, 1]
#         fidelities[index] = find_optimal_fidelity(chain_size, coupling_trial, Spin_Operator_List, initial_state, final_state, evolution_time, XY_HAM = False)
#     best_fid = np.amax(fidelities)
#     first_j2_optimal = j2s[np.where(fidelities == best_fid)[0][0]]
#     print(best_fid)
#     print(first_j2_optimal)
#     print(initial_state)
#     print(final_state)
#     plt.ylabel("Fidelity") 
#     plt.xlabel("j2")
#     plt.title("Best fidelity found for different j2s") 
#     plt.plot(j2s,fidelities)
#     plt.axvline(x=math.sqrt(2), color='r', linestyle='-') 
#     plt.show()
    
# #plot_changing_fidelity(4, 10)

# def cost_function(couplings, chain_length, Spin_Operator_List, initial_state, final_state, total_time, XY_HAM = False):
#     """
#     #Calculate the cost function in terms of fidelity for a given coupling over a given time, using the maximum achieved fidelity
#     """
#     return (1 - find_optimal_fidelity(chain_length, couplings, Spin_Operator_List, initial_state, final_state, total_time, XY_HAM))

# def symmetric_chain_cost_function(couplings_values, chain_size, Spin_Operator_List, initial_state, final_state, XY_HAM = False):
#     number_couplings = chain_size - 1
#     new_couplings = np.zeros(number_couplings)
#     if number_couplings % 2 == 0:
#         number_variable_couplings = int(number_couplings / 2) #half the size
#         indicies = [i for i in range(number_variable_couplings)] + [number_couplings - 1 - i for i in range(number_variable_couplings)]
#         np.put(new_couplings, indicies, couplings_values)
#     if number_couplings % 2 == 1:
#         number_variable_couplings = int((number_couplings + 1) / 2)
#         indicies = [i for i in range(number_variable_couplings)] + [number_couplings - 1 - i for i in range(number_variable_couplings)]
#         np.put(new_couplings, indicies, couplings_values)
#     total_time = (10 * chain_size) / np.average(new_couplings)
#     return (1 - find_optimal_fidelity(chain_size, new_couplings, Spin_Operator_List, initial_state, final_state, total_time, XY_HAM))


# def return_fidelities(chain_size, couplings_0, consts):
#     """
#     # Final fidelity calculator. Should build the chain, calculate the fidelity over the chain and find the
#     # optimisation.
#     # Works only for chains of size > 3
#     """
#     initial_state = create_initial_final_states(chain_size, True)
#     final_state = create_initial_final_states(chain_size, False)
#     final_state = final_state.transpose()
#     Spin_Operator_List = Spin_List_Creator(chain_size)

#     res = minimize_pso(symmetric_chain_cost_function, couplings_0, args = (chain_size, Spin_Operator_List, initial_state, final_state), constraints= consts, options={'stable_iter' : 20, 'max_velocity' : 1, 'verbose' : True})
#     print(f"Best final fidelity: {1 - symmetric_chain_cost_function(res.x, chain_size, Spin_Operator_List, initial_state, final_state, XY_HAM = False)}")
#     print(f"Final Couplings: {res.x}")
        
#     #result = minimize(symmetric_chain_cost_function, couplings_0, args = (chain_size, Spin_Operator_List, initial_state, final_state, evolution_time), method='nelder-mead', options={'xtol': 1e-8, 'disp': True})
#     #print(result.x)

# #couplings_0 = np.random.uniform(1, 2, (2, 2))
# #return_fidelities(4, 1, couplings_0, cons)
# """
# # def Main():
# #     parser = argparse.ArgumentParser()
# #     parser.add_argument("chain_size", help = "The number of quantum states in your chain", type = int)
# #     parser.add_argument("particles", help = "The number of particles created in swarm", type = int)
# #     args = parser.parse_args()
# #     number_couplings = args.chain_size - 1
    
# #     cons = []
# #     if number_couplings % 2 == 0:
# #         number_variable_couplings = int(number_couplings / 2)
# #     if number_couplings % 2 == 1:
# #         number_variable_couplings = int((number_couplings + 1) / 2)
# #     for i in range(number_variable_couplings):
# #             cons.append({'type' : 'ineq', 'fun' : lambda x: x[i]})
# #     cons = tuple(cons)
# #     print(cons)
     
#      #cons = ({'type' : 'ineq', 'fun' : lambda x: x[0]},
#      #        {'type' : 'ineq', 'fun' : lambda x: x[1]})
# """
    
#     initial_couplings = init_feasible(cons, low=1., high=2., shape=(args.particles, number_variable_couplings))
#     return return_fidelities(args.chain_size, initial_couplings, cons)

# if __name__ == '__main__':
#     Main()
# """