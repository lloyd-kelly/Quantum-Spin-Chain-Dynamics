from Fidelity import *
import numpy as np



def plot_changing_fidelity(evolution_time):
    """
    #Try changing J2 to see what happens to the maximum fidelity achieved
    """
    j2s = np.arange(0, 2, 0.01)
    fidelities = np.zeros(j2s.size)
    for index, j2_value in np.ndenumerate(j2s):
        coupling_trial = [1, j2_value, 1]
        fidelities[index] = find_optimal_fidelity(coupling_trial, evolution_time, XY_HAM = False)
    plt.ylabel("Fidelity") 
    plt.xlabel("j2")
    plt.title("Best fidelity found for different j2s") 
    plt.plot(j2s,fidelities)
    plt.axvline(x=math.sqrt(2), color='r', linestyle='-') 
    plt.show()
    
plot_changing_fidelity(10)

def cost_function(couplings, total_time, XY_HAM = False):
    """
    #Calculate the cost function in terms of fidelity for a given coupling over a given time, using the maximum achieved fidelity
    """
    return (1 - find_optimal_fidelity(couplings, total_time, XY_HAM))
