import numpy as np

#pauli matrices
Sx = np.array([[0, 1], [1, 0]])
Sy = np.array([[0, -1j], [1j, 0]])
Sz = np.array([[1, 0], [0, -1]])

#up and down states
up = np.array([[1], 
               [0]])
down = np.array([[0],
                 [1]])

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
    return tensor_product_initialiser(chain_length, up, down, initial)

qubit_identity = np.identity(2)

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

"""
Qubit_1_Spin = np.array([chain_operator_constructor(1, Sx), chain_operator_constructor(1, Sy),
                         chain_operator_constructor(1, Sz)])
"""