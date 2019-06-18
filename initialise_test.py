import pytest
from initialise import *

def test_length_1_chain():
    chain_length = 1
    initial_state = create_initial_final_states(chain_length, True)
    final_state = create_initial_final_states(chain_length, False)
    Spin_Operator_List = Spin_List_Creator(chain_length)
    print(Spin_Operator_List)

    assert initial_state.all() == final_state.all()
    assert Spin_Operator_List[0][0].all() == Sx.all()
    assert Spin_Operator_List[0][1].all() == Sy.all()
    assert Spin_Operator_List[0][2].all() == Sz.all()
    assert type(Spin_Operator_List) == list
    assert len(Spin_Operator_List) == chain_length

def test_length_2_chain():
    chain_length = 2
    initial_state = create_initial_final_states(chain_length, True)
    final_state = create_initial_final_states(chain_length, False)
    Spin_Operator_List = Spin_List_Creator(chain_length)
    Sx0_actual = np.kron(Sx, qubit_identity) #Test using kron function
    Sz1_actual = np.array([1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1, 0, 0, 0, 0, -1]).reshape(4, 4) #Test building the matrix manually

    assert initial_state[1] == 1
    assert initial_state[2] == 0
    assert final_state[1] == 0
    assert final_state[2] == 1
    assert len(initial_state) == len(final_state)
    assert type(Spin_Operator_List) == list
    assert Spin_Operator_List[0][0].all() == Sx0_actual.all()
    assert Spin_Operator_List[1][2].all() == Sz1_actual.all()
    assert len(Spin_Operator_List) == chain_length

def test_length_3_chain():
    chain_length = 3
    initial_state = create_initial_final_states(chain_length, True)
    final_state = create_initial_final_states(chain_length, False)
    Spin_Operator_List = Spin_List_Creator(chain_length)
    Sy1_actual = np.kron(Sy, np.kron(qubit_identity, qubit_identity)) #Test using kron function
    Sz2_actual = np.kron(qubit_identity, np.kron(Sz, qubit_identity)) #Test using kron function
    Sx3_actual = np.kron(qubit_identity, np.kron(qubit_identity, Sy)) #Test using kron function

    assert len(initial_state) == len(final_state)
    assert len(initial_state) == 8
    assert type(Spin_Operator_List) == list
    #assert type(Spin_Operator_List[0]) == <class 'numpy.ndarray'>
    assert Spin_Operator_List[0][1].all() == Sy1_actual.all()
    assert Spin_Operator_List[1][2].all() == Sz2_actual.all()
    assert Spin_Operator_List[2][0].all() == Sx3_actual.all()
    assert len(Spin_Operator_List) == chain_length
    assert Spin_Operator_List[0][1].size == Sy1_actual.size

"""
print(initial_state)
print(final_state)

#print(up)
print(Spin_Operator_List)
"""