import pytest
from initialise import *
from Hamiltonian import *
import pytest

def test_Hamiltonian_length_chain_1():
    chain_length = 1
    Spin_Operator_List = Spin_List_Creator(chain_length)
    dot = dot_product_spin_operators(chain_length, Spin_Operator_List, 0, 0)
    dot_actual = np.matmul(Sx, Sx) + np.matmul(Sy, Sy) + np.matmul(Sz, Sz)

    assert dot.size == 4
    assert np.array_equal(dot, dot_actual) == True

def test_Hamiltonian_length_chain_2():
    chain_length = 2
    Spin_Operator_List = Spin_List_Creator(chain_length)
    dot = dot_product_spin_operators(chain_length, Spin_Operator_List, 0, 1)
    dot_actual = np.matmul(Spin_Operator_List[0][0], Spin_Operator_List[1][0]) + np.matmul(Spin_Operator_List[0][1], Spin_Operator_List[1][1]) + np.matmul(Spin_Operator_List[0][2], Spin_Operator_List[1][2])

    assert np.array_equal(dot, dot_actual) == True