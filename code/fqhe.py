import pennylane as qml
from pennylane import numpy as np


def phi(t, n_blocks):
    """
    This function computes the phases for the terms in the v=1/3 FQHE quantum state.

    :param t: Potential strength t=sqrt(V_30/V_10)
    :param n_blocks: Number of 3-qubit blocks in the FQHE system.
    :return: Angled phi_i for the bozonized system, that ensure correct phases.
    """
    phi_i = [np.arctan(-t)]
    for i in range(n_blocks):
        phi_i.insert(0, np.arctan(-t * np.cos(phi_i[0])))
    return phi


def variational_fqhe_circuit(n_blocks, phi_i, t=0.5):
    """

    :param n_blocks: Number of 3-qubit blocks in the FQHE system.
    :param t: Potential strength t=sqrt(V_30/V_10)
    :return: The v=1/3 FQHE state.
    """
    phi_i = phi(n_blocks, t)
    for i in range(n_blocks):
        qml.PauliX(3 * i)  # Stage 0
        qml.CRY(-2 * phi_i[i], wires=[3*i + 1, 3*(i + 1) + 1])  # Stage 1

    # Stage 2 - part 1
    for i in range(n_blocks):
        qml.CNOT(wires=[3 * i + 1, 3 * i + 2])
        qml.RZ(np.pi, wires=3 * i)
        qml.CNOT(wires=[3 * i + 1, 3 * i])

    # Stage 2 - part 2
    for i in range(n_blocks):
        qml.RZ(np.pi, wires=3 * i + 1)
        qml.CNOT(wires=[3 * i + 2, 3 * i + 3])

    # Stage 2 - part 3
    for i in range(n_blocks):
        qml.RZ(np.pi, wires=3 * i + 2)
        qml.CNOT(wires=[3 * i + 1, 3 * i])

    return qml.state()


def fqhe_circuit(n_blocks, t=0.5):
    return variational_fqhe_circuit(n_blocks, phi(t, n_blocks), t)


if __name__ == '__main__':
    print("PyCharm")
