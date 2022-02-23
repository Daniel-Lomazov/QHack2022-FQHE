from typing import Optional

import pennylane as qml
from pennylane import numpy as np


def phi(n_blocks: int, t: float) -> list[float]:
    """
    This function computes the phases for the terms in the v=1/3 FQHE quantum state.

    Parameters
    ----------
    t: Potential strength t=sqrt(V_30/V_10)
    n_blocks: Number of 3-qubit blocks in the FQHE system

    Returns
    -------
    Angles phi_i for the bozonized system, that ensure correct phases
    """
    phi_i = [np.arctan(-t)]
    for i in range(n_blocks):
        phi_i.insert(0, np.arctan(-t * np.cos(phi_i[0])))
    return phi_i


def variational_fqhe_circuit(n_blocks, t: float = 0.5, phi_i: Optional[list[float]] = None) -> qml.state():
    """

    Parameters
    ----------
    n_blocks: Number of 3-qubit blocks in the FQHE system.
    t: Potential strength t=sqrt(V_30/V_10)
    phi_i: Angles phi_i

    Returns
    -------
    The v=1/3 FQHE state
    """
    if phi_i is None:
        phi_i = phi(n_blocks=n_blocks, t=t)

    for i in range(n_blocks):
        qml.PauliX(wires=(3 * i))  # Stage 0
        qml.CRY(-2 * phi_i[i], wires=[3 * i + 1, 3 * (i + 1) + 1])  # Stage 1

    # Stage 2 - part 1
    for i in range(n_blocks):
        qml.CNOT(wires=[3 * i + 1, 3 * i + 2])
        qml.RZ(np.pi, wires=(3 * i))
        qml.CNOT(wires=[3 * i + 1, 3 * i])

    # Stage 2 - part 2
    for i in range(n_blocks):
        qml.RZ(np.pi, wires=(3 * i + 1))
        qml.CNOT(wires=[3 * i + 2, 3 * i + 3])

    # Stage 2 - part 3
    for i in range(n_blocks):
        qml.RZ(np.pi, wires=(3 * i + 2))
        qml.CNOT(wires=[3 * i + 1, 3 * i])

    return qml.state()


def fqhe_circuit(n_blocks, t=0.5):
    return variational_fqhe_circuit(n_blocks=n_blocks, t=t, phi_i=phi(n_blocks=n_blocks, t=t))


if __name__ == '__main__':
    print("PyCharm")
