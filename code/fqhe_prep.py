import pennylane as qml
from pennylane import numpy as np
import argparse
from time import strftime


def phi(t: float, n_blocks: int) -> list[float]:
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
    for i in range(n_blocks - 1):
        phi_i.insert(0, np.arctan(-t * np.cos(phi_i[0])))

    return phi_i


def fqhe_circuit(n_blocks, obs, phi_i: list[float]) -> list[float]:
    """

    Parameters
    ----------
    n_blocks: Number of 3-qubit blocks in the FQHE system.
    phi_i: Angles phi_i

    Returns
    -------
    A measurement of v=1/3 FQHE state according to the observables specified by obs

    """
    if phi_i is None:
        raise ValueError("Must provide phi")

    # Stage 0
    for i in range(n_blocks + 1):
        qml.PauliX(wires=(3 * i))

    qml.Barrier(wires=range(3 * (n_blocks + 1)))

    # Stage 1
    qml.RY(-2 * phi_i[0], wires=[1])
    for i in range(1, n_blocks):
        qml.CRY(-2 * phi_i[i], wires=[3 * (i - 1) + 1, 3 * i + 1])

    qml.Barrier(wires=range(3 * (n_blocks + 1)))

    # Stage 2 - part 1
    for i in range(n_blocks):
        qml.CNOT(wires=[3 * i + 1, 3 * i + 2])

    # Stage 2 - part 2
    for i in range(n_blocks):
        qml.RZ(np.pi, wires=(3 * i + 1))
        qml.CNOT(wires=[3 * i + 2, 3 * i + 3])

    # Stage 2 - part 3
    for i in range(n_blocks):
        qml.CNOT(wires=[3 * i + 1, 3 * i])

    qml.Barrier(wires=range(3 * (n_blocks + 1)))

    return obs()
    # return qml.expval(qml.PauliZ(0))


def local_device(n_wires, n_shots, name=""):
    dev = qml.device("default.qubit", wires=n_wires, shots=n_shots)
    return dev


def braket_device(n_wires, n_shots, name):
    my_bucket = "amazon-braket-amazon-braket-47ba137bf31d"
    my_prefix = "fqhe" + strftime("%d%m_%H_%M")
    s3_folder = (my_bucket, my_prefix)

    dev_arn = {"sv1": "arn:aws:braket:::device/quantum-simulator/amazon/sv1",
               'aspen11': "arn:aws:braket:::device/qpu/rigetti/Aspen-11",
               "aspenm1": "arn:aws:braket:us-west-1::device/qpu/rigetti/Aspen-M-1"}

    if name in dev_arn:
        dev_remote = qml.device('braket.aws.qubit', device_arn=dev_arn[name], wires=n_wires,
                                s3_destination_folder=s3_folder, shots=n_shots)
    else:
        dev_remote = qml.device('braket.local.qubit', wires=n_wires, shots=n_shots)

    print(dev_remote._device.name)
    return dev_remote


def get_circuit():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default="local")
    parser.add_argument('--name', type=str, default="local")
    parser.add_argument('--blocks', type=int, default=7)
    parser.add_argument('--shots', type=int, default=1000)
    args = parser.parse_args()
    dev_dict = {"local": local_device, "aws": braket_device}

    n_blocks = args.blocks
    n_shots = args.shots
    dev_type = args.device
    dev_name = args.name if dev_type != "local" else ""
    dev = dev_dict[dev_type](3 * (n_blocks + 1), n_shots, dev_name)

    print("Running FQHE v=1/3 simulation with device-{} on {} blocks for {} shots".format(dev_type + "-" + dev_name,
                                                                                          n_blocks, n_shots))
    print(dev.name)
    return dev, n_blocks, n_shots
