# QHack 2022 Open Hackathon
#### Team name: MarpeQ
#### Team members: Arzi Ofir & Lomazov Daniel

# Inside a Quantum Tardis
#### Project Description:
Our project aims to implement the recent suggestion by Rahmani et al. to simulate the Laughlin wavefunction of the ⅓ FQH state. This suggestion takes advantage of a clever reduction of the 2D state to a 1D system. We plan to implement and execute a circuit, verify it yields the required quantum state, and measure interesting physical properties such as the mutual statistics of quasi-particles. Since current NISQ devices are small and scarcely available, we also intend to implement a pennylane version of the parallelization system proposed by Barrat. The system allows simulations of large systems using only a handful of qubits. Since the FQH state is topologically ordered, we cannot rely on measurements of a local order parameter to verify the circuit generates the intended state. Thus it necessitates the measurement of non-local string operators. This means we must extend Barrat’s algorithm to facilitate measurements of non-local operators. We believe this is possible because the FQH ground state assumes the form of a matrix product state.

One challenging issue we need to face in order to complete our plan is the mitigation of noise. Beyond the inherent noise that inflicts each qubit during each gate application Rahmani et al.’s suggestion assumes a snake-like connectivity topology of the qubits in order to use nearest-neighbor two-qubits. Since the device does not provide such connectivity, the noise is amplified by the requirement to execute additional SWAP gates. These allow the device to provide the connectivity between far away qubits. To overcome this issue, we will use simulator measurements of Rahmani et al.’s idealized circuit as training data for a variational circuit. The parameters of the variational circuit will be optimized to yield the FQH state.

#### Presentation:
1) Creating and Manipulating a Laughlin-Type ν = 1/3 Fractional Quantum Hall State on a Quantum Computer with Linear Depth Circuit: 
    ###### https://doi.org/10.1103/PRXQuantum.1.020309
2) Parallel quantum simulation of large systems on small NISQ computers: 
    ###### https://doi.org/10.1038/s41534-021-00420-3

#### Source code: https://github.com/ShapeshiftingMasterOfDarkness/QHack2022-FQHE

#### "Which challenges/prizes would you like to submit your project for?"
* Amazon Braket Challenge
* IBM Qiskit Challenge
* Analog Quantum Computing Challenge
* Google Quantum AI Research Challenge
* Hybrid Algorithms Challenge
* QAOA Challenge
* Quantum Chemistry Challenge
* Science Challenge
* Simulation Challenge
