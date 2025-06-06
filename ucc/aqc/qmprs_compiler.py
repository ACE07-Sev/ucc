import numpy as np
from numpy.typing import NDArray
from qmprs.synthesis.mps_encoding import Sequential as QmprsSequential
from quick.circuit import QiskitCircuit
from qiskit import QuantumCircuit
from qiskit.quantum_info import partial_trace, entropy
import warnings


class QmprsCompiler:
    """Wrapper for `qmprs.synthesis.mps_encoding.Sequential` to approximately
    compile a statevector to a circuit using MPS encoding.

    For more information, see:
    https://github.com/Qualition/qmprs
    """
    def __init__(self, target_fidelity=0.99) -> None:
        """Initialize the QmprsCompiler with a target fidelity.

        Args:
            target_fidelity (float): The target fidelity for the MPS encoding.
            Defaults to 0.99.
        """
        self.sequential = QmprsSequential(QiskitCircuit)
        self.sequential.fidelity_threshold = target_fidelity

    @staticmethod
    def calculate_entanglement_entropy_slope(state: NDArray[np.complex128]) ->  float:
        """Calculate the slope of the entanglement entropy as the
        subsystem size increases, which can indicate whether the
        entanglement is volume-law or area-law. This returns a float
        as opposed to a string to provide a more dynamic response.

        Args:
            state (NDArray[np.complex128]): The quantum state represented as a statevector.

        Returns:
            float: The entanglement entropy of the state.
        """
        n = int(np.ceil(np.log2(len(state))))

        entropies = []
        for k in range(1, n//2 + 1):
            # Trace out rest of the qubits
            rho_A = partial_trace(state, list(range(k, n)))
            S = entropy(rho_A, base=2)
            entropies.append(S)

        # Check if the entropies form a straight line or a curve
        entropies = np.array(entropies[len(entropies) // 2:])
        x = np.arange(1, len(entropies) + 1)

        # Linear regression: fit y = ax + b
        x_mean = np.mean(x)
        y_mean = np.mean(entropies)

        numerator = np.sum((x - x_mean) * (entropies - y_mean))
        denominator = np.sum((x - x_mean)**2)

        slope = numerator / denominator if denominator != 0 else 0

        return slope

    @staticmethod
    def optimal_params(statevector: NDArray[np.complex128]) -> tuple[int, int]:
        """Calculate the optimal number of layers and sweeps for the
        MPS encoding.

        Args:
            statevector (NDArray[np.complex128]): The statevector to analyze.

        Returns:
            tuple[int, int]: A tuple containing the number of layers and sweeps.
        """
        num_qubits = int(np.ceil(np.log2(len(statevector))))
        slope = QmprsCompiler.calculate_entanglement_entropy_slope(statevector)

        # Entanglement entropy slope is between 0 and 1
        # Use a smooth transition between area-law (0 to 0.5) and volume-law (1)
        # The higher the slope, the more layers and sweeps are needed
        num_layers = int((2 + 2 * slope) * num_qubits / 1.5)
        num_sweeps = int((10 + 20 * slope) * num_qubits)

        return num_layers, num_sweeps

    def __call__(
            self,
            statevector: NDArray[np.complex128],
            verbose: bool = False
        ) -> QuantumCircuit:
        """Call the instance to create the circuit that encodes the statevector.

        Args:
            statevector (NDArray[np.complex128]): The statevector to convert.
            verbose (bool): If True, print additional information during the process.

        Returns:
            QuantumCircuit: The generated quantum circuit.
        """
        num_qubits = int(np.ceil(np.log2(len(statevector))))
        num_layers, num_sweeps = self.optimal_params(statevector)

        circuit = self.sequential.prepare_state(
            statevector=statevector,
            bond_dimension=2**num_qubits,
            num_layers=num_layers,
            num_sweeps=num_sweeps
        )

        fidelity = np.vdot(circuit.get_statevector(), statevector)

        if verbose:
            slope = self.calculate_entanglement_entropy_slope(statevector)
            if np.isclose(slope, 1, atol=0.1):
                warnings.warn(
                    "Warning: The state is volume-law entangled. Compression may be too lossy."
                )

            print(
                f"Fidelity: {fidelity:.4f}, "
                f"Number of qubits: {num_qubits}, "
                f"Number of layers: {num_layers}, "
                f"Number of sweeps: {num_sweeps}"
            )

        return circuit.circuit