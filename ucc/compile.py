import numpy as np
from qbraid.programs.alias_manager import get_program_type_alias
from qbraid.transpiler import ConversionGraph
from qbraid.transpiler import transpile as translate
from .aqc import MPS_Encoder
from .transpilers.ucc_defaults import UCCDefault1
from qiskit import transpile as qiskit_transpile
from qiskit.quantum_info import Statevector

import sys
import warnings
import psutil
import importlib

# Specify the supported Python version range
REQUIRED_MAJOR = 3
MINOR_VERSION_MIN = 12
MINOR_VERSION_MAX = 13

current_major = sys.version_info.major
current_minor = sys.version_info.minor

if current_major != REQUIRED_MAJOR or not (
    MINOR_VERSION_MIN <= current_minor <= MINOR_VERSION_MAX
):
    warnings.warn(
        f"Warning: This package is designed for Python {REQUIRED_MAJOR}.{MINOR_VERSION_MIN}-{REQUIRED_MAJOR}.{MINOR_VERSION_MAX}. "
        f"You are using Python) {current_major}.{current_minor}."
    )
supported_circuit_formats = ConversionGraph().nodes()

# Check if `qmprs` is installed for AQC compilation
qmprs_available = importlib.util.find_spec("qmprs") is not None


def has_enough_memory(num_qubits: int) -> tuple[bool, float, float]:
    """Check if the available user RAM is enough to represent
    the statevector IR.

    Args:
        num_qubits (int): The number of qubits for the statevector.

    Returns:
        has_memory (bool): Whether the user has enough RAM.
        memory_required_gb (float): Amount of memory required to
            store the statevector IR in GB.
        available_memory_gb (float): Amount of free memory that
            can be dedicated to storing the statevector IR in GB.
    """
    available_memory_gb = psutil.virtual_memory().available
    available_memory_gb = available_memory_gb / 2**30

    # Calculate approximately how much memory the statevector
    # requires at worst-case (volume-law)
    # statevectors use np.complex128 which needs 16 bytes
    # Use half of the memory available to store the IR
    memory_required_gb = 2**(4 + num_qubits - 31)
    has_memory = memory_required_gb <= available_memory_gb

    return has_memory, memory_required_gb, available_memory_gb


def approx_compile(circuit):
    """Compiles the qiskit circuit provided to approximately compile,
    and if the circuit state requires more memory than is available or
    the fidelity of aqc is lower than required it will return the original
    circuit.

    Args:
        qiskit_circuit (QuantumCircuit): The circuit to approximately compile.

    Returns:
        QuantumCircuit: The compiled or unchanged circuit.
    """
    if not qmprs_available:
        warnings.warn(
            "Warning: AQC compilation is requested, but `qmprs` is not installed. "
            "Falling back to vanilla sequential encoding."
        )

    if circuit.num_qubits == 1:
        return circuit

    # If the circuit's statevector IR requires more RAM than the user has,
    # ignore the compilation and return the inputted circuit as is
    has_memory, required_memory, available_memory = has_enough_memory(
        circuit.num_qubits
    )
    if not has_memory:
        warnings.warn(
            "Required memory to store statevector IR is more than available memory. \n"
            f"Required_memory: {required_memory} GB \n"
            f"Available memory to allocate to storing statevector IR: {available_memory} GB \n\n"
        )
        return circuit

    target_sv = Statevector(circuit).data
    aqc_circuit = MPS_Encoder()(target_sv)

    # Fallback protocol for low fidelity, which discards the compiled
    # circuit and returns the original one
    # TODO: This should be modified depending on maintainer notes
    fidelity = np.vdot(target_sv, Statevector(aqc_circuit).data)
    if fidelity < 0.8:
        warnings.warn(
            f"Warning: Fidelity {fidelity:.4f} is too low. Discarding compression."
        )
        return circuit

    # If the compiled circuit is deeper and has more cx than permitted, discard
    # the compilation
    aqc_transpiled = qiskit_transpile(aqc_circuit, basis_gates=["u3", "cx"], optimization_level=3)
    original_transpiled = qiskit_transpile(circuit, basis_gates=["u3", "cx"], optimization_level=3)

    aqc_cx_count = aqc_transpiled.count_ops().get("cx", 0)
    aqc_depth = aqc_transpiled.depth()

    original_cx_count = original_transpiled.count_ops().get("cx", 0)
    original_depth = original_transpiled.depth()

    # Fallback protocol for worse depth and cx counts, which discards
    # the compiled circuit and returns the original one
    # TODO: This should be modified depending on maintainer notes
    if (aqc_cx_count >= original_cx_count) and (aqc_depth >= original_depth):
        return circuit

    return aqc_circuit

def compile(
    circuit,
    return_format="original",
    target_gateset=None,
    target_device=None,
    custom_passes=None,
    aqc=False,
):
    """Compiles the provided quantum `circuit` by translating it to a Qiskit
    circuit, transpiling it, and returning the optimized circuit in the
    specified `return_format`.

    Args:
        circuit (object): The quantum circuit to be compiled.
        return_format (str): The format in which your circuit will be returned.
            e.g., "TKET", "OpenQASM2". Check ``ucc.supported_circuit_formats``.
            Defaults to the format of the input circuit.
        target_gateset (set[str]): (optional) The gateset to compile the circuit to.
            e.g. {"cx", "rx",...}. Defaults to the gate set of the target device if
            available. If no `target_gateset` or ` target_device` is provided, the
            basis gates of the input circuit are not changed.
        target_device (qiskit.transpiler.Target): (optional) The target device to compile the circuit for. None if no device to target
        custom_passes (list[qiskit.transpiler.TransformationPass]): (optional) A list of custom passes to apply after the default set
            of passes. Defaults to None.
        aqc (bool): (optional) If True, uses the AQC compilation strategy. Defaults to False.

    Returns:
        object: The compiled circuit in the specified format.
    """
    if return_format == "original":
        return_format = get_program_type_alias(circuit)

    # Translate to Qiskit Circuit object
    qiskit_circuit = translate(circuit, "qiskit")

    # Utilize approximate quantum compilation via MPS encoding
    # This is useful for area-law entangled circuits which can
    # be significantly compressed without losing much fidelity
    if aqc:
        qiskit_circuit = approx_compile(qiskit_circuit)

    ucc_default1 = UCCDefault1(target_device=target_device)
    if custom_passes is not None:
        ucc_default1.pass_manager.append(custom_passes)
    compiled_circuit = ucc_default1.run(qiskit_circuit)

    if target_gateset is not None:
        # Translate into user-defined gateset; no optimization
        compiled_circuit = qiskit_transpile(
            compiled_circuit, basis_gates=target_gateset, optimization_level=0
        )
    elif hasattr(target_device, "operation_names"):
        if target_gateset not in target_device.operation_names:
            warnings.warn(
                f"Warning: The target gateset {target_gateset} is not supported by the target device. "
            )
        # Use target_device gateset if available
        target_gateset = target_device.operation_names

        # Translate into the target device gateset; no optimization
        compiled_circuit = qiskit_transpile(
            compiled_circuit,
            basis_gates=target_gateset,
            optimization_level=0,
        )

    # Translate the compiled circuit to the desired format
    final_result = translate(compiled_circuit, return_format)
    return final_result
