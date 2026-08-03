"""Submodule for defining the basic gate class.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sympy import Matrix


class Gate:

    __slots__ = ['_matrix', '_properties', '_unitary']

    def __init__(self,
        matrix: NDArray,
        qubits: int | tuple = None
    ) -> None:
        """Initialize a gate using its matrix definition.

        Args:
            matrix (NDArray): numpy array defining the unitary matrix.
            qubits (int | tuple): qubit label(s).
        """
        self._matrix = matrix
        self._unitary = matrix
        self._properties = {
            'alias':  None,
            'dim':    self._matrix.shape[0],
            'locally_equivalent': None,
            'matrix': matrix,
            'name':   'Gate',
            'qubits': qubits if type(qubits) is tuple else (qubits,),
            'params': {},
            'subspace': 'GE'

        }

    def __call__(self) -> Matrix:
        """The sympy expression for the numpy array.

        Returns:
            Matrix: sympy matrix.
        """
        return Matrix(self._matrix.round(3))

    def __repr__(self) -> str:
        """Returns information about the gate.

        Returns:
            str: string representation of the gate.
        """
        return (
            f'{self.qubits} ' + f'{self.name} \n' +
            np.array_repr(np.around(self._matrix, 3)
        ))

    def __str__(self) -> str:
        """Returns information about the gate.

        Returns:
            str: string representation of the gate.
        """
        return (
            f'{self.qubits} ' + f'{self.name} \n' +
            np.array_repr(np.around(self._matrix, 3)
        ))

    def __eq__(self, other: object) -> bool:
        """Check equality between two gates.

        Two gates are equal if they have the same name, qubits, subspace, and
        params.

        Args:
            other (object): object to compare against.

        Returns:
            bool: True if the gates are equal, False otherwise.
        """
        if not isinstance(other, Gate):
            return NotImplemented
        return (
            self.name == other.name
            and self.qubits == other.qubits
            and self.subspace == other.subspace
            and self._properties.get('params', {})
            == other._properties.get('params', {})
        )

    def __hash__(self) -> int:
        """Hash the gate by its name, qubits, and subspace.

        Returns:
            int: hash of the gate.
        """
        return hash((self.name, self.qubits, self.subspace))

    def _repr_html_(self):
        """Returns information about the gate.

        Returns:
            html: html representation of the gate.
        """
        df = pd.DataFrame([
            [(pd.DataFrame(self._matrix)
                .style
                .format(precision=3)
                .hide(axis='index')
                .hide(axis='columns')
                .set_table_attributes('class="matrix"')
                .to_html()
            )]
        ], dtype="object")
        df.index = [self.qubits]
        df.columns = ['Matrix']
        df.insert(0, 'Gate', [self.name])

        df_styler = df.style.set_table_styles([
            {"selector": ".matrix", "props": "position: relative;"},
            {"selector": ".matrix:before, .matrix:after",
             "props":  'content: ""; position: absolute; top: 0; border: 1px \
                solid #000; width: 6px; height: 100%;'
            },
            {"selector": ".matrix:before",
             "props": "left: -0px; border-right: -0;"},
            {"selector": ".matrix:after",
              "props": "right: -0px; border-left: 0;"}
        ])

        return df_styler.to_html()

    @property
    def alias(self) -> str:
        """The alias(es) of the gate.

        Returns:
            str: alias of the gate.
        """
        return self._properties['alias']

    @property
    def dim(self) -> int:
        """The Hilbert space dimension of the unitary operator.

        Returns:
            int: Hilbert space dimension.
        """
        return self._properties['dim']

    @property
    def is_single_qubit(self) -> bool:
        """Whether or not the gate acts on a single qubit.

        Returns:
            bool: single-qubit gate or not.
        """
        return len(self.qubits) == 1 and self.dim == 2

    @property
    def is_single_qudit(self) -> bool:
        """Whether or not the gate acts on a single qudit.

        Returns:
            bool: single-qudit gate or not.
        """
        return len(self.qudits) == 1

    @property
    def is_multi_qubit(self) -> bool:
        """Whether or not the gate acts on multiple qubits.

        Returns:
            bool: multi-qubit gate or not.
        """
        return len(self.qubits) > 1 and self.dim == 2 ** len(self.qubits)

    @property
    def is_multi_qudit(self) -> bool:
        """Whether or not the gate acts on multiple qudits.

        Returns:
            bool: multi-qudit gate or not.
        """
        return len(self.qudits) > 1

    @property
    def is_measurement(self) -> bool:
        """Whether or not gate is a measurement operation.

        Returns:
            bool: measurement or not.
        """
        return False

    @property
    def locally_equivalent(self) -> str:
        """The names of the locally-equivalent gates.

        Returns:
            str: names of the locally-equivalent gates.
        """
        return self._properties['locally_equivalent']

    @property
    def matrix(self) -> NDArray:
        """The numpy array of the matrix.

        Returns:
            NDArray: numpy array of the matrix.
        """
        return self._matrix

    @property
    def unitary(self) -> NDArray:
        """The unitary matrix of the gate in the full qudit space.

        For 2×2 gates with ``subspace='EF'``, the matrix is embedded
        into the |1⟩–|2⟩ block of a 3×3 qutrit space. All other
        gates are returned as-is.

        Returns:
            NDArray: numpy array of the unitary matrix, or None for
                non-unitary operations (measurements, reset).
        """
        if (self._properties['subspace'] == 'EF'
                and self._unitary.shape[0] == 2):
            U = np.eye(3, dtype=complex)
            U[1:3, 1:3] = self._unitary
            return U
        return self._unitary

    @unitary.setter
    def unitary(self, matrix: NDArray) -> None:
        """Set a measured (potentially imperfect) unitary matrix.

        Args:
            matrix (NDArray): numpy array of the measured unitary.
        """
        self._unitary = matrix

    @property
    def name(self) -> str:
        """The name of the gate.

        Returns:
            str: name of the gate.
        """
        return self._properties['name']

    @property
    def properties(self) -> dict:
        """Properties of the gate.

        Returns:
            Dict: gate properties.
        """
        return self._properties

    @property
    def subspace(self) -> str:
        """Subspace (GE or EF) within which the gate acts.

        Returns:
            str: GE or EF.
        """
        return self._properties['subspace']

    @property
    def qubits(self) -> tuple:
        """The qubit(s) that the gate acts on.

        Returns:
            tuple: qubit label(s).
        """
        return self._properties['qubits']

    @property
    def qudits(self) -> tuple:
        """The qudit(s) that the gate acts on.

        Returns:
            tuple: qudit label(s).
        """
        return self.qubits
