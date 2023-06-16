"""Submodule for defining the basic gate class.
"""
import numpy as np
import pandas as pd

from numpy.typing import NDArray
from sympy import Matrix
from typing import Dict, Tuple, Union


class Gate:

    __slots__ = ['_matrix', '_properties']

    def __init__(self, 
            matrix: NDArray, 
            qubits: Union[int, Tuple] = None
        ) -> None:
        """Initialize a gate using its matrix definition.

        Args:
            matrix (NDArray): numpy array defining the unitary matrix.
            qubits (int | tuple): qubit label(s).
        """
        self._matrix = matrix
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
    def is_single_qubit(self) -> bool: # TODO: make compatible with qutrits
        """Whether or not the gate acts on a single qubit.

        Returns:
            bool: single-qubit gate or not.
        """
        if len(self.qubits) == 1:
            return True
        else:
            return False
        
    @property
    def is_multi_qubit(self) -> bool: # TODO: make compatible with qutrits
        """Whether or not the gate acts on multiple qubits.

        Returns:
            bool: multi-qubit gate or not.
        """
        if len(self.qubits) > 1:
            return True
        else:
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
    def name(self) -> str:
        """The name of the gate.

        Returns:
            str: name of the gate.
        """
        return self._properties['name']
    
    @property
    def properties(self) -> Dict:
        """Properties of the gate.

        Returns:
            Dict: gate properties.
        """
        return self._properties
    
    @property
    def qubits(self) -> tuple:
        """The qubit(s) that the gate acts on.

        Returns:
            tupe: qubit label(s).
        """
        return self._properties['qubits']