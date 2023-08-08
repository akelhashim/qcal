"""Submodule for RB routines.

"""
from qcal.config import Config
from qcal.qpu.qpu import QPU

import logging

from typing import Any, Callable, List, Tuple, Iterable

logger = logging.getLogger(__name__)


def SRB(
        
    ) -> Callable:


    class SRB:
        """True-Q SRB protocol."""
        import trueq as tq

        def __init__(self,
                qpu:             QPU,
                config:          Config,
                qubit_labels:    Iterable[int, Iterable[int]],
                circuit_depths:  List[int] | Tuple[int],
                compiler:        Any | None = None, 
                transpiler:      Any | None = None,
                n_circuits:      int = 30,
                n_shots:         int = 1024, 
                n_batches:       int = 1, 
                n_circs_per_seq: int = 1,
                compiled_pauli:  bool = True,
                include_rcal:    bool = True,
                **kwargs
            ) -> None:
            import trueq as tq