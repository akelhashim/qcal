"""Submodule for loading and saving experimental configuration files.

Basic example useage:

    cfg = Config('config.yaml')
    cfg.qubits       # Returns a list of the qubits on the processor
    cfg.qubit_pairs  # Returns a list of the qubit pairs on the processor
    cfg.parameters   # Returns a dictionary of the entire config
    cfg.processor()  # Plots the processor connectivity
"""
import copy
import io
import logging
import yaml

from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class Config:
    """Class for loading and storing configuration files.
    """

    __slots__ = ['_parameters', '_filename']

    def __init__(self, filename: str = None) -> None:

        self._filename = filename
        if filename is not None:
            self.load(filename)
        else:
            logger.warning('No configuration file was provided.')
            self._parameters = {}

    def __call__(self) -> Dict:
        return self._parameters

    def __copy__(self, deep_copy=False) -> Dict:
        """Copy the config dictionary.

        Args:
            deep_copy (bool, optional): deep copy of the config dictionary.
                Defaults to False.

        Returns:
            dict: copy of the config dictionary.
        """
        if deep_copy is True:
            return copy.deepycopy(self._parameters)
        else:
            return self._parameters.copy()

    def __iter__(self):
        return self._parameters.__iter__()

    def __len__(self) -> int:
        return len(self._parameters)

    def __next__(self):
        return self._parameters_dict.__next__()

    def __str__(self) -> str:
        return str(self._parameters_dict)

    def __repr__(self) -> str:
        return repr(self._parameters)

    @property
    def filename(self) -> str:
        """Returns the filename of the config.yaml file.

        Returns:
            str: filename of the config.yaml file.
        """
        return self._filename
    
    @property
    def n_qubits(self) -> int:
        """Number of qubits on the processor.

        Returns:
            int: number of qubits
        """
        return len(self.qubits)

    @property
    def parameters(self) -> Dict:
        """All of the config parameters.

        Returns:
            Dict: config parameters
        """
        return self._parameters
    
    @property
    def qubits(self) -> list:
        """Available qubits on the processor.

        Returns:
            list: qubit labels
        """
        return list(self.get('qubits').keys())
    
    @property
    def qubit_pairs(self) -> List[tuple]:
        """Available qubit pairs on the processor.

        Returns:
            list[tuple]: qubit pairs
        """
        return [eval(key) for key in self.get('qubit_pairs').keys()]
    
    def get(self, param: str) -> Any:
        """Get the parameter from the config (if it exists).

        Args:
            param (str): parameter of interest

        Returns:
            Any: the string or value returned by config[param]. This defaults
                to None if the parameter cannot be found in the config.
        """
        try:
            return self._parameters[param]
        except Exception:
            logger.warning(f"Parameter '{param}' not found in the config!")
            return None
        
    def items(self) -> tuple:
        return self._parameters.items()

    def load(self, yaml_file):

        # TODO: add automatic saved backup of file after each load

        with open(yaml_file, "r") as config:
            try:
                self._parameters = yaml.load(config, Loader=yaml.FullLoader)
            except yaml.YAMLError as exc:
                print(exc)

    def processor(self):
        """Plot a graph displaying the connectivity of the quantum processor.
        """
        from qcal.plotting.graphs import draw_processor_connectivity
        draw_processor_connectivity(self)

    def save(self, filename: str = None):

        # assert self._yaml_file is not None, ""

        with io.open(
                filename if filename is not None else self._yaml_file, 'w', 
                encoding='utf8'
            ) as yaml_file:
                yaml.dump(
                    self._parameters, yaml_file, default_flow_style=False, 
                    allow_unicode=True
                )
