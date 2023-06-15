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
import pandas as pd
import yaml

from typing import Any, Dict, List, Union

logger = logging.getLogger(__name__)


__all__ = ['Config']


# TODO: make recursive
def nested_index(dictionary: Dict, n_levels: int = 2) -> List[List]:
    """Returns a nested list for a multi-index Dataframe.

    Args:
        dictionary (Dict): nested dictionary.
        n_levels (int, optional): number of recursive levels to index.
            Defaults to 2.

    Returns:
        List[List]: nested index list.
    """
    def itterdict(dct, lst, idx):
        for key, value in dct.items():
            if isinstance(value, dict):
                lst[idx].extend(
                    [key] * sum([len(v) if isinstance(v, dict) else 1 for v in 
                                 value.values()]
                            )
                    )
                itterdict(value, lst, idx+1)
            else:
                index[idx].append(key)
                for n in range(idx+1, len(lst)):
                    index[n].append('')

    index = []
    for _ in range(n_levels):
        index.append(list())

    itterdict(dictionary, index, 0)

    # for key, value in dictionary.items():
    #     if isinstance(value, dict):
    #         index[0].extend([key] * len(value))

    #         for k, v in value.items():
    #             if isinstance(v, dict):
    #                 index[1].extend([k] * len(v))
    #                 for k in v.keys():
    #                     index[2].append(k)

    #             else:
    #                 index[1].append(k)
    #                 for n in range(2, n_levels):
    #                     index[n].append('')
            
    #     else:
    #         index[0].append(key)
    #         for n in range(1, n_levels):
    #             index[n].append('')

    return index


def recursive_items(dictionary: Dict):
    """Yields all of the items in a nested dictionary.

    Args:
        dictionary (Dict): nested dictionary.

    Yields:
        Any: items in nested dictionary.
    """
    for key, value in dictionary.items():
        if isinstance(value, dict):
            yield from recursive_items(value)
        else:
            yield (key, value)


def recursive_values(dictionary: Dict):
    """Yields all of the values in a nested dictionary.

    Args:
        dictionary (Dict): nested dictionary.

    Yields:
        Any: values in nested dictionary.
    """
    for value in dictionary.values():
        if isinstance(value, dict):
            yield from recursive_values(value)
        else:
            yield value


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

    # def __next__(self):
    #     return self._parameters.__next__()

    def __repr__(self) -> str:
        return repr(self._parameters)
    
    def __str__(self) -> str:
        return str(self._parameters)

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
            int: number of qubits.
        """
        return len(self.qubits)

    @property
    def parameters(self) -> Dict:
        """All of the config parameters.

        Returns:
            Dict: config parameters.
        """
        return self._parameters
    
    @property
    def qubits(self) -> list:
        """Available qubits on the processor.

        Returns:
            list: qubit labels.
        """
        return list(
            q for q in self.parameters['single_qubit'].keys() if q[0] == 'Q'
        )
    
    @property
    def qubit_pairs(self) -> List[tuple]:
        """Available qubit pairs on the processor.

        Returns:
            list[tuple]: qubit pairs.
        """
        return [eval(key) for key in self.get('two_qubit').keys()]
    
    @property
    def single_qubit(self) -> pd.DataFrame:
        """Single-qubit parameters in a table format.

        Returns:
            pd.DataFrame: table of single-qubit parameters.
        """
        
        dfs = []
        dfs.append(pd.DataFrame(
            data=list(map(list, zip(*[
                    [val for val in recursive_values(
                            self.parameters['single_qubit'][q]
                        )
                    ] for q in self.qubits]
                ))),
            columns=self.qubits,
            index=nested_index(
                self.parameters['single_qubit'][self.qubits[0]],
                n_levels=3
            )
        ))

        # for key in self.parameters['single_qubit'].keys():
        #     if key not in self.qubits:
        #         dfs.append(
        #             pd.DataFrame(
        #                 data=[val for val in recursive_values(
        #                         self.parameters['single_qubit'][key]
        #                     )],
        #                 columns=[key],
        #                 index=nested_index(
        #                         self.parameters['single_qubit'][key]
        #                     )
        #             )
        #         )

        return pd.concat(dfs, join='outer')
    
    def get(self, param: Union[List[str], str]) -> Any:
        """Get the parameter from the config (if it exists).

        Args:
            param (Union[List[str], str]): parameter of interest. This can be a
                str (e.g. 'single_qubit') or a list of strings (e.g.
                ['single_qubit', 'Q0', 'GE', 'freq']).

        Returns:
            Any: the string or value returned by config[param]. This defaults
                to None if the parameter cannot be found in the config.
        """
        try:
            if isinstance(param, list):
                params = self._parameters
                for p in param:
                    params =  params[p]
                return params
            else:        
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
        from qcal.plotting.graphs import draw_processor
        draw_processor(self)

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
