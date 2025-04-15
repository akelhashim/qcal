"""Submodule for loading and saving experimental configuration files.

Basic example useage:

    cfg = Config('config.yaml')
    cfg.qubits       # Returns a list of the qubits on the processor
    cfg.qubit_pairs  # Returns a list of the qubit pairs on the processor
    cfg.parameters   # Returns a dictionary of the entire config
    cfg.processor()  # Plots the processor connectivity
"""
import qcal.settings as settings

from qcal.math.utils import round_sig_figures

import copy
import io
import logging
import pandas as pd
import yaml

from collections import defaultdict
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)


__all__ = ['Config']


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
    """Class for loading and storing configuration files."""

    __slots__ = ('_parameters', '_filename')

    def __init__(self, filename: str = None) -> None:

        self._filename = filename
        if filename is not None:
            self.load(filename)
        else:
            try:
                self.load(
                    settings.Settings.config_path + 'config.yaml'
                )
                self._filename = (
                    settings.Settings.config_path + 'config.yaml'
                )
            except Exception:
                logger.warning(' No configuration file was provided.')
                self._parameters = defaultdict(lambda: None, {})

    def __call__(self) -> Dict:
        return dict(self._parameters)

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
        
    def __getitem__(self, param: str) -> Any:
        """Subscript the config object with a string.

        Args:
            param (str): parameter of interest. This should be a string with
                nested keys separate by '/' (e.g. 'single_qubit/0/GE/freq').

        Returns:
            Any: the string or value returned by config[param].
        """
        param = param.split('/')
        param = [eval(p) if p.isdigit() else p for p in param]
        return self.get(param)

    def __setitem__(self, param: str, newvalue: Any):
        """Assign a new value to a parameter in the config.

        Args:
            param (str): parameter of interest. This should be a string with
                nested keys separate by '/' (e.g. 'single_qubit/0/GE/freq').
            newvalue (Any): new value to assign to the parameter.
        """
        param = param.split('/')
        param = [eval(p) if p.isdigit() else p for p in param]
        self.set(param, newvalue)

    # def __iter__(self):
    #     return self._parameters.__iter__()

    def __len__(self) -> int:
        return len(dict(self._parameters))

    # def __next__(self):
    #     return self._parameters.__next__()

    def __repr__(self) -> str:
        return repr(dict(self._parameters))
    
    def __str__(self) -> str:
        return str(dict(self._parameters))
    
    @property
    def native_gates(self) -> Dict:
        """Native gates for each qubit (subspace) and qubit pair.

        Returns:
            Dict: native gates.
        """
        native_gates = dict()
        native_set = set()

        single_qubit = dict()
        for q in self.qubits:
            subspace = {}
            for sbsp in self.parameters['single_qubit'][q].keys():
                gates = []
                for k, v in self.parameters['single_qubit'][q][sbsp].items():
                    if isinstance(v, dict) and 'pulse' in v.keys():
                        gates.append(k)
                        native_set.add(k)
                subspace[sbsp] = gates
            single_qubit[q] = subspace
        native_gates['single_qubit'] = single_qubit

        two_qubit = dict()
        for p in self.qubit_pairs:
            gates = []
            for k, v in self.parameters['two_qubit'][str(p)].items():
                if isinstance(v, dict) and 'pulse' in v.keys():
                        gates.append(k)
                        native_set.add(k)
            two_qubit[p] = gates
        native_gates['two_qubit'] = two_qubit

        native_gates['set'] = native_set

        return native_gates

    @property
    def filename(self) -> str:
        """The filename of the config.yaml file.

        Returns:
            str: filename of the config.yaml file.
        """
        return self._filename
    
    @property
    def hardware(self) -> pd.DataFrame:
        """Hardware parameters in a table format.

        Returns:
            pd.DataFrame: hardware properties.
        """
        return pd.DataFrame.from_dict(
            self.parameters['hardware'], orient='index', columns=['hardware']
        )
    
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
        return dict(self._parameters)
    
    @property
    def readout(self) -> pd.DataFrame:
        """Readout parameters in a table format.

        Returns:
            pd.DataFrame: readout parameters.
        """
        dfs = []
        dfs.append(pd.DataFrame(
                    data=list(map(list, zip(*[
                            [val for val in recursive_values(
                                    self.parameters['readout'][q]
                                )
                            ] for q in self.qubits]
                        ))),
                    columns=self.qubits,
                    index=[
                        item[0] for item in recursive_items(
                            self.parameters['readout'][self.qubits[0]]
                        )
                    ]
                )
        )
                
        for key, value in self.parameters['readout'].items():
            if key not in self.qubits:
                dfs.append(pd.DataFrame.from_dict(
                        {key: [value] * self.n_qubits},
                        orient='index'
                    )
                )

        df = pd.concat(dfs, ignore_index=False)
        return df
    
    @property
    def reset(self) -> pd.DataFrame:
        """Reset parameters in table format.

        Returns:
            pd.DataFrame: reset parameters.
        """
        return pd.DataFrame(
            data=[val for val in recursive_values(self.parameters['reset'])],
            columns=['reset'],
            index=nested_index(
                self.parameters['reset'],
                n_levels=2
            )
        )
    
    @property
    def single_qubit(self) -> pd.DataFrame:
        """Single-qubit parameters in a table format.

        Returns:
            pd.DataFrame: table of single-qubit parameters.
        """
        df = pd.DataFrame(
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
        )
        return df

    @property
    def two_qubit(self) -> pd.DataFrame:
        """Two-qubit parameters in a table format.

        Returns:
            pd.DataFrame: table of two-qubit parameters.
        """
        df = pd.DataFrame(
            data=list(map(list, zip(*[
                    [val for val in recursive_values(
                            self.parameters['two_qubit'][str(p)]
                        )
                    ] for p in self.qubit_pairs]
                ))),
            columns=self.qubit_pairs,
            index=nested_index(
                self.parameters['two_qubit'][str(self.qubit_pairs[0])],
                n_levels=3
            )
        )
        return df
    
    @property
    def qubits(self) -> list:
        """Available qubits on the processor.

        Returns:
            list: qubit labels.
        """
        return tuple(self.parameters['single_qubit'].keys())
    
    @property
    def qubits_even(self) -> list:
        """Even labeled qubits on the process

        Returns:
            list: qubit labels.
        """
        return tuple([q for q in self.qubits if q % 2 == 0])
    
    @property
    def qubits_odd(self) -> list:
        """Odd labeled qubits on the process

        Returns:
            list: qubit labels.
        """
        return tuple([q for q in self.qubits if q % 2 == 1])
    
    @property
    def qubit_pairs(self) -> List[tuple]:
        """Available qubit pairs on the processor.

        Returns:
            list[tuple]: qubit pairs.
        """
        return [eval(key) for key in self.parameters['two_qubit'].keys()]
    
    def get(self, param: List[Any]) -> Any:
        """Get the parameter from the config (if it exists).

        Args:
            param (List[Any]): parameter of interest. This should be a
                list of entries used to index the parameters dictionary (e.g. 
                ['single_qubit', 0, 'GE', 'freq']).

        Returns:
            Any: the string or value returned by config[param]. This defaults
                to None if the parameter cannot be found in the config.
        """
        try:
            cfg_param = self._parameters
            for p in param:
                cfg_param =  cfg_param[p]
            return cfg_param
        except Exception:
            logger.warning(f' Parameter {param} not found in the config!')
            return None
        
    def set(self, param: List[str], newvalue: Any) -> None:
        """Set the parameter in the config to the given value.

        Args:
            param (Union[List[str], str]): parameter of interest.  This should 
                be a list of entries used to index the parameters dictionary 
                (e.g. ['single_qubit', 0, 'GE', 'freq']).
            newvalue (Any): new value to assign to the parameter.
        """
        if isinstance(newvalue, float):
            if abs(newvalue) > 1 or newvalue == 0:
                newvalue = round(float(newvalue), 5)
            elif abs(newvalue) < 1:
                newvalue = round_sig_figures(float(newvalue), 8)
        cfg_param = self.get(param[:-1])
        cfg_param[param[-1]] = newvalue
        logger.info(f' Param {param} set to {newvalue}.')
        
    def items(self) -> tuple:
        return self._parameters.items()

    def load(self, filename: str | None = None):
        """Load the yaml file."""
        if filename is None:
            filename = self._filename

        with open(filename, "r") as config:
            try:
                self._parameters = defaultdict(
                    lambda: None,
                    yaml.load(
                        config, Loader=yaml.FullLoader
                    )
                )
            except yaml.YAMLError as exc:
                logger.error(exc)

    def reload(self):
        """Reload from the yaml file."""
        self.load()

    def draw_qpu(self):
        """Plot a graph displaying the connectivity of the quantum processor.
        """
        from qcal.plotting.graphs import draw_qpu
        draw_qpu(self)

    def plot_freqs(self,
            qubits:      List | Tuple | None = None,
            qubit_pairs: List | Tuple | None = None
        ) -> None:
        """Plot all qubit, two_qubit and readout frequencies.
        
        Args:
            qubits (List | Tuple | None, optional): qubits to plot. Defaults to 
                None.
            qubit_pairs (List | Tuple | None, optional): qubit pairs to plot. 
                Defaults to None.
        """
        from qcal.plotting.frequency import plot_freq_spectrum
        plot_freq_spectrum(
            self, 
            qubits=qubits,
            qubit_pairs=qubit_pairs,
            plot_GE=True, 
            plot_EF=True, # TODO: add checking EF
            plot_readout=True, 
            plot_two_qubit=True
        )

    def plot_qubit_freqs(self,
            qubits:  List | Tuple | None = None,
            plot_EF: bool = False
        ) -> None:
        """Plot all qubit frequencies.

        Args:
            qubits (List | Tuple | None, optional): qubits to plot. Defaults to 
                None.
            plot_EF (bool, optional): plot EF frequencies. Defaults to False.
        """
        from qcal.plotting.frequency import plot_freq_spectrum
        plot_freq_spectrum(
            self, 
            qubits=qubits,
            plot_GE=True, 
            plot_EF=plot_EF,
            plot_readout=False, 
            plot_two_qubit=False
        )

    def plot_readout_freqs(self, qubits:  List | Tuple | None = None):
        """Plot all readout frequencies.
        
        Args:
            qubits (List | Tuple | None, optional): qubits to plot. Defaults to 
                None.
        """
        from qcal.plotting.frequency import plot_freq_spectrum
        plot_freq_spectrum(
            self,
            qubits=qubits,
            plot_GE=False, 
            plot_EF=False,
            plot_readout=True, 
            plot_two_qubit=False
        )

    def save(self, filename: str | None = None):
        """Save the config to a YAML file

        Args:
            filename (str | None, optional): filename for the YAML file.
                Defaults to None.
        """
        with io.open(
                filename if filename is not None else self._filename, 'w', 
                encoding='utf8'
            ) as yaml_file:
                yaml.dump(
                    dict(self._parameters), 
                    yaml_file, 
                    default_flow_style=False, 
                    allow_unicode=True,
                    sort_keys=False
                )
