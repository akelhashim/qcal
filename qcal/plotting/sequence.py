"""Submodule for plotting pulses and sequences.

"""
from qcal.config import Config
from qcal.units import MHz, ns

import logging
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

from numpy.typing import NDArray
from plotly.subplots import make_subplots
from typing import List

logger = logging.getLogger(__name__)


def plot_pulse(
        config:          Config,
        pulse_duration:  float,
        pulse_envelope:  NDArray[np.complex64],
        pulse_qubit:     int | None = None,
        neighbor_qubits: List[int] | None = []
    ) -> None:
    """Plot the time and frequency domain of a pulse.

    The frequency domain gives the spectral amplitude of the Fourier spectra of 
    a pulse.

    Args:
        config (Config): qcal Config object.
        pulse_duration (float): duration of pulse in seconds.
        pulse_envelope (NDArray[np.complex64]): pulse envelope.
        pulse_qubit (int): qubit on which the pulse is to be played. Defaults to
            None. If not None, the qubit's frequencies will be plotted.
        neighbor_qubits (List[int] | None): neighboring qubits on which the 
            pulse is not being played. Defaults to None. If not None, the
            frequencies of the qubits will be plotted.
    """
    pio.templates.default = 'plotly'

    t = np.linspace(0, pulse_duration, len(pulse_envelope))
    dt = t[1] - t[0]
    # freq_spectrum = np.fft.fftshift(np.fft.fft(pulse_envelope))
    # freqs = np.fft.fftshift(np.fft.fftfreq(len(pulse_envelope), dt))

    padded_pulse_envelope = np.pad(
        pulse_envelope, 
        (1000, 1000), 
        mode='constant', 
        constant_values=0.0 + 0.0j
    )

    freq_spectrum = np.fft.fftshift(np.fft.fft(padded_pulse_envelope))
    freqs = np.fft.fftshift(np.fft.fftfreq(len(padded_pulse_envelope), dt))
    spectrum_magnitude = np.abs(freq_spectrum)
    
    # Create the plot
    fig = make_subplots(
        rows=2, cols=1,
        # subplot_titles=('Time Domain', 'Frequency Domain'),
        vertical_spacing=0.12,
        specs=[[{"secondary_y": False}],
               [{"secondary_y": False, "type": "scatter"}]]
    )

    # Plot the I and Q components of the pulse
    fig.add_trace(
        go.Scatter(
            x=t/ns,
            y=np.real(pulse_envelope),
            mode='lines',
            name='I (in-phase)',
            line=dict(color='blue', width=2),
            hovertemplate=(
                'Time: %{x:.2f} ns<br>I Amplitude: %{y:.3f}<extra></extra>'
            )
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=t/ns,
            y=np.imag(pulse_envelope),
            mode='lines',
            name='Q (quadrature)',
            line=dict(color='red', width=2, dash='dash'),
            hovertemplate=(
                'Time: %{x:.2f} ns<br>Q Amplitude: %{y:.3f}<extra></extra>'
            )
        ),
        row=1, col=1
    )
    
    # Add spectrum trace
    if pulse_qubit:  # TODO: make compatible with EF
        freqs += config[f'single_qubit/{pulse_qubit}/GE/freq']

    fig.add_trace(
        go.Scatter(
            x=freqs, # / MHz
            y=spectrum_magnitude,
            mode='lines',
            name='|Ω̂_IQ(f)|',
            line=dict(color='black', width=2),
            hovertemplate=(
                'Frequency: %{x:.2f} MHz<br>Amplitude: %{y:.2e}<extra></extra>'
            )
        ),
        row=2, col=1
    )

    if pulse_qubit:  # TODO: make compatible with EF
        fig.add_vline(
            x=config[f'single_qubit/{pulse_qubit}/GE/freq'], 
            line_dash="solid", 
            line_color="blue",
            annotation_text=f"Q{pulse_qubit} EF",
            annotation_position="top",
            row=2, col=1
        )
        fig.add_vline(
            x=config[f'single_qubit/{pulse_qubit}/EF/freq'], 
            line_dash="dash", 
            line_color="blue",
            annotation_text=f"Q{pulse_qubit} EF",
            annotation_position="top",
            row=2, col=1
        )

    for q in neighbor_qubits:
        fig.add_vline(
            x=config[f'single_qubit/{q}/GE/freq'], 
            line_dash="solid", 
            line_color="red",
            annotation_text=f"Q{q} GE",
            annotation_position="top",
            row=2, col=1
        )
        fig.add_vline(
            x=config[f'single_qubit/{q}/EF/freq'], 
            line_dash="dash", 
            line_color="red",
            annotation_text=f"Q{q} EF",
            annotation_position="top",
            row=2, col=1
        )
    
    # Update layout
    fig.update_layout(
        # title='Pulse',
        template="plotly",
        showlegend=True,
        hovermode='x unified',
        width=1000,
        height=800
    )
    
    # Add grid
    fig.update_xaxes(
        title_text="Time (ns)", 
        showgrid=True, 
        gridcolor='lightgray',
        row=1, col=1,
    )
    fig.update_yaxes(
        title_text="Amplitude", 
        showgrid=True, 
        gridcolor='lightgray',
        row=1, col=1, 
    )
    fig.update_xaxes(
        title_text="Frequency (Hz)", 
        showgrid=True, 
        gridcolor='lightgray',
        row=2, col=1, 
    )
    fig.update_yaxes(
        title_text="Spectral Amplitude", 
        type="log", 
        showgrid=True, 
        gridcolor='lightgray',
        row=2, col=1,
    )

    save_properties = {
        'toImageButtonOptions': {
            'format': 'png', # one of png, svg, jpeg, webp
            'filename': 'pulse_time_frequency_domain',
            'height': 800,
            'width': 1000,
            'scale': 10 # Multiply title/legend/axis/canvas sizes by this factor
        }
    }

    fig.show(config=save_properties)
    