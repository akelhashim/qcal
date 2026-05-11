"""Submodule for plotting pulses and sequences.

"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from numpy.typing import NDArray
from plotly.subplots import make_subplots

if TYPE_CHECKING:
    from qcal.circuit import Circuit

from qcal.config import Config
from qcal.sequence.pulse_envelopes import cosine_square, gaussian, square
from qcal.units import ns, us

logger = logging.getLogger(__name__)


def plot_pulse(
        config:          Config,
        pulse_duration:  float,
        pulse_envelope:  NDArray[np.complex64],
        pulse_qubit:     int | None = None,
        neighbor_qubits: List[int] | None = None
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

    if neighbor_qubits is None:
        neighbor_qubits = []

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
            line={'color': 'blue', 'width': 2},
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
            line={'color': 'red', 'width': 2, 'dash': 'dash'},
            hovertemplate=(
                'Time: %{x:.2f} ns<br>Q Amplitude: %{y:.3f}<extra></extra>'
            )
        ),
        row=1, col=1
    )

    # Add spectrum trace
    if pulse_qubit is not None:  # TODO: make compatible with EF
        freqs += config[f'single_qubit/{pulse_qubit}/GE/freq']

    fig.add_trace(
        go.Scatter(
            x=freqs, # / MHz
            y=spectrum_magnitude,
            mode='lines',
            name='|Ω̂_IQ(f)|',
            line={'color': 'black', 'width': 2},
            hovertemplate=(
                'Frequency: %{x:.2f} MHz<br>Amplitude: %{y:.2e}<extra></extra>'
            )
        ),
        row=2, col=1
    )

    if pulse_qubit is not None:  # TODO: make compatible with EF
        fig.add_vline(
            x=config[f'single_qubit/{pulse_qubit}/GE/freq'],
            line_dash="solid",
            line_color="blue",
            annotation_text=f"Q{pulse_qubit} GE",
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


def plot_mock_sequence(
        circuit:                Circuit,
        single_qubit_gate_time: float = 20 * ns,
        two_qubit_gate_time:    float = 100 * ns,
        measurement_time:       float = 1 * us,
    ) -> None:
    """Plot a mock pulse sequence for a qcal Circuit.

    Each gate in each cycle is rendered as a pulse envelope on a per-qubit
    subplot: Gaussian for single-qubit gates, cosine-square for multi-qubit
    gates, and square for measurements. Rz/VirtualZ gates are virtual (zero
    duration) and are omitted. Idle gates use the duration stored in their
    parameters.

    Args:
        circuit (Circuit): qcal Circuit to plot.
        single_qubit_gate_time (float): duration of a single-qubit gate in
            seconds. Defaults to 20 ns.
        two_qubit_gate_time (float): duration of a two-qubit gate in seconds.
            Defaults to 100 ns.
        measurement_time (float): duration of a measurement in seconds. Defaults
            to 1 us.
    """
    pio.templates.default = 'plotly'

    SAMPLE_RATE = 8e9
    QUBIT_COLORS = [
        'steelblue', 'firebrick', 'seagreen', 'darkorange', 'mediumpurple',
        'saddlebrown', 'deeppink', 'teal', 'goldenrod', 'slategray',
    ]

    all_qubits = sorted(circuit.qubits)
    n_qubits = len(all_qubits)
    qubit_row   = {q: i + 1 for i, q in enumerate(all_qubits)}
    qubit_color = {
        q: QUBIT_COLORS[i % len(QUBIT_COLORS)] for i, q in enumerate(all_qubits)
    }

    vertical_spacing = min(0.08, 0.6 / max(1, n_qubits - 1))
    fig = make_subplots(
        rows=n_qubits, cols=1,
        shared_xaxes=True,
        vertical_spacing=vertical_spacing,
        subplot_titles=[f'Q{q}' for q in all_qubits],
    )

    t_current = 0.0

    for cycle in circuit.cycles:
        if cycle.is_barrier:
            continue

        cycle_duration = 0.0

        for gate in cycle.gates:
            name = gate.properties['name']
            gate_qubits = gate.qubits

            if gate.is_measurement:
                duration = measurement_time
                envelope_fn = square
                amp = 1.0
            elif gate.is_multi_qubit:
                duration = two_qubit_gate_time
                envelope_fn = cosine_square
                amp = 1.0
            elif gate.is_single_qubit:
                if name in ('Rz', 'S', 'Sdag', 'VirtualZ', 'Z', 'Z90'):
                    duration = 0.0
                elif name == 'Idle':
                    duration = gate.properties['params']['duration']
                    envelope_fn = square
                    amp = 0.0
                else:
                    duration = single_qubit_gate_time
                    envelope_fn = gaussian
                    amp = 1.0
            else:
                continue

            cycle_duration = max(cycle_duration, duration)

            if duration == 0.0:
                continue

            envelope = envelope_fn(duration, SAMPLE_RATE, amp=amp)
            t = np.linspace(t_current, t_current + duration, len(envelope))
            label = f"{name}({', '.join(str(q) for q in gate_qubits)})"

            for q in gate_qubits:
                if q not in qubit_row:
                    continue
                fig.add_trace(
                    go.Scatter(
                        x=t / ns,
                        y=np.real(envelope),
                        mode='lines',
                        name=label,
                        line={'color': qubit_color[q], 'width': 2},
                        showlegend=False,
                        hovertemplate=(
                            f'{label}<br>'
                            'Time: %{x:.2f} ns<br>'
                            'Amplitude: %{y:.3f}<extra></extra>'
                        ),
                    ),
                    row=qubit_row[q], col=1,
                )

        t_current += cycle_duration

    plot_height = max(400, 200 * n_qubits + 100)

    fig.update_layout(
        template='plotly',
        hovermode='x unified',
        showlegend=False,
        width=1200,
        height=plot_height,
    )

    for i in range(1, n_qubits + 1):
        fig.update_yaxes(
            title_text='Amplitude',
            showgrid=True,
            gridcolor='lightgray',
            row=i, col=1,
        )

    fig.update_xaxes(
        title_text='Time (ns)',
        showgrid=True,
        gridcolor='lightgray',
        row=n_qubits, col=1,
    )

    save_properties = {
        'toImageButtonOptions': {
            'format': 'png',
            'filename': 'mock_pulse_sequence',
            'height': plot_height,
            'width': 1200,
            'scale': 3,
        }
    }

    fig.show(config=save_properties)
