"""Submodule for ploting Cumulative Distribution Functions.

"""
import logging
import pandas as pd
import plotly.express as px

logger = logging.getLogger(__name__)


def CDF(df: pd.DataFrame, x: str, color: str) -> None:
    """Empirical Cumulative Distribution Function plot.

    Args:
        df (pd.DataFrame): Pandas DataFrame.
        x (str): name of coloumn with the data to be plotted.
        color (str): name of column with labels for sorting the data.
    """
    fig = px.ecdf(df, x=x, color=color, marginal="histogram")
    fig.update_traces(
        nbinsx=50,  # number of bins for X-axis histogram
        selector=dict(type="histogram")
    )
    fig.update_layout(
        yaxis_title="Cumulative Probability",
        
        # Change axis title font sizes
        xaxis_title_font=dict(size=20),
        yaxis_title_font=dict(size=20),
        
        # Change tick label font sizes
        xaxis_tickfont=dict(size=14),
        yaxis_tickfont=dict(size=14)
    )

    save_properties = {
        'toImageButtonOptions': {
            'format': 'png', # one of png, svg, jpeg, webp
            'filename': 'frequency_plot',
            'height': 500,
            'width': 1000,
            'scale': 10 # Multiply title/legend/axis/canvas sizes by this factor
        }
    }
    fig.show(config=save_properties)