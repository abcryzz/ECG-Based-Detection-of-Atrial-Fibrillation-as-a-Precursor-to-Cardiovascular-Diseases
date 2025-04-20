
import numpy as np
import plotly.graph_objs as go
import plotly.io as pio
from plotly.subplots import make_subplots

# 12‑lead ECG lead names
default_lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']


def generate_ecg_interactive(data, sampling_rate=500):
    """
    Create interactive Plotly graphs for ECG data:
      1. Overlapped all leads in one plot
      2. Separated 12‑lead subplots

    Args:
        data (np.ndarray): 2D array of shape (timesteps, 12)
        sampling_rate (int): Hz, samples per second

    Returns:
        tuple of str: (html_all_leads, html_sep_leads)
    """
    # Input validation
    if not isinstance(data, np.ndarray):
        raise TypeError(f"Expected numpy array, got {type(data)}")
    if data.ndim != 2 or data.shape[1] != 12:
        raise ValueError(f"ECG data must be shape (timesteps, 12), got {data.shape}")

    timesteps = data.shape[0]
    time = np.arange(timesteps) / sampling_rate

    # 1. Overlapped plot
    fig_all = go.Figure()
    for idx, lead in enumerate(default_lead_names):
        fig_all.add_trace(
            go.Scatter(x=time, y=data[:, idx], mode='lines', name=lead)
        )
    fig_all.update_layout(
        title='Overlapped 12‑Lead ECG',
        xaxis_title='Time (s)',
        yaxis_title='Amplitude',
        height=400
    )
    html_all = pio.to_html(fig_all, full_html=False, include_plotlyjs='cdn')

    # 2. Separated subplots
    fig_sep = make_subplots(rows=4, cols=3, subplot_titles=default_lead_names)
    for i, lead in enumerate(default_lead_names):
        r, c = (i // 3) + 1, (i % 3) + 1
        fig_sep.add_trace(
            go.Scatter(x=time, y=data[:, i], mode='lines', name=lead, showlegend=False),
            row=r, col=c
        )
        fig_sep.update_xaxes(title_text='Time (s)', row=r, col=c)
        fig_sep.update_yaxes(title_text='Amplitude', row=r, col=c)

    fig_sep.update_layout(
        title='12‑Lead ECG Details',
        height=900,
        width=1200,
        showlegend=False,
        margin=dict(t=50, b=50, l=50, r=50)
    )
    html_sep = pio.to_html(fig_sep, full_html=False, include_plotlyjs=False)

    return html_all, html_sep


def generate_statistics_plotly(channel_stats, reference_stats):
    """
    Create interactive Plotly bar charts comparing per-lead ECG stats
    against reference normal and AF statistics.

    Args:
        channel_stats (dict): {
            lead_name: {'mean': float, 'median': float, 'std': float, ...},
            ... (12 leads)
        }
        reference_stats (dict): {
            'normal': {lead_name: {'mean': float, 'median': float, 'std': float}, ...},
            'af':     {lead_name: {'mean': float, 'median': float, 'std': float}, ...}
        }

    Returns:
        dict: {
            'mean_comparison': html div str,
            'median_comparison': html div str,
            'std_comparison': html div str
        }
    """
    # Ensure consistent lead order
    leads = [lead for lead in default_lead_names if lead in channel_stats]
    metrics = ['mean', 'median', 'std']

    plots = {}
    for idx, metric in enumerate(metrics):
        # Prepare values for each series
        user_vals   = [channel_stats[lead][metric] for lead in leads]
        normal_vals = [reference_stats['normal'][lead][metric] for lead in leads]
        af_vals     = [reference_stats['af'][lead][metric] for lead in leads]

        # Build grouped bar chart
        fig = go.Figure(
            data=[
                go.Bar(name='Your ECG', x=leads, y=user_vals),
                go.Bar(name='Normal',   x=leads, y=normal_vals),
                go.Bar(name='AF',       x=leads, y=af_vals)
            ]
        )
        fig.update_layout(
            barmode='group',
            title=f"{metric.title()} by Lead",
            xaxis_title='Lead',
            yaxis_title=metric.title(),
            height=400
        )

        # Export HTML snippet; include Plotly.js only for the first metric
        include_js = 'cdn' if idx == 0 else False
        html = pio.to_html(fig, full_html=False, include_plotlyjs=include_js)
        plots[f"{metric}_comparison"] = html

    return plots
