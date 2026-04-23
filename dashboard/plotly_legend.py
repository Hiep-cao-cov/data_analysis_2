"""Reusable Plotly layout snippets so chart appearance stays consistent."""


def update_legend_horizontal_bottom(fig, legend_fontsize):
    if not fig:
        return
    fig.update_layout(
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.2,
            xanchor="center",
            x=0.5,
            font=dict(size=legend_fontsize),
        )
    )


def update_legend_vertical_right(fig, legend_fontsize):
    if not fig:
        return
    fig.update_layout(
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.02,
            font=dict(size=legend_fontsize),
        )
    )
