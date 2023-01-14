# %%


# trace = go.Pie(labels=labels, values=values,
#                hoverinfo='label+percent', textinfo='value',
#                textfont=dict(size=20),
#                marker=dict(colors=colors,
#                            line=dict(color='rgb(100,100,100)',
#                                      width=1)
#                           )
import json
from sklearn.metrics import auc, roc_curve
from matplotlib import pyplot as plt
import plotly.express as px
from pathlib import Path
import plotly.graph_objects as go
import pandas as pd

colors = ['aliceblue',  'aqua', 'aquamarine', 'darkturquoise']


def plot_roc(fpr, tpr, auc):
    df = pd.DataFrame({'fpr': fpr, "tpr": tpr})
    fig = px.area(df,
                  x="fpr", y="tpr", color_discrete_sequence=['darkturquoise'],
                  # x="fpr", y="tpr",color_discrete_sequence=['#a389d4'],
                    title=f'ROC Curve (AUC={auc:.4f})',
                  labels=dict(x='False Positive Rate', y='True Positive Rate'),
                  width=467, height=333,
                  )
    # .update_traces(marker=dict(color='#a389d4'))
    fig.add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=0, y1=1
    )
    # fig.update_traces(marker=dict(color='red'))
    fig.update_yaxes(scaleanchor="x", scaleratio=1, showgrid=False)
    fig.update_xaxes(constrain='domain', showgrid=False)
    fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)',
                       'paper_bgcolor': 'rgba(0, 0, 0, 0)'})
    fig.update_layout(
        font=dict(
            # family="Courier New, monospace",
            size=10,
            color="darkturquoise")
    )
    # fig.show()
    return fig


# %%
fpr = range(1, 5)
tpr = [10, 23, 15, 10]
plot_roc(fpr, tpr, 1)
# %%
