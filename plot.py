#!/usr/bin/env python3

import argparse
import pandas as pd
import plotly.express as px

parser = argparse.ArgumentParser()
parser.add_argument("inputs", nargs='*',
                    default=["controllers/motor_controller/robot_data.csv",
                             "controllers/supervisor/simulator_data.csv"],
                    help="The csv files containing the simulation output")
parser.add_argument("--tmin", type=float, default=0.0,
                    help="Data with t<tmin are ignored")
parser.add_argument("--output", type=str, default=None,
                    help="Output file, if not provided, output is shown on browser")
args = parser.parse_args()

df = None
for path in args.inputs:
    tmp_df = pd.read_csv(path)
    tmp_df = tmp_df[tmp_df.t >= args.tmin]
    if df is None:
        df = tmp_df
    else:
        df = pd.concat([df, tmp_df])

fig = px.scatter(df, x="t", y="value", color="source",
                 facet_row="order", facet_col="variable")
fig.update_traces(marker_size=3)
fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[1]))
fig.update_layout(font=dict(size=14))
# Free some space if there is an unique source
if len(pd.unique(df.source)) == 1:
    fig.update_layout(showlegend=False)
# Free 'y' scale are suited, but only when there is a single variable
if len(pd.unique(df.variable)) == 1:
    # Dirty hack to remove facet_col legend
    fig['layout']['annotations'][0]['text'] = ''
    fig.update_yaxes(matches=None, title="")
if args.output is None:
    fig.show()
else:
    fig.update_layout(width=1200, height=600)
    fig.write_image(args.output)
