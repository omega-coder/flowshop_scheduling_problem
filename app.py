#!/usr/bin/env python

import flask
from flask import Flask, render_template, url_for, abort
import plotly
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import json




def create_plot():
    n = 30
    x = np.linspace(0, 1, n)
    y = np.random.randn(n)
    df = pd.DataFrame({'x': x, 'y': y})

    data = [
        go.Bar(
            x = df["x"],
            y = df["y"]
        )
    ]

    graphJSON = json.dumps(data, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON


app = Flask(__name__)

@app.route('/')
def index():
    bar = create_plot()
    return render_template('index.html', plot=bar)




if __name__ == "__main__":
    app.run(debug=True, port=1337)







