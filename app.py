#!/usr/bin/env python

import flask
from flask import Flask, render_template, url_for, abort


app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index')


if __name__ == "__main__":
    app.run(debug=True, port=1337)







