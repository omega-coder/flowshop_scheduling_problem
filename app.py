#!/usr/bin/env python

import datetime
import json

import plotly
import plotly.figure_factory as ff
from flask import Flask, render_template, request
from flowshop import Flowshop

app = Flask(__name__)


def parse_problem_data(data):
    data_splitted = data.split('\n')
    number_machines = int(data_splitted[0].strip())
    number_jobs = int(data_splitted[1].strip())
    processing_t__ = []
    for line in data_splitted[2::]:
        temp = list(map(int, line.strip('\n').split(' ')))
        processing_t__.append(temp)
    print(processing_t__)
    return number_machines, number_jobs, processing_t__


def ganttfig_to_json(fig):
    """
    Transforms a Gantt chart figure to a json using plotly json encoder

    Attributes:
        fig: a plotly gantt chart figure

    Returns:
        str: json representation of a gantt figure
    """
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)


def jobs_to_gantt_fig(scheduled_jobs, nb_machines, nb_jobs):
    """
    returns a plotly figure for a gantt chart made from passed jobs
    Attributes:
        scheduled_jobs (list): List of jobs on machines m1, m2, ..., mn
        nb_machines (int): Number of machines
        nb_jobs (int): number of jobs

    Returns:
        Plotly.figure: a plotly figure for generated gantt chart
    """
    tasks = []
    curr_date = datetime.datetime.now()
    zipped_jobs_list = list(zip(*scheduled_jobs))
    for job in zipped_jobs_list:
        job = list(job)
        for m_id in range(0, nb_machines):
            start_t = (datetime.timedelta(
                minutes=job[m_id]['start_time']) + curr_date).strftime("%Y-%m-%d %H:%M:%S")
            finish_t = (datetime.timedelta(
                minutes=job[m_id]['end_time']) + curr_date).strftime("%Y-%m-%d %H:%M:%S")
            task = {"Task": "M{}".format(
                m_id+1), "Start": start_t, "Finish": finish_t, "Resource": job[m_id]["name"]}
            tasks.append(task)
    fig = ff.create_gantt(tasks, show_colorbar=True, index_col="Resource",
                          showgrid_x=True, showgrid_y=True, group_tasks=True, bar_width=0.1)
    return fig


@app.route('/solve', methods=["POST"])
def solve():
    if request.method == "POST":
        prob = request.get_json()
        pfsp_algorithm = prob["algorithm"]
        data = prob["data"]
        num_machines, num_jobs, procesing_times = parse_problem_data(data)
        problem_inst = Flowshop(procesing_times, num_machines, num_jobs)
        if pfsp_algorithm == "johnson":
            _, jobs_m1, jobs_m2 = problem_inst.solve_johnson()
            print(jobs_m1, jobs_m2)
            fig = jobs_to_gantt_fig([jobs_m1, jobs_m2], num_machines, num_jobs)
            graph_json = ganttfig_to_json(fig)
            response = app.response_class(
                response=graph_json,
                status=200,
                mimetype='application/json',
            )
            return response

    return render_template("index.html", plot=None)


@app.route('/')
def index():
    nb_m = 2
    nb_j = 6
    data = [[6, 2, 10, 4, 5, 3], [5, 4, 3, 8, 2, 4]]
    pfsp = Flowshop(data, nb_m, nb_j)
    _, jobs_m1, jobs_m2 = pfsp.solve_johnson()

    df = []
    curr_date = datetime.datetime.now()
    for i, j in zip(jobs_m1, jobs_m2):
        Start = (
            datetime.timedelta(
                seconds=i["start_time"],
            ) + curr_date
        ).strftime(
            "%Y-%m-%d %H:%M:%S",
        )
        Finish = (
            datetime.timedelta(
                seconds=i["end_time"],
            ) + curr_date
        ).strftime(
            "%Y-%m-%d %H:%M:%S",
        )
        task = {"Task": "M1",
                "Start": Start,
                "Finish": Finish,
                "Resource": i["name"]}
        df.append(task)
        Start = (
            datetime.timedelta(
                seconds=j["start_time"],
            ) + curr_date
        ).strftime(
            "%Y-%m-%d %H:%M:%S",
        )
        Finish = (
            datetime.timedelta(
                seconds=j["end_time"],
            ) + curr_date
        ).strftime(
            "%Y-%m-%d %H:%M:%S",
        )
        task = {"Task": "M2",
                "Start": Start,
                "Finish": Finish,
                "Resource": j["name"]}
        df.append(task)
    fig = ff.create_gantt(
        df,
        group_tasks=True,
        index_col="Resource",
        show_colorbar=True,
        showgrid_x=True,
        showgrid_y=True,
        bar_width=0.1,
    )
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return render_template('index.html', plot=graphJSON)


if __name__ == "__main__":
    app.run(debug=True, port=1337)
