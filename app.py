#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import datetime
import json
import time
import numpy as np
import plotly
import plotly.figure_factory as ff
from flask import Flask, render_template, request
import re
from flowshop import Flowshop, RandomFlowshop

app = Flask(__name__)


def parse_problem_data(data):
    split_re = re.compile(r"\d+")
    data_splitted = data.split('\n')
    processing_t__ = []
    for line in data_splitted:
        parsed_ints = re.findall(split_re, line)
        temp = list(map(int, list(parsed_ints)))
        processing_t__.append(temp)
    return processing_t__


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
    def r(): return np.random.randint(0, 256, dtype=int)
    colors = ['#%02X%02X%02X' % (r(), r(), r()) for _ in range(nb_jobs)]

    tasks = []
    curr_date = datetime.datetime.now()
    zipped_jobs_list = list(zip(*scheduled_jobs))
    for job in zipped_jobs_list:
        job = list(job)
        for m_id in range(0, nb_machines):
            start_t = (datetime.timedelta(
                minutes=int(job[m_id]['start_time'])) + curr_date).strftime("%Y-%m-%d %H:%M:%S")
            finish_t = (datetime.timedelta(
                minutes=int(job[m_id]['end_time'])) + curr_date).strftime("%Y-%m-%d %H:%M:%S")
            task = {"Task": "M{}".format(
                m_id+1), "Start": start_t, "Finish": finish_t, "Resource": job[m_id]["name"]}
            tasks.append(task)
    fig = ff.create_gantt(tasks, show_colorbar=True, index_col="Resource",
                          showgrid_x=True, showgrid_y=True, group_tasks=True, bar_width=0.08, colors=colors)
    return fig


def random_johnson(nb_machines, nb_jobs):
    random_problem = RandomFlowshop(nb_machines, nb_jobs)
    rand_prob_inst = random_problem.get_problem_instance()
    seq, jobs, opt_makespan, t_t = rand_prob_inst.solve_johnson()
    fig = jobs_to_gantt_fig(jobs, random_problem.get_number_machines(
    ), random_problem.get_number_jobs())
    gantt_json = ganttfig_to_json(fig)
    return gantt_json, seq, opt_makespan, t_t


@app.route('/solve', methods=["POST"])
def solve():
    if request.method == "POST":
        prob = request.get_json()
        pfsp_algorithm = prob["algorithm"]
        data = prob["data"]
        num_machines = int(prob["nb_machines"])
        num_jobs = int(prob["nb_jobs"])
        procesing_times = parse_problem_data(data)
        problem_inst = Flowshop(procesing_times, num_machines, num_jobs)
        if pfsp_algorithm == "johnson":
            seq, jobs, optim_makespan, t_t = problem_inst.solve_johnson()

            fig = jobs_to_gantt_fig(jobs, num_machines, num_jobs)
            graph_json = ganttfig_to_json(fig)
            if float(t_t) * 1000 > 1000.0:
                time_ts = t_t
                time_typ = "seconds"
            else:
                time_ts = float(t_t * 1000)
                time_typ = "msecs"
            resp = json.dumps(
                {"graph": graph_json, "optim_makespan": optim_makespan, "opt_seq": seq, "t_time": time_ts, "tt": time_typ})

            response = app.response_class(
                response=resp,
                status=200,
                mimetype='application/json',
            )
            return response
        elif pfsp_algorithm == "palmer":
            seq, jobs, opt_makespan, t_t = problem_inst.palmer_heuristic()
            fig = jobs_to_gantt_fig(jobs, num_machines, num_jobs)
            graph_json = ganttfig_to_json(fig)

            if float(t_t) * 1000 > 1000.0:
                time_ts = t_t
                time_typ = "seconds"
            else:
                time_ts = float(t_t * 1000)
                time_typ = "msecs"
            resp = json.dumps(
                {"graph": graph_json, "optim_makespan": opt_makespan, "opt_seq": seq, "t_time": time_ts, "tt": time_typ})
            response = app.response_class(
                response=resp,
                status=200,
                mimetype="application/json",
            )
            return response
        elif pfsp_algorithm == "neh":
            seq, jobs, opt_makespan, t_t = problem_inst.neh_heuristic()
            fig = jobs_to_gantt_fig(jobs, num_machines, num_jobs)
            graph_json = ganttfig_to_json(fig)
            if float(t_t) * 1000 > 1000.0:
                time_ts = t_t
                time_typ = "seconds"
            else:
                time_ts = float(t_t * 1000)
                time_typ = "msecs"
            resp = json.dumps(
                {"graph": graph_json, "optim_makespan": opt_makespan, "opt_seq": seq, "t_time": time_ts, "tt": time_typ})
            response = app.response_class(
                response=resp,
                status=200,
                mimetype="application/json",
            )
            return response
        elif pfsp_algorithm == "bruteforce":
            seq, jobs, opt_makespan, t_t = problem_inst.brute_force_exact()
            fig = jobs_to_gantt_fig(jobs, num_machines, num_jobs)
            graph_json = ganttfig_to_json(fig)
            if float(t_t) * 1000 > 1000.0:
                time_ts = t_t
                time_typ = "seconds"
            else:
                time_ts = float(t_t * 1000)
                time_typ = "msecs"
            resp = json.dumps(
                {"graph": graph_json, "optim_makespan": opt_makespan, "opt_seq": seq, "t_time": time_ts, "tt": time_typ})
            response = app.response_class(
                response=resp,
                status=200,
                mimetype="application/json",
            )
            return response

        elif pfsp_algorithm == "genetic-algorithm":
            population_number = int(prob["population_number"])
            it_number = int(prob["it_number"])
            p_crossover = float(prob["p_crossover"])
            p_mutation = float(prob["p_mutation"])
            nograph = prob["nograph"]
            if nograph:
                seq, jobs, opt_makespan, t_t = problem_inst.genetic_algorithm(population_number, it_number, p_crossover, p_mutation, nograph=True)    
                print("No Graph genetic done")
            else:
                seq, jobs, opt_makespan, t_t = problem_inst.genetic_algorithm(population_number, it_number, p_crossover, p_mutation)
                fig = jobs_to_gantt_fig(jobs, num_machines, num_jobs)
                graph_json = ganttfig_to_json(fig)
            
            if float(t_t) * 1000 > 1000.0:
                time_ts = t_t
                time_typ = "seconds"
            else:
                time_ts = float(t_t * 1000)
                time_typ = "msecs"
            if nograph:
                resp = json.dumps(
                    {"graph": None, "optim_makespan": opt_makespan, "opt_seq": seq, "t_time": time_ts, "tt": time_typ})
            else:
                resp = json.dumps(
                    {"graph": graph_json, "optim_makespan": opt_makespan, "opt_seq": seq, "t_time": time_ts, "tt": time_typ})
            
            response = app.response_class(
                response=resp,
                status=200,
                mimetype="application/json",
            )
            return response

        elif pfsp_algorithm == "cds":
            seq, jobs, opt_makespan, t_t = problem_inst.cds()
            fig = jobs_to_gantt_fig(jobs, num_machines, num_jobs)
            graph_json = ganttfig_to_json(fig)
            if float(t_t) * 1000 > 1000.0:
                time_ts = t_t
                time_typ = "seconds"
            else:
                time_ts = float(t_t * 1000)
                time_typ = "msecs"
            resp = json.dumps(
                {"graph": graph_json, "optim_makespan": opt_makespan, "opt_seq": seq, "t_time": time_ts, "tt": time_typ})
            response = app.response_class(
                response=resp,
                status=200,
                mimetype="application/json",
            )
            return response

        elif pfsp_algorithm == "simulated-annealing":
            ti = float(prob["ti"])
            tf = float(prob["tf"])
            alpha = float(prob["alpha"])
            seq, jobs, opt_makespan, t_t = problem_inst.simulated_annealing(ti, tf, alpha)
            fig = jobs_to_gantt_fig(jobs, num_machines, num_jobs)
            graph_json = ganttfig_to_json(fig)
            if float(t_t) * 1000 > 1000.0:
                time_ts = t_t
                time_typ = "seconds"
            else:
                time_ts = float(t_t * 1000)
                time_typ = "msecs"
            resp = json.dumps(
                {"graph": graph_json, "optim_makespan": opt_makespan, "opt_seq": seq, "t_time": time_ts, "tt": time_typ})
            response = app.response_class(
                response=resp,
                status=200,
                mimetype="application/json",
            )
            return response


@app.route('/random', methods=["POST"])
def random():
    if request.method == "POST":
        req = request.get_json()
        nb_machines = int(req["nb_machines"])
        nb_jobs = int(req["nb_jobs"])
        random_problem = RandomFlowshop(nb_machines, nb_jobs)
        data = random_problem.get_data()
        resp_data = ""
        for i in data:
            resp_data += " ".join(map(lambda x: str(x), i)) + "\n"

        return resp_data.rstrip('\n')
    # will never reach this hahahha
    return "This is not intended for you to call :D"


@app.route('/')
def index():
    graph_json, seq, opt_makespan, t_t = random_johnson(2, 6)
    return render_template('index.html', plot=graph_json, seq=seq, opt_makespan=opt_makespan, time=t_t)


if __name__ == "__main__":
    app.run(debug=False, port=1337)
