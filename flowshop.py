#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import time
from functools import lru_cache
from itertools import permutations
from random import randint, shuffle

import numpy as np

from geneticFunctions import *


class Flowshop(object):
    """
    A class for initiaizing & solving a Permutation Flowshop Scheduling Problem
    """

    def __init__(self, data=None, nb_machines=2, nb_jobs=6):
        """[summary]

        Keyword Arguments:
            data {list} -- A 2D array of processing time on machines (default: {None})
            nb_machines {int} -- Number of machines for the given problem must be the number of rows of first param (default: {2})
            nb_jobs {[type]} -- Number of jobs for the given problem, must be equal to the number of columns of the data param. (default: {6})
        """

        self.nb_machines = nb_machines
        self.nb_jobs = nb_jobs
        if data is not None:
            self.data = data
        else:
            self.data = RandomFlowshop(
                self.nb_machines, self.nb_jobs).get_data()

    def solve_johnson(self):
        """Solves a permutation flowshop problem using johnson's rule for a permutation problem of 2 machines and N jobs

        Raises:
            Exception: Raises exception when given a problem with more than 2 machines

        Returns:
            tuple -- returns a tuple containing the optimal sequence of scheduled jobs and a list of scheduled jobs on the 2 machines
        """

        if self.nb_machines != 2:
            raise Exception(
                "Johnson's algorithm only possible for a 2 machine problem"
            )
        default_timer = None
        if sys.platform == "win32":
            default_timer = time.clock
        else:
            default_timer = time.time
        s = default_timer.__call__()
        # Build optimal sequence array
        machine_1_sequence = [j for j in range(
            self.nb_jobs) if self.data[0][j] <= self.data[1][j]]
        machine_1_sequence.sort(key=lambda x: self.data[0][x])
        machine_2_sequence = [j for j in range(
            self.nb_jobs) if self.data[0][j] > self.data[1][j]]
        machine_2_sequence.sort(key=lambda x: self.data[1][x], reverse=True)
        seq = machine_1_sequence + machine_2_sequence
        e = default_timer.__call__()
        jobs_m1, jobs_m2 = [], []
        job_name_rad = "job_"
        job = {"name": job_name_rad +
               str(
                   seq[0]+1,
               ),
               "start_time": 0,
               "end_time": self.data[0][seq[0]]}
        jobs_m1.append(job)
        job = {"name": job_name_rad+str(seq[0]+1), "start_time": self.data[0]
               [seq[0]],
               "end_time": self.data[0][seq[0]] + self.data[1][seq[0]]}
        jobs_m2.append(job)

        for job_id in seq[1::]:
            # job on machine 1
            job_name = job_name_rad + str(job_id + 1)
            job_start_m1 = jobs_m1[-1]["end_time"]
            job_end_m1 = job_start_m1 + self.data[0][job_id]
            job = {"name": job_name, "start_time": job_start_m1,
                   "end_time": job_end_m1}
            jobs_m1.append(job)

            # job on machine 2
            job_start_m2 = max(job_end_m1, jobs_m2[-1]["end_time"])
            job_end_m2 = job_start_m2 + self.data[1][job_id]

            job = {"name": job_name, "start_time": job_start_m2,
                   "end_time": job_end_m2}
            jobs_m2.append(job)
        optim_makespan = int(jobs_m2[-1]["end_time"])
        t_t = e - s
        return seq, [jobs_m1, jobs_m2], optim_makespan, t_t 

    @staticmethod
    def johnson_seq(data):
        # data matrix must have only two machines
        nb_machines = len(data)
        nb_jobs = len(data[0])
        machine_1_sequence = [j for j in range(
            nb_jobs) if data[0][j] <= data[1][j]]
        machine_1_sequence.sort(key=lambda x: data[0][x])
        machine_2_sequence = [j for j in range(
            nb_jobs) if data[0][j] > data[1][j]]
        machine_2_sequence.sort(key=lambda x: data[1][x], reverse=True)
        seq = machine_1_sequence + machine_2_sequence
        return seq

    @staticmethod
    def johnson_seq_var_2(data):
        job_count = len(data)
        job_ids = list(range(0, job_count))
        l1 = []
        l2 = []
        for job_info in sorted(zip(job_ids, data), key=lambda t: min(t[1])):
            job_id = job_info[0]
            job_times = job_info[1]
            if job_times[0] < job_times[1]:
                l1.append(job_id)
            else:
                l2.insert(0, job_id)
        return l1 + l2



    def cds(self):
        if type(self.data) is not np.ndarray:
            data_ndarray = np.array(self.data)
        else:
            data_ndarray = self.data
        data_transposed = data_ndarray.T
        default_timer = None
        if sys.platform == "win32":
            default_timer = time.clock
        else:
            default_timer = time.time
        s = default_timer.__call__()

        merged_times = [[0, sum(j_t)] for j_t in data_transposed]
        perms = []
        for i in range(0, self.nb_machines-1):
            for j in range(0, self.nb_jobs):
                merged_times[j][0] += data_transposed[j][i]
                merged_times[j][1] -= data_transposed[j][i]
            perms.append(Flowshop.johnson_seq_var_2(merged_times))
        
        seq = min(perms, key=lambda p: self._get_makespan(p, self.data))

        e = default_timer.__call__()

        schedules = np.zeros((self.nb_machines, self.nb_jobs), dtype=dict)
        # schedule first job alone first
        task = {"name": "job_{}".format(
            seq[0]+1), "start_time": 0, "end_time": self.data[0][seq[0]]}
        schedules[0][0] = task
        for m_id in range(1, self.nb_machines):
            start_t = schedules[m_id-1][0]["end_time"]
            end_t = start_t + self.data[m_id][0]
            task = {"name": "job_{}".format(
                seq[0]+1), "start_time": start_t, "end_time": end_t}
            schedules[m_id][0] = task

        for index, job_id in enumerate(seq[1::]):
            start_t = schedules[0][index]["end_time"]
            end_t = start_t + self.data[0][job_id]
            task = {"name": "job_{}".format(
                job_id+1), "start_time": start_t, "end_time": end_t}
            schedules[0][index+1] = task
            for m_id in range(1, self.nb_machines):
                start_t = max(schedules[m_id][index]["end_time"],
                              schedules[m_id-1][index+1]["end_time"])
                end_t = start_t + self.data[m_id][job_id]
                task = {"name": "job_{}".format(
                    job_id+1), "start_time": start_t, "end_time": end_t}
                schedules[m_id][index+1] = task
        max_mkspn = int(schedules[self.nb_machines-1][-1]["end_time"])
        t_t = e - s
        return seq, schedules, max_mkspn, t_t

    def palmer_heuristic(self):
        """solves an N machines M jobs pfsp problem using Palmer's Heuristic
        Returns:
            tuple -- a tuple containing the job sequence, scheduled jobs and optimal makespan.
        """

        def palmer_f(x): return -(self.nb_machines - (2*x - 1))
        
        default_timer = None
        if sys.platform == "win32":
            default_timer = time.clock
        else:
            default_timer = time.time
        s = default_timer.__call__()
        
        weights = list(map(palmer_f, range(1, self.nb_machines+1)))
        ws = []
        for job_id in range(self.nb_jobs):
            p_ij = sum([self.data[j][job_id]*weights[j]
                        for j in range(self.nb_machines)])
            ws.append((job_id, p_ij))
        ws.sort(key=lambda x: x[1], reverse=True)
        h_seq = [x[0] for x in ws]
        e = default_timer.__call__()
        schedules = np.zeros((self.nb_machines, self.nb_jobs), dtype=dict)
        # schedule first job alone first
        task = {"name": "job_{}".format(
            h_seq[0]+1), "start_time": 0, "end_time": self.data[0][h_seq[0]]}
        schedules[0][0] = task
        for m_id in range(1, self.nb_machines):
            start_t = schedules[m_id-1][0]["end_time"]
            end_t = start_t + self.data[m_id][0]
            task = {"name": "job_{}".format(
                h_seq[0]+1), "start_time": start_t, "end_time": end_t}
            schedules[m_id][0] = task

        for index, job_id in enumerate(h_seq[1::]):
            start_t = schedules[0][index]["end_time"]
            end_t = start_t + self.data[0][job_id]
            task = {"name": "job_{}".format(
                job_id+1), "start_time": start_t, "end_time": end_t}
            schedules[0][index+1] = task
            for m_id in range(1, self.nb_machines):
                start_t = max(schedules[m_id][index]["end_time"],
                              schedules[m_id-1][index+1]["end_time"])
                end_t = start_t + self.data[m_id][job_id]
                task = {"name": "job_{}".format(
                    job_id+1), "start_time": start_t, "end_time": end_t}
                schedules[m_id][index+1] = task
        opt_makespan = int(schedules[self.nb_machines-1][-1]["end_time"])
        t_t = e - s
        return h_seq, schedules, opt_makespan, t_t

    def _get_makespan(self, seq, data):
        c = np.zeros((self.nb_machines, len(seq)), dtype=object)
        c[0][0] = (0, data[0][seq[0]])
        for m_id in range(1, self.nb_machines):
            s_t = c[m_id-1][0][1]
            e_t = s_t + data[m_id][0]
            c[m_id][0] = (s_t, e_t)
        if len(seq) > 1:
            for i, job_id in enumerate(seq[1::]):
                s_t = c[0][i][1]
                e_t = s_t + data[0][job_id]
                c[0][i+1] = (s_t, e_t)
                for m_id in range(1, self.nb_machines):
                    s_t = max(c[m_id][i][1], c[m_id-1][i+1][1])
                    e_t = s_t + data[m_id][job_id]
                    c[m_id][i+1] = (s_t, e_t)
        return c[self.nb_machines-1][-1][1]

    def neh_heuristic(self):
        sums = []
        default_timer = None
        if sys.platform == "win32":
            default_timer = time.clock
        else:
            default_timer = time.time
        s = default_timer.__call__()

        for job_id in range(self.nb_jobs):
            p_ij = sum([self.data[j][job_id]
                        for j in range(self.nb_machines)])
            sums.append((job_id, p_ij))
        sums.sort(key=lambda x: x[1], reverse=True)
        seq = []
        for job in sums:
            cands = []
            for i in range(0, len(seq) + 1):
                cand = seq[:i] + [job[0]] + seq[i:]
                cands.append((cand, self._get_makespan(cand, self.data)))
            seq = min(cands, key=lambda x: x[1])[0]

        e = default_timer.__call__()

        schedules = np.zeros((self.nb_machines, self.nb_jobs), dtype=dict)
        # schedule first job alone first
        task = {"name": "job_{}".format(
            seq[0]+1), "start_time": 0, "end_time": self.data[0][seq[0]]}
        schedules[0][0] = task
        for m_id in range(1, self.nb_machines):
            start_t = schedules[m_id-1][0]["end_time"]
            end_t = start_t + self.data[m_id][0]
            task = {"name": "job_{}".format(
                seq[0]+1), "start_time": start_t, "end_time": end_t}
            schedules[m_id][0] = task

        for index, job_id in enumerate(seq[1::]):
            start_t = schedules[0][index]["end_time"]
            end_t = start_t + self.data[0][job_id]
            task = {"name": "job_{}".format(
                job_id+1), "start_time": start_t, "end_time": end_t}
            schedules[0][index+1] = task
            for m_id in range(1, self.nb_machines):
                start_t = max(schedules[m_id][index]["end_time"],
                              schedules[m_id-1][index+1]["end_time"])
                end_t = start_t + self.data[m_id][job_id]
                task = {"name": "job_{}".format(
                    job_id+1), "start_time": start_t, "end_time": end_t}
                schedules[m_id][index+1] = task
        max_mkspn = int(schedules[self.nb_machines-1][-1]["end_time"])
        
        t_t = e - s
        return seq, schedules, max_mkspn, t_t

    @lru_cache(maxsize=128)
    def brute_force_exact(self):
        default_timer = None
        if sys.platform == "win32":
            default_timer = time.clock
        else:
            default_timer = time.time
        s = default_timer.__call__()

        jobs_perm = permutations(range(self.nb_jobs))
        seq = min(jobs_perm, key=lambda x: self._get_makespan(x, self.data))
        
        e = default_timer.__call__()
        
        schedules = np.zeros((self.nb_machines, self.nb_jobs), dtype=dict)
        # schedule first job alone first
        task = {"name": "job_{}".format(
            seq[0]+1), "start_time": 0, "end_time": self.data[0][seq[0]]}
        schedules[0][0] = task
        for m_id in range(1, self.nb_machines):
            start_t = schedules[m_id-1][0]["end_time"]
            end_t = start_t + self.data[m_id][0]
            task = {"name": "job_{}".format(
                seq[0]+1), "start_time": start_t, "end_time": end_t}
            schedules[m_id][0] = task

        for index, job_id in enumerate(seq[1::]):
            start_t = schedules[0][index]["end_time"]
            end_t = start_t + self.data[0][job_id]
            task = {"name": "job_{}".format(
                job_id+1), "start_time": start_t, "end_time": end_t}
            schedules[0][index+1] = task
            for m_id in range(1, self.nb_machines):
                start_t = max(schedules[m_id][index]["end_time"],
                              schedules[m_id-1][index+1]["end_time"])
                end_t = start_t + self.data[m_id][job_id]
                task = {"name": "job_{}".format(
                    job_id+1), "start_time": start_t, "end_time": end_t}
                schedules[m_id][index+1] = task
        makespan = int(schedules[self.nb_machines-1][-1]["end_time"])
        t_t = e -s
        return seq, schedules, makespan, t_t

    def genetic_algorithm(self, population_number, it_number=5000, p_crossover=1.0, p_mutation=1.0, nograph=False):
        if population_number is None:
            population_number = self.nb_jobs**2
            
        default_timer = None
        if sys.platform == "win32":
            default_timer = time.clock
        else:
            default_timer = time.time
        s = default_timer.__call__()

        optimal = [4534, 920, 1302]
        opt = 0
        no_of_jobs, no_of_machines = self.nb_jobs, self.nb_machines
        processing_time = []
        for i in range(no_of_jobs):
            temp = []
            for j in range(no_of_machines):
                temp.append(self.data[j][i])
            processing_time.append(temp)
        # generate an initial population proportional to no_of_jobs
        number_of_population = population_number
        no_of_iterations = it_number
        p_crossover = p_crossover
        p_mutation = p_mutation

        # Initialize population
        population = initialize_population(
            number_of_population, no_of_jobs)

        for evaluation in range(no_of_iterations):
            # Select parents
            parent_list = select_parent(
                population, processing_time, no_of_jobs, no_of_machines)
            childs = []

            # Apply crossover to generate children
            for parents in parent_list:
                r = np.random.rand()
                if r < p_crossover:
                    childs.append(crossover(parents))
                else:
                    if r < 0.5:
                        childs.append(parents[0])
                    else:
                        childs.append(parents[1])

            # Apply mutation operation to change the order of the n-jobs
            mutated_childs = []
            for child in childs:
                r = np.random.rand()
                if r < p_mutation:
                    mutated_child = mutation(child)
                    mutated_childs.append(mutated_child)

            childs.extend(mutated_childs)
            if len(childs) > 0:
                update_population(
                    population, childs, processing_time, no_of_jobs, no_of_machines)

        costed_population = []
        for individual in population:
            ind_makespan = (calc_makespan(
                individual, processing_time, no_of_jobs, no_of_machines), individual)
            costed_population.append(ind_makespan)
        costed_population.sort(key=lambda x: x[0])

        seq = list(map(int, costed_population[0][1]))
        makespan = self._get_makespan(seq, self.data)
        e = default_timer.__call__()
        if nograph:
            t_t = e - s
            return seq, None, makespan, t_t  

        schedules = np.zeros((self.nb_machines, self.nb_jobs), dtype=dict)
        # schedule first job alone first
        task = {"name": "job_{}".format(
                seq[0]+1), "start_time": 0, "end_time": self.data[0][seq[0]]}
        schedules[0][0] = task
        for m_id in range(1, self.nb_machines):
            start_t = schedules[m_id-1][0]["end_time"]
            end_t = start_t + self.data[m_id][0]
            task = {"name": "job_{}".format(
                    seq[0]+1), "start_time": start_t, "end_time": end_t}
            schedules[m_id][0] = task

        for index, job_id in enumerate(seq[1::]):
            start_t = schedules[0][index]["end_time"]
            end_t = start_t + self.data[0][job_id]
            task = {"name": "job_{}".format(
                    job_id+1), "start_time": start_t, "end_time": end_t}
            schedules[0][index+1] = task
            for m_id in range(1, self.nb_machines):
                start_t = max(schedules[m_id][index]["end_time"],
                              schedules[m_id-1][index+1]["end_time"])
                end_t = start_t + self.data[m_id][job_id]
                task = {"name": "job_{}".format(
                        job_id+1), "start_time": start_t, "end_time": end_t}
                schedules[m_id][index+1] = task
        t_t = e - s
        return seq, schedules, makespan, t_t

    def swapTwoJobs(self,seq,pos1,pos2):
        seq[pos1], seq[pos2] = seq[pos2], seq[pos1]
        return seq

    def simulated_annealing(self,Ti = 790,Tf = 3 ,alpha = 0.93):
        #Number of jobs given
        n = self.nb_jobs
        default_timer = None
        if sys.platform == "win32":
            default_timer = time.clock
        else:
            default_timer = time.time
        s = default_timer.__call__()
        #Initialize the primary seq
        old_seq,schedules,old_makeSpan, _ = self.palmer_heuristic()
        new_seq = []       
        delta_mk1 = 0
        #Initialize the temperature
        T = Ti
        Tf = Tf
        alpha = alpha
        # of iterations
        temp_cycle = 0
        while T >= Tf  :
            new_seq = old_seq.copy()
            job = new_seq.pop(randint(0,n-1))
            new_seq.insert(randint(0,n-1),job)
            new_make_span = self._get_makespan(new_seq,self.data)
            delta_mk1 = new_make_span - old_makeSpan
            if delta_mk1 <= 0:
                old_seq = new_seq
                old_makeSpan = new_make_span
            else :
                Aprob = np.exp(-(delta_mk1/T))
                if Aprob > np.random.uniform(0.5,0.9):
                    old_seq = new_seq
                    old_makeSpan = new_make_span
                else :
                    #The solution is discarded
                    pass
            T = T * alpha 
            temp_cycle += 1

        e = default_timer.__call__()
        #Result Sequence
        seq = old_seq
        schedules = np.zeros((self.nb_machines, self.nb_jobs), dtype=dict)
        # schedule first job alone first
        task = {"name": "job_{}".format(
            seq[0] + 1), "start_time": 0, "end_time": self.data[0][seq[0]]}
        schedules[0][0] = task
        for m_id in range(1, self.nb_machines):
            start_t = schedules[m_id - 1][0]["end_time"]
            end_t = start_t + self.data[m_id][0]
            task = {"name": "job_{}".format(
                seq[0] + 1), "start_time": start_t, "end_time": end_t}
            schedules[m_id][0] = task

        for index, job_id in enumerate(seq[1::]):
            start_t = schedules[0][index]["end_time"]
            end_t = start_t + self.data[0][job_id]
            task = {"name": "job_{}".format(
                job_id + 1), "start_time": start_t, "end_time": end_t}
            schedules[0][index + 1] = task
            for m_id in range(1, self.nb_machines):
                start_t = max(schedules[m_id][index]["end_time"],
                              schedules[m_id - 1][index + 1]["end_time"])
                end_t = start_t + self.data[m_id][job_id]
                task = {"name": "job_{}".format(
                    job_id + 1), "start_time": start_t, "end_time": end_t}
                schedules[m_id][index + 1] = task
        t_t = e - s
        return seq, schedules, old_makeSpan, t_t

class RandomFlowshop:
    """This module makes an instance of random flowshop problem,
     given number of machines and number of jobs,
     and generates random data of jobs processing times

    Returns:
        RandomFlowshop object -- A RandomFlowshop object instance
    """

    def __init__(self, nb_machines, nb_jobs):
        self.nb_machines = nb_machines
        self.nb_jobs = nb_jobs
        self.data = self.get_random_p_times(10)

    def get_random_p_times(self, p_times_ub):
        """
        Generates matrix of random processing times of jobs in machines

        Attributes:
            p_times_ub (int): upper bound of processing times of each job

        Returns:
            ndarray of random processing times of size (nb_machines x nb_jobs)
        """
        return np.random.randint(
            1,
            p_times_ub,
            size=(
                self.nb_machines,
                self.nb_jobs
            )
        )

    def get_data(self):
        """
        Getter for data attribute

        Returns:
            ndarray: the return value of random processing times matrix
        """
        return self.data

    def get_number_machines(self):
        """
        number of machines getter

        Returns:
            int: Returns number of machines specified in instance problem
        """
        return self.nb_machines

    def get_number_jobs(self):
        """
        Number of jobs getter

        Returns:
            int: returns the number of jobs in instance problem
        """
        return self.nb_jobs

    def get_problem_instance(self):
        """
        Returns a Flowshop instance from randomly generated problem
        Returns:
            Flowshop object: A Flowshop object with the randomly generated data
        """
        return Flowshop(self.data, self.nb_machines, self.nb_jobs)


if __name__ == "__main__":
    random_problem = RandomFlowshop(20, 100)
    random_problem_instance = random_problem.get_problem_instance()
    #seq = random_problem_instance.cds()
    #b_seq, b_scheds, b_opt_makespan = random_problem_instance.brute_force_exact()
    ga_seq, ga_scheds, ga_makespan, t_t  = random_problem_instance.genetic_algorithm(population_number=None, it_number=500, p_crossover=1.0, p_mutation=1.0)
    print("ga_mkspan: {}, t_t: {}".format(ga_makespan, t_t))

