#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np


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
        # Build optimal sequence array
        machine_1_sequence = [j for j in range(
            self.nb_jobs) if self.data[0][j] <= self.data[1][j]]
        machine_1_sequence.sort(key=lambda x: self.data[0][x])
        machine_2_sequence = [j for j in range(
            self.nb_jobs) if self.data[0][j] > self.data[1][j]]
        machine_2_sequence.sort(key=lambda x: self.data[1][x], reverse=True)

        seq = machine_1_sequence + machine_2_sequence
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
        return seq, [jobs_m1, jobs_m2], optim_makespan

    def branch_and_bound():
        pass

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
        self.data = self.get_random_p_times(100)

    def get_random_p_times(self, p_times_ub):
        """
        Generates matrix of random processing times of jobs in machines

        Attributes:
            p_times_ub (int): upper bound of processing times of each job

        Returns:
            ndarray of random processing times of size (nb_machines x nb_jobs)
        """
        return np.random.randint(
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
    random_problem = RandomFlowshop(2, 6)
    random_problem_instance = random_problem.get_problem_instance()
    seq, jobs = random_problem_instance.solve_johnson()
    print("Sequence: {} \nJobs on Machine 1: \n {} \n Jobs on machine 2:\n {} \n".format(
        seq, jobs[0], jobs[1]))
