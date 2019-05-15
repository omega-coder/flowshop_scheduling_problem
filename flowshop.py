#!/usr/bin/env python

import numpy as np


class Flowshop(object):

    def __init__(self, data, nb_machines, nb_jobs):
        self.data = data
        self.nb_machines = nb_machines
        self.nb_jobs = nb_jobs

    def solve_johnson(self):
        if self.nb_machines != 2:
            raise Exception("Can not run johnson algorithm on a Problem having more than 2 machines")
        # Build optimal sequence array
        machine_1_sequence = [j for j in range(self.nb_jobs) if self.data[0][j] <= self.data[1][j]]
        machine_1_sequence.sort(key = lambda x: self.data[0][x])
        machine_2_sequence = [j for j in range(self.nb_jobs) if self.data[0][j] > self.data[1][j]]
        machine_2_sequence.sort(key=lambda x: self.data[1][x], reverse=True)

        seq = machine_1_sequence + machine_2_sequence
        jobs_m1 = []
        jobs_m2 = []
        job_name_rad = "job_"
        job = {"name": job_name_rad+str(seq[0]+1), "start_time": 0, "end_time": self.data[0][seq[0]]}
        jobs_m1.append(job)
        job = {"name": job_name_rad+str(seq[0]+1), "start_time": self.data[0][seq[0]], "end_time": self.data[0][seq[0]] + self.data[1][seq[0]]}
        jobs_m2.append(job)


        for job_id in seq[1::]:
            # job on machine 1
            job_name = job_name_rad + str(job_id + 1)
            job_start_m1 = jobs_m1[-1]["end_time"]
            job_end_m1 = job_start_m1 + self.data[0][job_id]
            job = {"name": job_name, "start_time": job_start_m1, "end_time": job_end_m1}
            jobs_m1.append(job)

            # job on machine 2
            job_start_m2 = max(job_end_m1, jobs_m2[-1]["end_time"])
            job_end_m2 = job_start_m2 + self.data[1][job_id]

            job = {"name": job_name, "start_time": job_start_m2, "end_time": job_end_m2}
            jobs_m2.append(job)

        return seq, jobs_m1, jobs_m2
        #return machine_1_sequence + machine_2_sequence

if __name__ == "__main__":
    nb_m = 2
    nb_j = 6
    data = [[6, 2, 10, 4, 5, 3], [5, 4, 3, 8, 2, 4]]
    data = np.array(data)
    pfsp = Flowshop(data, nb_m, nb_j)
    print(pfsp.solve_johnson())

