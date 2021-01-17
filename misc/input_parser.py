import random
from misc.constants import *


class InputParser():
    def __init__(self, file_name="magic04.data"):
        self.file_name = file_name

    def parse(self):
        all_tuples = []
        f = open(self.file_name, "r")
        line_index = 0
        line = f.readline().rstrip()

        while len(line) != 0:
            splitted = line.split(',')
            all_tuples.append(([float(splitted[i]) for i in range(10)], splitted[10]))
            line = f.readline().rstrip()
            line_index += 1
        f.close()
        random.shuffle(all_tuples)

        return all_tuples

    def get_samples_and_labels(self):
        all_tuples = self.parse()
        samples, labels = [], []
        gamma_count = 0
        for cur_tuple in all_tuples:
            if cur_tuple[1] == 'g' and gamma_count < HADRON_COUNT:
                samples.append(cur_tuple[0])
                labels.append(cur_tuple[1])
                gamma_count += 1
            elif cur_tuple[1] == 'h':
                samples.append(cur_tuple[0])
                labels.append(cur_tuple[1])

        return samples, labels
