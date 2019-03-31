import numpy as np
import operator


def create_simple_data_set():
    data_set = [[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]]
    labels = ['A', 'A', 'B', 'B']
    return data_set, labels


def file2matrix(file_name):
    fr = open(file_name)
    # Read all lines, and return it's content as a list 
    # containing all lines
    lines = fr.readlines()
    num_lines = len(lines)

    fr.close()
    # Build a num_lines-row，3-column matrix filling with 0
    data = np.zeros((num_lines, 3))
    labels = []
    index = 0
    for l in lines:
        li = l.strip()
        line_list = li.split('\t')
        # Matrix's assignment: a 4-element list as a value
        # is assigned to one row of matrix.
        data[index, :] = line_list[0:3]
        labels.append(int(line_list[-1]))
        index += 1

    return data, labels



def classify0(in_x, data_set, labels, k):
    data_set_size = np.shape(data_set)[0]
    diff_mat = np.tile(in_x, (data_set_size, 1)) - data_set
    sq_diff_mat = diff_mat ** 2;
    sq_distances = sq_diff_mat.sum(axis=1)
    distances = sq_distances ** 0.5
    sorted_dist_index = distances.argsort()

    class_cnt = {}

    for i in range(k):
        vote_labels = labels[sorted_dist_index[i]]
        class_cnt[vote_labels] = class_cnt.get(vote_labels, 0) + 1

    sorted_class = sorted(class_cnt.items(), key=operator.itemgetter(1), reverse=True)

    return sorted_class[0][0]
