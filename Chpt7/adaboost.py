import numpy as np


def load_simple_data():
    data_mat = np.matrix([[1.,  2.1],
                          [2.,  1.1],
                          [1.3, 1.],
                          [1. , 1.],
                          [2. , 1.]])
    class_label = [1.0, 1.0, -1.0, -1.0, 1.0]

    return data_mat, class_label


def stump_classify(data_mat, dim, thresh_val, thresh_ineq):
    ret_array = np.ones((np.shape(data_mat)[0], 1))
    if thresh_ineq == 'lt':
        ret_array[data_mat[:, dim] <= thresh_val] = -1.0
    else:
        ret_array[data_mat[:, dim] > thresh_val] = -1.0

    return ret_array


def build_stump(data, class_label, D):
    data_mat = np.mat(data)
    labels = np.mat(class_label).T
    m, n = np.shape(data_mat)
    num_steps = 10.0
    best_stump = {}
    best_class_est = np.mat(np.zeros((m, 1)))
    min_error = np.inf

    for i in range(n):
        range_min = data_mat[:, i].min()
        range_max = data_mat[:, i].max()
        step_size = (range_max - range_min)/num_steps

        for j in range(-1, int(num_steps) + 1):
            for inq in ['lt', 'gt']:
                thresh_val = range_min + step_size * float(j)
                pred_vals = stump_classify(data_mat, i, thresh_val, inq)
                err_att = np.mat(np.ones((m, 1)))
                err_att[pred_vals == labels] = 0
                weighted_err = D.T * err_att;
                print('split: dim %d, thresh %.2f, thresh ineq %s'
                      ', the weighted err %.3f' % (i, thresh_val, inq, weighted_err ))

                if weighted_err < min_error:
                    min_error = weighted_err
                    best_class_est = pred_vals.copy()
                    best_stump['dim'] = i
                    best_stump['thresh'] = thresh_val
                    best_stump['ineq'] = inq

    return best_stump, min_error, best_class_est
