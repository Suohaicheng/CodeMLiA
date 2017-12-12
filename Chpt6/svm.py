import numpy as np


def load_data_set(file_name):
    data_mat = []
    label_mat = []
    fr = open(file_name)
    for line in fr.readlines():
        line_attr = line.strip().split('\t')
        data_mat.append([float(line_attr[0]), float(line_attr[1])])
        label_mat.append(float(line_attr[2]))
    fr.close()
    return data_mat, label_mat


def select_j_rand(i, m):
    j = i
    while j == i:
        j = int(np.random.uniform(0, m))
    return j


def clip_alpha(aj, h, l):
    if aj > h:
        aj = h
    if aj < l:
        aj = l
    return aj


def smo_simple(data_in, labels, c, toler, max_iter):
    data_mat = np.mat(data_in)
    label_mat = np.mat(labels).transpose()
    b = 0
    m, n = np.shape(data_mat)
    alphas = np.mat(np.zeros((m, 1)))
    iter = 0
    while iter < max_iter:
        alpha_pairs_changed = 0
        for i in range(m):
            f_xi = float(np.multiply(alphas, label_mat).T *\
                         (data_mat * data_mat[i, :].T)) + b
            e_i = f_xi - float(label_mat[i])
            if label_mat[i] * e_i < -toler and alphas[i] < c or \
                label_mat[i] * e_i > toler and alphas[i] > 0:

                j = select_j_rand(i, m)
                f_xj = float(np.multiply(alphas, label_mat).T *\
                             (data_mat * data_mat[j, :].T)) + b
                e_j = f_xj - float(label_mat[j])

                alpha_i_old = alphas[i].copy()
                alpha_j_old = alphas[j].copy()
                if label_mat[i] != label_mat[j]:
                    l = max(0, alphas[j] - alphas[i])
                    h = min(c, c + alphas[j] - alphas[i])
                else:
                    l = max(0, alphas[j] + alphas[i] - c)
                    h = min(c, alphas[j] + alphas[i])
                if l == h:
                    print('L == H')
                    continue

                eta = 2.0 * data_mat[i, :] * data_mat[j, :].T - \
                    data_mat[i, :] * data_mat[i, :].T - \
                    data_mat[j, :] * data_mat[j, :].T
                if eta >= 0:
                    print('eta >= 0')
                    continue
                alphas[j] -= label_mat[j] * (e_i - e_j) / eta
                alphas[j] = clip_alpha(alphas[j], h, l)

                if abs(alphas[j] - alpha_j_old) < 0.00001:
                    print('j not moving enough')
                    continue
                alphas[i] += label_mat[j] * label_mat[i] * \
                             (alpha_j_old - alphas[j])
                b1 = b - e_i - label_mat[i] * (alphas[i] - alpha_i_old) * \
                    data_mat[i, :] * data_mat[i, :].T - \
                    label_mat[j] * (alphas[j] - alpha_j_old) * \
                    data_mat[i, :] * data_mat[j, :].T
                b2 = b - e_j - label_mat[i] * (alphas[i] - alpha_i_old) * \
                    data_mat[i, :] * data_mat[j, :].T - \
                    label_mat[j] * (alphas[j] - alpha_j_old) * \
                    data_mat[j, :] * data_mat[j, :].T
                if 0 < alphas[i] and c > alphas[i]:
                    b = b1
                elif 0 < alphas[j] and c > alphas[j]:
                    b = b2
                else:
                    b = (b1 + b2) / 2.0
                alpha_pairs_changed += 1
                print('iter %d i :%d, pairs changed %d' %
                      (iter, i, alpha_pairs_changed))

                if alpha_pairs_changed == 0:
                    iter += 1
                else:
                    iter = 0
                print('iteration number: %d' % iter)

    return b, alphas
