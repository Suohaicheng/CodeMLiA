import kNN


"""
test_data_set = [[0, 0], [1, 1], [2, 3], [5,-1]]

data_set, labels = kNN.create_simple_data_set()

for i in range(len(test_data_set)):
    t = kNN.classify0(test_data_set[i], data_set, labels, 3)
    print(t) 
"""

d, l = kNN.file2matrix('datingTestSet2.txt')
print(d)
print(l)