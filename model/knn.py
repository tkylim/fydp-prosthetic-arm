import csv
import math
from statistics import mode

import features

def import_data(fname):
    d = list(csv.reader(open(fname, encoding="utf-8-sig")))

    # TODO: maybe filter here

    return d

def calc_distance(p1, p2):
    if len(p1) != len(p2):
        print("ERROR {}!={}".format(len(p1), len(p2)))
        return 0

    s = 0
    for i in range(len(p1)):
        s += (abs(float(p1[i])) - abs(float(p2[i]))) ** 2

    return math.sqrt(s)

def add_to_classification(p, data, best, key):
    for point in data: 
        d = calc_distance(p, point)

        # filter out 0 distance for now as the point existsas the point exists in the dataset and we shouldn't bias based off that
        if d == 0:
            continue

        if d < best[-1][1]:
            best[-1] = (key, d)

            # sort in ascending order
            best.sort(key=lambda x: x[1])

# expecting a 2d array of input data with rows of features
def add_training_data(data, new_data, label, ignore_index=-1):
    for i in range(len(new_data)):
        if ignore_index != i:
            data.append((label, new_data[i]))

def get_k_nearest(sets, p, K):
    best = [("N/A",999)] * K
    data = features.sets_to_csv([s[1] for s in sets])
    for i in range(len(data)):
        d = calc_distance(data[i], p)
        label = sets[i][0]

        if d < best[-1][1]:
            best[-1] = (label, d)

            # sort in ascending order
            best.sort(key=lambda x: x[1])

    return best

def calc_mode(b):
    return mode([x[0] for x in b])

#K = 11

#test_points = import_data("model/testdata.csv")
#
#for p in test_points:
#    # K closest 
#    closest = [("N/A",999)] * K
#    answer = p[0]
#    p = p[1:]
#
#    add_to_classification(p, import_data("model/open.csv"), closest, "OPEN")
#    add_to_classification(p, import_data("model/close.csv"), closest, "CLOSE")
#    add_to_classification(p, import_data("model/rest.csv"), closest, "REST")
#    add_to_classification(p, import_data("model/thumb_index.csv"), closest, "THUMB_INDEX")
#    add_to_classification(p, import_data("model/index.csv"), closest, "INDEX")
#
#    m = calc_mode(closest)
#    
#    print("{}: {}".format(("RIGHT" if (m == answer) else "WRONG"), m))
