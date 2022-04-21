import csv
import math
from statistics import mode

# number of muscles
M = 16

def import_data(fname):
    d = list(csv.reader(open(fname, encoding="utf-8-sig")))

    # TODO: maybe filter here

    return d

# ABS before touching any data

# returns the bessel corrected variance
def get_variances(data, means):
    totals = [0] * M
    for row in data:
        for muscle in range(M):
            totals[muscle] += (abs(float(row[muscle])) - means[muscle]) ** 2

    for muscle in range(M):
        totals[muscle] /= len(data) - 1

    return totals

def get_means(data):
    means = [0] * M
    for row in data:
        for muscle in range(M):
            means[muscle] += abs(float(row[muscle]))

    for muscle in range(M):
        means[muscle] /= len(data)
    return means

# This is what would be implemented on the device
def get_gaussian_probs(value, mean, variance):
    return (1/(math.sqrt(2 * math.pi * variance)) * math.exp(-1 * (((value - mean) ** 2)/(2 * variance))))



# START remote training
data = [import_data("model/open.csv"), 
        import_data("model/close.csv"), 
        import_data("model/rest.csv"), 
        import_data("model/index.csv"),
        import_data("model/thumb_index.csv")]

GESTURES = ["OPEN", "CLOSE", "REST", "INDEX", "THUMB_INDEX"]

G = len(data)
means = [[0] * M for i in range(G)]
variances = [[0] * M for i in range(G)]
priors = [0] * G
total_point_num = 0

for i in range(G):
    means[i] = get_means(data[i])
    variances[i] = get_variances(data[i], means[i])

    priors[i] = len(data[i])
    total_point_num += len(data[i])

for i in range(G):
    priors[i] /= total_point_num

# END remote

test_points = import_data("model/testdata.csv")
num_correct = 0

for p in test_points:
    # probability of this point being gesture i
    P = [0] * G
    for i in range(G):
        # remember to offset test_points by 1
        pxy = 1
        normalizers = [0] * M

        for muscle in range(M):
            pxy *= get_gaussian_probs(abs(float(p[muscle + 1])), means[i][muscle], variances[i][muscle])

            # TODO:  normalize
            #for yprime in range(G):
                #normalizers[muscle] = sum_of_all_probs_for_class1 * priors[1]

        P[i] = priors[i] * pxy / 1 #sum(normalizers)


    predicted = GESTURES[P.index(max(P))]
    print(P)
    print("{}/{}\n".format(predicted, p[0]))
    num_correct += 1 if (predicted == p[0]) else 0

print("{}/{}: {}%".format(num_correct, len(test_points), num_correct/len(test_points) * 100))

