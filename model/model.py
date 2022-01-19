
import csv
import time

import matplotlib.pyplot as plt

import features
import knn


FNAME_IN_PREFIX = "raw"
FNAME_OUT_PREFIX = "model"
TNUM = 5

GESTURES = ["LP", "TA", "TLFO", "TIFO", "TLFE", "TIFE", "IMFE", "LFE", "IFE", "TE", "WF", "WE", "FS", "FP", "HO", "HC", "HR"]
feature_sets = []

def load_feature_sets_from_file():
    fsets = []
    try:
        for g in GESTURES:
            fsets.append(list(csv.reader(open("{}/model_{}.csv".format(FNAME_OUT_PREFIX, g.lower())))))
    except Exception as e:
        print("Error loading {}: {}".format(g, e))
        return None

    return fsets

def get_feature_sets(force_remodel=False):
    print("Loading feature sets")
    
    if not force_remodel:
        fsets = load_feature_sets_from_file()

        if fsets:
            print("Loaded feature sets from file successfully")
            return fsets

    fsets = []

    for g in GESTURES:
        pnum = 1 # participant
        print("Modelling {}...".format(g))

        f = []
        for tnum in range(TNUM):
            f += features.get_features(features.import_data("{}/p{}_{}_t{}.csv".format(FNAME_IN_PREFIX, pnum, g.lower(), tnum + 1)), g)

        features.write_features_to_csv("{}/model_{}.csv".format(FNAME_OUT_PREFIX, g.lower(), tnum + 1), f)
        fsets.append(f)

    return load_feature_sets_from_file()

def load_test_points(test_gesture):
    return list(csv.reader(open("{}/model_{}_test.csv".format(FNAME_OUT_PREFIX, test_gesture.lower()))))

# 9s to model everything and write to csv

# response time should be under 300ms

feature_sets = get_feature_sets(force_remodel=False)

# TODO: maybe implement weighting
K = 3

knn_data = []

for i in range(len(GESTURES)):
    knn_data = knn.add_training_data(knn_data, feature_sets[i], GESTURES[i])

# TEMP
test_gesture = "IFE"

test_points = load_test_points(test_gesture)

start = time.perf_counter()

classification_results = dict()
for g in GESTURES:
    classification_results[g] = 0

for p in test_points:
    k_nearest = knn.get_k_nearest(knn_data, p, K)
    for c in k_nearest:
        classification_results[c[0]] += 1

    label = knn.calc_mode(k_nearest)

end = time.perf_counter()

total_num = K * len(test_points)

duration_ms = (end - start) * 1000

print("Duration: {}ms for {} test points, {}ms per point".format(duration_ms, len(test_points), duration_ms/len(test_points)))

plt.title("Classification of {} with K-NN (K={})".format(test_gesture, K))
plt.xticks(fontsize=7)
plt.bar(classification_results.keys(), [x/total_num for x in classification_results.values()])
plt.savefig("img/{}_k{}.png".format(test_gesture, K))
plt.show()
