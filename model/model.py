
import csv
import time

import matplotlib.pyplot as plt

import features
import knn
import sys


FNAME_IN_PREFIX = "../data/2022-03-22"
FNAME_OUT_PREFIX = "model/2022-03-22"
#TNUM = 5

#GESTURES = [["LP", "TA", "TLFO", "TIFO", "TLFE", "TIFE", "TIFE", "LFE", "IFE", "TE", "WF", "WE", "FS", "FP", "HO", "HC", "HR"]
GESTURES = ["HO", "HR", "HC", "TIFE"]
DIM_NUM = 6
feature_sets = []

def load_feature_sets_from_file():
    fsets = []
    try:
        for g in GESTURES:
            fsets.append(list(csv.reader(open("{}/model_{}_1.csv".format(FNAME_OUT_PREFIX, g.lower())))))
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
    return list(csv.reader(open("{}/model_{}_1.csv".format(FNAME_OUT_PREFIX, test_gesture.lower()))))

# 9s to model everything and write to csv

# response time should be under 300ms

# TODO FIX THIS
#feature_sets = get_feature_sets(force_remodel=False)
def calc_certainty(label, kn):
    return round([p[0] for p in kn].count(label) / len(kn), 2)

feature_sets = []
print("Generating model...")
for g in GESTURES:
    # Generate features
    f = features.get_features(features.import_data("{}/{}.csv".format(FNAME_IN_PREFIX, g.lower())), g.lower(), DIM_NUM)
    
    # Save model
    features.write_features_to_csv("{}/model_{}_{}.csv".format(FNAME_OUT_PREFIX, g.lower(), 1), f)

    feature_sets.append(f)

# TODO: maybe implement weighting
K = 3
bar = False
realtime = True
WINDOW_LEN_MS = int(256 / 1260 * 1000)

knn_data = []

for i in range(len(GESTURES)):
    knn.add_training_data(knn_data, feature_sets[i], GESTURES[i])

#for test_gesture in GESTURES:
test_gesture = "test2"
test_points = load_test_points(test_gesture)

start = time.perf_counter()

classification_results = []
classification_bests = [("TIFE", 1)]
classification_filtered = ["TIFE"]

DELAY_MS = 2400

#for g in GESTURES:
    #classification_results[g] = 0


test_len = 109
if realtime:
    input("Press any key to start real-time classification...")

for i in range(test_len):
    ps = time.perf_counter()
    k_nearest = knn.get_k_nearest(knn_data, test_points[i], K)

    label = knn.calc_mode(k_nearest)
    classification_results.append(k_nearest)

    certainty = calc_certainty(label, k_nearest) 

    if realtime:
        time.sleep((WINDOW_LEN_MS - (time.perf_counter() - ps)) / 1000)

    # filter
    cur_label = classification_filtered[-1]
    accepted = False
    if i >= 2 and certainty >= 1:
        # TIFE is rarely misclassified
        if label == "TIFE":
            accepted = True
        # As gesture recognition is mostly identified throughout the initial transition into the gesture, indentified periods of rest are normal throughout the gesture and can be ignored
        elif label == "HR":
            accepted = False
        # For other gestures, ensure that the previous matches
        elif label != classification_filtered[-1]:
            if (label == classification_bests[-1][0] and classification_bests[-1][1] >= 0.6):
                accepted = True
            else:
                accepted = False
    else:
        accepted = False

    if accepted:
        ms = (i + 1) * WINDOW_LEN_MS
        print("[{}]: {}".format(ms, label))
        classification_filtered.append(label)
    else:
        classification_filtered.append(cur_label)

    classification_bests.append((label, certainty))
    #print("{} ({}%)".format(label, int(certainty * 100)))
    sys.stdout.flush()


end = time.perf_counter()

total_num = len(test_points)

duration_ms = (end - start) * 1000

print("{}ms of input data processed".format(WINDOW_LEN_MS * len(classification_results)))
print("Duration: {}ms for {} test points, {}ms per point".format(duration_ms, len(test_points), duration_ms/len(test_points)))
#print("Accuracy {} (K={}) {}%".format(test_gesture, K, int(classification_results[test_gesture] * 100 / total_num)))

imgname = ""

plt.title("First layer Classifier with K-NN (K={})".format(K))
plt.xticks(fontsize=7)
if bar:
    plt.bar(classification_results.keys(), [x/total_num for x in classification_results.values()])
    imgname = "img/{}_k{}_all.png"
else:
    total_len = WINDOW_LEN_MS * len(test_points)
    mapping = dict([(5, dict([(0.2, 1), (0.4, 4), (0.6, 32), (0.8, 64), (1, 128)])), (3, dict([(0.33, 1), (0.67, 32), (1, 128)]))])
    plt.scatter([x * WINDOW_LEN_MS for x in range(len(classification_bests))], [x[0] for x in classification_bests], s=[mapping[K][x[1]] for x in classification_bests])
    imgname = "img/{}_k{}_bests.png"

plt.savefig("out/first.png")
plt.show()

plt.clf()
plt.title("Second layer Classifier")
plt.xticks(fontsize=7)
plt.scatter([x * WINDOW_LEN_MS for x in range(len(classification_filtered))], classification_filtered)
plt.savefig("out/second.png")
plt.show()

# certaiintty
plt.clf()
plt.title("Certainty")
plt.xticks(fontsize=7)
plt.scatter([x * WINDOW_LEN_MS for x in range(len(classification_bests))], [x[1] for x in classification_bests])
plt.savefig("out/certainty.png")
plt.show()
