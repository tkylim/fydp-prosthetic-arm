import csv

import matplotlib.pyplot as plt

import features
import knn

FNAME_IN_PREFIX = "../data/2022-03-03"
FNAME_OUT_PREFIX = "model/2022-03-03"

GESTURES = ["HR", "HO", "HC", "IMFE"]
DIM_NUM = 6
K = 3

fsets = []

print("Generating model...")
for g in GESTURES:
    # Generate features
    f = features.get_features(features.import_data("{}/{}.csv".format(FNAME_IN_PREFIX, g.lower())), g.lower(), DIM_NUM)
    
    # Save model
    features.write_features_to_csv("{}/model_{}_{}.csv".format(FNAME_OUT_PREFIX, g.lower(), 1), f)

    fsets.append(f)

correct = [0] * len(GESTURES)
total = [0] * len(GESTURES)

print("Starting performance check...")
for i in range(len(fsets)):
    knn_data = []
    for j in range(len(fsets)):
        if j != i:
            knn.add_training_data(knn_data, fsets[j], GESTURES[j])

    knn_data_others = knn_data.copy()

    total[i] = len(fsets[i])

    # Add all except 1 point
    for j in range(len(fsets[i])):
        # Reset
        knn_data = knn_data_others.copy()

        # Add all points except j
        knn.add_training_data(knn_data, fsets[i], GESTURES[i], ignore_index=j)

        # Classify j
        if knn.calc_mode(knn.get_k_nearest(knn_data, features.sets_to_csv([fsets[i][j]])[0], K)) == GESTURES[i]:
            correct[i] += 1


print("Accuracy (K={}):\n{}\n{}%".format(K, GESTURES, ["{}%".format(int(correct[i] * 100 / (total[i]))) for i in range(len(correct))]))
