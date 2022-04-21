# We will use the first 6 as training data and the final one as test data

import csv

from dataclasses import dataclass, field

import numpy as np

#DIM_NUM = 16
FEATURES_PER_INPUT = 4

@dataclass
class FeatureSet:
    
    label: str = ""
    mav: float = 0.0
    zcrossings: int = 0
    sschanges: int = 0
    waveformlen: float = 0.0


def sets_to_csv(sets):
    out = []
    for s in sets:
        st = []
        for f in s:
            st += [f.mav, f.zcrossings, f.sschanges, f.waveformlen]

        out.append(st)

    return out

def csv_to_sets(csv, dim_num):
    out = []
    for row in csv:
        s = []
        for i in range(0, dim_num * FEATURES_PER_INPUT, FEATURES_PER_INPUT):
            s.append(FeatureSet(mav=float(csv[row][i]), \
                                zcrossings=int(csv[row][i+1]), \
                                sschanges=int(csv[row][i+2]), \
                                waveformlen=float(csv[row][i+3])))

    return out


def import_data(fname):
    d = list(csv.reader(open(fname, encoding="utf-8-sig")))

    # TODO: maybe filter here

    return d

def get_features(raw_data, label, dim_num):
    features = []

    data = np.array(raw_data).astype(float)

    SAMPLE_DURATION_S = 30
    SAMPLE_COUNT = len(data)
    SAMPLE_RATE = int(SAMPLE_COUNT / SAMPLE_DURATION_S)
    print("Loading {} at {}Hz".format(label, SAMPLE_RATE))

    WINDOW_LEN_SAMPLES = 256
    WINDOW_OVERLAP_SAMPLES = 32

    DEADZONE_V = 1e-8

    win_start = 0
    win_end = win_start + WINDOW_LEN_SAMPLES

    while win_end < SAMPLE_COUNT:

        #print("Window from {} to {}".format(win_start, win_end))
        feature_set = []

        for d in range(dim_num):
            f = FeatureSet()
            f.label = label
    
            win_data = data[win_start:win_end, d]

            # Compute Mean Absolute Value
            f.mav = np.sum(np.absolute(win_data)) / WINDOW_LEN_SAMPLES

            for i in range(WINDOW_LEN_SAMPLES):
                # Compute zero crossings
                if i > 0:
                    if (win_data[i] < 0 and win_data[i - 1] > 0) or (win_data[i] > 0 and win_data[i - 1] < 0):
                        if abs(win_data[i] - win_data[i - 1]) >= DEADZONE_V:
                            f.zcrossings += 1

                # Compute slope sign changes
                if i > 0 and i < WINDOW_LEN_SAMPLES - 1:
                    if (win_data[i] > win_data[i - 1] and win_data[i] > win_data[i + 1]) or (win_data[i] < win_data[i - 1] and win_data[i] < win_data[i + 1]):
                        if abs(win_data[i] - win_data[i - 1]) >= DEADZONE_V or abs(win_data[i] - win_data[i + 1]) >= DEADZONE_V: 
                            f.sschanges += 1

                # Compute waveform legnth
                if i > 0:
                    if abs(win_data[i] - win_data[i - 1]) >= DEADZONE_V:
                        f.waveformlen += abs(win_data[i] - win_data[i - 1])

            feature_set.append(f)

        features.append(feature_set)

        # Advance the window
        win_start += WINDOW_LEN_SAMPLES - WINDOW_OVERLAP_SAMPLES
        win_end += WINDOW_LEN_SAMPLES - WINDOW_OVERLAP_SAMPLES

    return features

def write_features_to_csv(fname, f):
    csv.writer(open(fname, "w+", newline='')).writerows(sets_to_csv(f))

