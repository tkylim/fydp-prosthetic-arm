import csv
import math

fname_model = "model.csv"
fname_test = "testdata.csv"

MotionName = ['Lateral Prehension','Thumb Adduction','Thumb and Little Finger Opposition',
    'Thumb and Index Finger Opposition','Thumb and Index Finger Extension','Thumb and Little Finger Extension',
    'Index and Middle Finger Extension','Little Finger Extension','Index Finger Extension',
    'Thumb Finger Extension','Wrist Flexion','Wrist Extension',
    'Forearm Supination','Forearm Pronation','Hand Open',
    'Hand Close', 'Rest']

# Number of muscles
n = 16
g = 17

# Take the provided data point and find the distance between it and each gesture on the n-dimensional plane. The nearest neighbour will be the output
with open(fname_model) as model:
    modeldata = list(csv.reader(model))
    with open(fname_test, encoding="utf-8-sig") as test:
        testdata = list(csv.reader(test))

        # pythons do [row][column]
        nn = [0] * g
        for i in range(0, len(testdata)): # loop through the test data
            for j in range(0, g): # loop through the gestures
                s = 0
                for k in range(0, n): # loop through the muscles
                    #print("Evaluating: (%s - %s/%s)^2" % (modeldata[j][k], testdata[i][0], testdata[i][k]))
                    x = (float(modeldata[j][k])-(float(testdata[i][0])/float(testdata[i][k])))**2
                    print("Test %d:  %s muscle %d:  %f" %(i, MotionName[j], k+1, x))
                    s += x

                nn[j] = math.sqrt(s)

            # Find the min
            gesture = 0
            least = 999

            for m in range(0, g):
                if nn[m] < least and (m == 14 or m == 15):
                    least = nn[m]
                    gesture = m
            #print("Open: %f Close: %f" %(nn[14], nn[15]))
            print(nn)
            print("Gesture: %s" % (MotionName[gesture]))
