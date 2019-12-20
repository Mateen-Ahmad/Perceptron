import random
import pandas as pd
import sys


threshold = 0.2


def learning():
    data = pd.read_csv(sys.argv[2])
    #data = data.iloc[:100]
    dataList = data.values.tolist()
    numberOfColumns = len(data.columns)
    numberOfRows = len(dataList)
    #print("Rows: ", numberOfRows)
    learningRate = 0.1
    flage = True
    weights = []
    for i in range(0, numberOfColumns-1):
        weights.append(round(random.uniform(-0.5, 0.5), 3))

    numberOfInputs = len(weights)
    while(flage == True):
        flage = False
        for row in dataList:
            value = 0
            for i in range(0, numberOfInputs):
                value = value + (row[i]*weights[i])
            if(value >= threshold):  # if perceptoron excited
                output = 1
            else:
                output = 0

            index = numberOfColumns-1
            if(output != row[index]):
                for i in range(0, numberOfColumns-1):
                    weights[i] = round(
                        weights[i]+(learningRate*(row[index]-output)*row[i]), 3)
                # to round the out to 1 decimal point
                # print(weights)
                flage = True

    # print(weights)

    weightsFile = open('weights.txt', 'w')
    for i in range(0, numberOfInputs):
        weightsFile.write('{0}\n'.format(weights[i]))
    weightsFile.close()


def testing():
    data = pd.read_csv(sys.argv[2])
    #data = data.iloc[300:]
    dataList = data.values.tolist()
    numberOfColumns = len(data.columns)
    numberOfRows = len(dataList)
    #print("Rows: ", numberOfRows)
    weights = []
    weightsFile = open('weights.txt', 'r')
    for w in weightsFile.readlines():
        weights.append(float(w))

    numberOfInputs = len(weights)
    columns = []
    for i in range(0, numberOfInputs):
        columns.append('x'+str(i+1))
    columns.append('Actual')
    columns.append('Predicted')
    out = pd.DataFrame(columns=columns)
    for i in range(0, numberOfRows):
        values = []
        for j in range(0, numberOfColumns):
            values.append(data.iloc[i, j])
        actual = 0
        for k in range(0, numberOfInputs):
            actual += weights[k]*dataList[i][k]
        if actual < threshold:
            actual = 0
        else:
            actual = 1
        values.append(actual)
        out.loc[len(out)] = values

    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for i in range(0, numberOfRows):
        if (out.iloc[i, -1] == out.iloc[i, -2]) & (out.iloc[i, -1] == 1):
            tp += 1
        elif(out.iloc[i, -1] == out.iloc[i, -2]) & (out.iloc[i, -1] == 0):
            tn += 1
        elif(out.iloc[i, -1] != out.iloc[i, -2]) & (out.iloc[i, -1] == 1):
            fp += 1
        elif(out.iloc[i, -1] != out.iloc[i, -2]) & (out.iloc[i, -1] == 0):
            fn += 1

    precision = tp/(tp+fp)*100
    recall = tp/(tp+fn)*100
    accuracy = (tp+tn)/(tp+tn+fp+fn)*100
    results = open('Accuracy.txt', 'w')
    results.write("Precision: ")
    results.write('%d\n' % precision)
    results.write("Recall: ")
    results.write('%d\n' % recall)
    results.write("Accuracy: ")
    results.write('%d\n' % accuracy)
    results.close()

    print(precision)
    print(recall)
    print(accuracy)
    out.to_csv("results.csv", index=False)


if __name__ == '__main__':

    if '-learning' in sys.argv:
        learning()
    if '-testing' in sys.argv:
        testing()
