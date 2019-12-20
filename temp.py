import random
import pandas
import sys

thetha = 0.5


def learning():
    # Or gate Learning

    # in and gate you can increase decimal points as much as you want
    # to round the out to 1 decimal point
    weight1 = round(random.uniform(-0.5, 0.5), 2)
    # to round the out to 1 decimal point
    weight2 = round(random.uniform(-0.5, 0.5), 2)
    alpha = 0.1
    flage = True     # flage will be false when convergence reaches
    output = 1
    andGate = ((0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 1))
    while(flage == True):
        flage = False
        for examp in andGate:
            value = (examp[0]*weight1)+(examp[1]*weight2)
            if(value >= thetha):  # if perceptoron excited
                output = 1
            else:
                output = 0
            if(output != examp[2]):
                # to round the out to 1 decimal point
                weight1 = round(weight1+(alpha*(examp[2]-output)*examp[0]), 2)
                # to round the out to 1 decimal point
                weight2 = round(weight2+(alpha*(examp[2]-output)*examp[1]), 2)
                flage = True

    weightsFile = open('weights.txt', 'w')
    weightsFile.write('{0}\n'.format(weight1))
    weightsFile.write('{0}\n'.format(weight2))
    weightsFile.close()


def testing():
    test = pandas.read_csv("test.csv")
    listt = test.values.tolist()

    weights = []
    weightsFile = open('weights.txt', 'r')
    for w in weightsFile.readlines():
        weights.append(float(w))

    weight1 = weights[0]
    weight2 = weights[1]

    columns = ['x', 'y', 'actual', 'predicted']
    out = pandas.DataFrame(columns=columns)
    for i in range(0, len(listt)):
        values = []
        for j in range(0, len(test.columns)):
            values.append(test.iloc[i, j])

        desired = weight1*listt[i][0] + weight2*listt[i][1]
        if desired < thetha:
            desired = 0
        else:
            desired = 1
        values.append(desired)
        out.loc[len(out)] = values

    out.to_csv("results.csv", index=False)


if __name__ == '__main__':
    if '-learning' in sys.argv:
        learning()
    if '-testing' in sys.argv:
        testing()
