import argparse
import numpy as np
import pandas as pd


def matrices_setup(rawdata):
    my_data = np.genfromtxt(rawdata, delimiter=',')

    # independent variable matrix
    x = my_data[:, 0:-1].reshape(-1, my_data.shape[1] - 1)
    ones = np.ones([x.shape[0], 1])
    x = np.concatenate([ones, x], 1)

    # dependent variable matrix
    y = my_data[:, -1].reshape(-1, 1)

    # weights
    theta = np.zeros([1, x.shape[1]])

    return x, y, theta


def computecost(x, y, theta):
    sqrerror = np.power(((x @ theta.T) - y), 2)
    return np.sum(sqrerror)


def gradientdescent(filename, alpha, threshold):
    x, y, theta = matrices_setup(filename)

    costvalues = []
    result = []
    j = 0
    while True:
        currentresult = []
        currentresult.append(j)
        j += 1

        cost = computecost(x, y, theta)

        for _ in theta:
            for __ in _:
                currentresult.append(round(__, 4))
        currentresult.append(round(cost, 4))
        costvalues.append(cost)
        result.append(currentresult)

        theta = theta - alpha * np.sum((x @ theta.T - y) * x, axis=0)

        if j > 1:
            if (costvalues[-2] - costvalues[-1]) <= threshold:
                break
    return result


parser = argparse.ArgumentParser(description='A tutorial of argparse!')
parser.add_argument('--data', default='random.csv', type=str, help='Dataset Name')
parser.add_argument('--learningRate', default=0.0001, type=float, help='Learning Rate')
parser.add_argument('--threshold', default=0.0001, type=float, help='Threshold')
args = parser.parse_args()

answer3 = gradientdescent(args.data, args.learningRate, args.threshold)
outputfilename = "answer_" + args.data
pd.DataFrame(answer3).to_csv(outputfilename, header=None, index=None)
print("The output is saved in " + outputfilename)

# Sample Output
# D:\>python linearregr.py --data random.csv --learningRate 0.0001 --threshold 0.0001
# The output is saved in answer_random.csv
