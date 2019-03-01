import sys
from KNN import KNN
from NB import NB


def main():
    train_filename = sys.argv[1]
    test_filename = sys.argv[2]
    if sys.argv[3][1] == 'N':
        k_value = int(sys.argv[3][0])
        knn = KNN()
        train_data, label_data = knn.getTrainData(train_filename)
        test_data = knn.getData(test_filename)
        for i in range(len(test_data)):
            neighbors = knn.get_neighbors(train_data, label_data, test_data[i], k_value)
            print(knn.vote(neighbors))

    else:
        nb = NB()
        train_data = nb.getData(train_filename)
        test__data = nb.getData(test_filename)
        result = nb.getPredictions(test__data, train_data)
        # separated = nb.separateByClass(train_data)
        # print(separated[1][separated[1]=='yes'].count())
        # print(result[20][8],result[21][8],result[22][8])
        for i in range(len(result)):
            if result[i][8] == 1:
                print('yes')
            else:
                print('no')


if __name__ == '__main__':
    main()
