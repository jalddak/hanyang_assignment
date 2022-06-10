from surprise import SVD
from surprise import Dataset
from surprise import Reader
import sys
import numpy as np
import pandas as pd

def main():
    # 입력 받아오는 부분
    args = sys.argv[1:]
    base_name = args[0]
    test_name = args[1]

    base_data = pd.read_csv("./data/" + base_name, delimiter='\t', names=['user_id', 'item_id', 'rating', 'time_stamp'])
    test_data = pd.read_csv("./data/" + test_name, delimiter='\t', names=['user_id', 'item_id', 'rating', 'time_stamp'])
    del base_data['time_stamp']
    del test_data['time_stamp']

    reader = Reader(line_format='user item rating')
    base_dataset = Dataset.load_from_df(base_data[['user_id', 'item_id', 'rating']], reader)

    trainset = base_dataset.build_full_trainset()
    testset = list(test_data.values)
    for i in range(len(testset)):
        testset[i] = tuple(testset[i])

    algo = SVD(n_factors=50, random_state=0)
    algo.fit(trainset)
    predictions = algo.test(testset)

    output = open('./data/' + base_name + "_prediction.txt", 'w', encoding='utf-8')
    for p in predictions:
        output.write(str(p.uid) + '\t')
        output.write(str(p.iid) + '\t')
        output.write(str(round(p.est)) + '\n')
    output.close()


if __name__ == '__main__':
    main()