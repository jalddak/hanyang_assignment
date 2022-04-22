import sys
import csv

import gini

def main():
    # 입력 받아오는 부분
    args = sys.argv[1:]
    try:
        train_file = open(args[0], 'r', encoding='utf-8')
    except FileNotFoundError:
        print("*** train 파일이 없습니다. ***\n")
        return 0
    try:
        test_file = open(args[1], 'r', encoding='utf-8')
    except FileNotFoundError:
        print("*** test 파일이 없습니다. ***\n")
        return 0
    result_file = open(args[2], 'w', encoding='utf-8')

    training_rdr = csv.reader(train_file, delimiter='\t')
    test_rdr = csv.reader(test_file, delimiter='\t')

    training_data_set = []
    for line in training_rdr:
        training_data_set.append(line)
    training_attributes = training_data_set.pop(0)
    training_data_dic = []
    for data in training_data_set:
        dic = {}
        for i in range(len(data)):
            dic[training_attributes[i]] = data[i]
        training_data_dic.append(dic)

    test_data_set = []
    for line in test_rdr:
        test_data_set.append(line)
    test_attributes = test_data_set.pop(0)
    test_data_dic = []
    for data in test_data_set:
        dic = {}
        for i in range(len(data)):
            dic[test_attributes[i]] = data[i]
        test_data_dic.append(dic)

    decision_tree = []
    gini.calc(training_data_dic)
    gini.compare(training_data_dic, training_attributes)

    train_file.close()
    test_file.close()
    result_file.close()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()