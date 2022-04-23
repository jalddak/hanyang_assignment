import sys
import csv
import numpy as np
import pandas as pd

import node_class
import gini
import info
import gain_ratio

def main():
    # 입력 받아오는 부분
    args = sys.argv[1:]
    try:
        train_file = open('./data/' + args[0], 'r', encoding='utf-8')
    except FileNotFoundError:
        print("*** train 파일이 없습니다. ***\n")
        return 0
    try:
        test_file = open('./data/' + args[1], 'r', encoding='utf-8')
    except FileNotFoundError:
        print("*** test 파일이 없습니다. ***\n")
        return 0
    result_file = open('./data/' + args[2], 'w', encoding='utf-8')

    result_file_gini = open('./data/gini_' + args[2], 'w', encoding='utf-8')
    result_file_gain_ratio = open('./data/gain_ratio_' + args[2], 'w', encoding='utf-8')
    result_file_info = open('./data/info_' + args[2], 'w', encoding='utf-8')

    # 트레이닝 데이터 포맷 만드는 과정
    training_rdr = csv.reader(train_file, delimiter='\t')
    test_rdr = csv.reader(test_file, delimiter='\t')

    training_data_set = []
    for line in training_rdr:
        training_data_set.append(line)
    training_attributes = training_data_set.pop(0)
    training_data = {}
    for attribute in training_attributes:
        training_data[attribute] = []
    for data in training_data_set:
        for i in range(len(data)):
            training_data[training_attributes[i]].append(data[i])
    training_df = pd.DataFrame(training_data)

    # gini_index 트리
    dt_gini = node_class.Node()
    dt_gini = dt_gini.make_tree(training_df, None, 'gini')

    # gain_ratio 트리
    dt_gain_ratio = node_class.Node()
    dt_gain_ratio = dt_gain_ratio.make_tree(training_df, None, 'gain_ratio')

    # information 트리
    dt_info = node_class.Node()
    dt_info = dt_info.make_tree(training_df, None, 'info')

    # 테스트 데이터 포맷 만드는 과정
    test_data_set = []
    for line in test_rdr:
        test_data_set.append(line)
    test_attributes = test_data_set.pop(0)
    test_data = {}
    for attribute in test_attributes:
        test_data[attribute] = []
    for data in test_data_set:
        for i in range(len(data)):
            test_data[test_attributes[i]].append(data[i])
    test_df = pd.DataFrame(test_data)
    test_df[training_attributes[len(training_attributes) - 1]] = None

    test_data_set_gini = test_df.values
    test_data_set_gain_ratio = test_df.values
    test_data_set_info = test_df.values
    for i in range(len(training_attributes)):
        if i == len(training_attributes) - 1:
            result_file_gini.write(training_attributes[i])
            result_file_gain_ratio.write(training_attributes[i])
            result_file_info.write(training_attributes[i])
        else:
            result_file_gini.write(training_attributes[i] + '\t')
            result_file_gain_ratio.write(training_attributes[i] + '\t')
            result_file_info.write(training_attributes[i] + '\t')
    result_file_gini.write('\n')
    result_file_gain_ratio.write('\n')
    result_file_info.write('\n')

    # gini_index 트리 결과로 마이닝 하는 과정
    for i in range(len(test_data_set_gini)):
        test_data_set_gini[i] = dt_gini.mining(test_data_set_gini[i], training_attributes, 'gini')
        for j in range(len(training_attributes)):
            if j == len(training_attributes) - 1:
                result_file_gini.write(test_data_set_gini[i][j])
            else:
                result_file_gini.write(test_data_set_gini[i][j] + '\t')
        result_file_gini.write('\n')

    # gain_ratio 트리 결과로 마이닝 하는 과정
    for i in range(len(test_data_set_gain_ratio)):
        test_data_set_gain_ratio[i] = dt_gain_ratio.mining(test_data_set_gain_ratio[i], training_attributes, 'gain_ratio')
        for j in range(len(training_attributes)):
            if j == len(training_attributes) - 1:
                result_file_gain_ratio.write(test_data_set_gain_ratio[i][j])
            else:
                result_file_gain_ratio.write(test_data_set_gain_ratio[i][j] + '\t')
        result_file_gain_ratio.write('\n')

    # info_index 트리 결과로 마이닝 하는 과정
    for i in range(len(test_data_set_info)):
        test_data_set_info[i] = dt_info.mining(test_data_set_info[i], training_attributes, 'info')
        for j in range(len(training_attributes)):
            if j == len(training_attributes) - 1:
                result_file_info.write(test_data_set_info[i][j])
            else:
                result_file_info.write(test_data_set_info[i][j] + '\t')
        result_file_info.write('\n')

    train_file.close()
    test_file.close()
    result_file_gini.close()
    result_file_gain_ratio.close()
    result_file_info.close()
    gini_value = check(args[2], 'gini')
    gain_ratio_value = check(args[2], 'gain_ratio')
    info_value = check(args[2], 'info')
    values = [gini_value, gain_ratio_value, info_value]
    print(values)
    maxindex = np.argmax(list(values))
    if maxindex == 0:
        result_file_gini = open('./data/gini_' + args[2], 'r', encoding='utf-8')
        for line in result_file_gini:
            result_file.write(line)
        result_file_gini.close()
        print("gini_index 방법이 " + str(gini_value) +' / ' + str(len(test_data_set)) + " 의 정확도로 가장 정확했습니다.")
    elif maxindex == 1:
        result_file_gain_ratio = open('./data/gain_ratio_' + args[2], 'r', encoding='utf-8')
        for line in result_file_gain_ratio:
            result_file.write(line)
        result_file_gain_ratio.close()
        print("gini_index 방법이 " + str(gain_ratio_value) + ' / ' + str(len(test_data_set)) + "의 정확도로 가장 정확했습니다.")
    elif maxindex == 2:
        result_file_info = open('./data/info_' + args[2], 'r', encoding='utf-8')
        for line in result_file_info:
            result_file.write(line)
        result_file_info.close()
        print("gini_index 방법이 " + str(info_value) + ' / ' + str(len(test_data_set)) + "의 정확도로 가장 정확했습니다.")
    result_file.close()


def check(result_file_name, measure):
    i = -5
    num = ''
    if not result_file_name[i].isdigit():
        try:
            answer_file = open('./data/dt_answer.txt', 'r', encoding='utf-8')
        except FileNotFoundError:
            print("*** answer 파일이 없습니다. ***\n")
            return 0
        try:
            result_file = open('./data/' + measure + '_dt_result.txt', 'r', encoding='utf-8')
        except FileNotFoundError:
            print("*** " + measure + "_result 파일이 없습니다. ***\n")
            return 0
    else:
        while result_file_name[i].isdigit():
            num = result_file_name[i] + num
            i -= 1
        try:
            answer_file = open('./data/dt_answer' + num + '.txt', 'r', encoding='utf-8')
        except FileNotFoundError:
            print("*** answer 파일이 없습니다. ***\n")
            return 0
        try:
            result_file = open('./data/' + measure + '_dt_result' + num + '.txt', 'r', encoding='utf-8')
        except FileNotFoundError:
            print("*** " + measure + "_result 파일이 없습니다. ***\n")
            return 0

    answer_rdr = csv.reader(answer_file, delimiter='\t')
    result_rdr = csv.reader(result_file, delimiter='\t')

    answer_data_set = []
    for line in answer_rdr:
        answer_data_set.append(line)
    answer_attributes = answer_data_set.pop(0)
    answer_data = {}
    for attribute in answer_attributes:
        answer_data[attribute] = []
    for data in answer_data_set:
        for i in range(len(data)):
            answer_data[answer_attributes[i]].append(data[i])
    answer_df = pd.DataFrame(answer_data)

    result_data_set = []
    for line in result_rdr:
        result_data_set.append(line)
    result_attributes = result_data_set.pop(0)
    result_data = {}
    for attribute in result_attributes:
        result_data[attribute] = []
    for data in result_data_set:
        for i in range(len(data)):
            result_data[result_attributes[i]].append(data[i])
    result_df = pd.DataFrame(result_data)

    answer_data_set = answer_df.values
    result_data_set = result_df.values

    value = 0
    for i in range(len(result_data_set)):
        if result_data_set[i].all() == answer_data_set[i].all():
            value += 1

    answer_file.close()
    result_file.close()

    return value


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
