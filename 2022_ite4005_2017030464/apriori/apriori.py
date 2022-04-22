import sys
import csv
import numpy as np
from itertools import combinations
import math

def main():
    # 입력 받아오는 부분
    args = sys.argv[1:]
    min_sup = int(args[0])
    try:
        input = open(args[1], 'r', encoding='utf-8')
    except FileNotFoundError:
        print("*** input 파일이 없습니다. ***\n")
        return 0
    output = open(args[2], 'w', encoding='utf-8')

    # 인풋 자르고 트랜잭션으로 해가지고 다 넣는 모습
    rdr = csv.reader(input, delimiter='\t')
    transaction = []
    transactions = []
    tran_count = 0
    dict = {}
    sub_dict = {}
    for line in rdr:
        for i in line:
            transaction.append(int(i))
            if tuple([int(i), ]) in sub_dict:
                sub_dict[tuple([int(i), ])] += 1
            else:
                sub_dict[tuple([int(i), ])] = 1
        transaction_tp = tuple(transaction)
        transaction = []
        transactions.append(transaction_tp)
        tran_count += 1
    transactions_np = np.array(transactions, dtype=object)

    element_count = 1

    while True:
        if element_count != 1:
            # 살아남은 집합들 합집합해서 우선 집합 요소 개수로 맞는 애들만 살려놈 (self joining)
            key_sets_sj = []
            for i in range(len(keys)):
                for j in range(i + 1, len(keys)):
                    key_set = set(list(keys[i])).union(set(list(keys[j])))
                    if len(key_set) == element_count and key_set not in key_sets_sj:
                        key_sets_sj.append(key_set)
            # 프루닝 과정
            key_sets = []
            for key_set in key_sets_sj:
                subset_list = list(combinations(key_set, element_count - 1))
                new_subset_list = []
                for subset_element in subset_list:
                    subset_element_list = list(subset_element)
                    subset_element_list.sort()
                    new_subset_list.append(tuple(subset_element_list))
                subset = set(new_subset_list)
                if subset == set(keys).intersection(subset):
                    key_sets.append(key_set)
            # 구해낸 집합이 트랜잭션에 몇개나 있을까 확인 중
            sub_dict = {}
            for key_set in key_sets:
                for i in range(len(transactions_np)):
                    bool_index = (key_set == set(list(transactions_np[i])).intersection(key_set))
                    if bool_index:
                        key_set_list = list(key_set)
                        key_set_list.sort()
                        key_set_tuple = tuple(key_set_list)
                        # key_set_tuple = tuple(key_set)
                        if key_set_tuple in sub_dict:
                            sub_dict[key_set_tuple] += 1
                        else:
                            sub_dict[key_set_tuple] = 1

        # 서포트 안맞는애들 다 걸러내는 중
        keys = list(sub_dict.keys())
        values = list(sub_dict.values())
        del_list = []
        for i in range(len(values)):
            support = values[i] / tran_count * 100
            if support < min_sup:
                del_list.append(keys[i])
        for i in del_list:
            del sub_dict[i]

        # 서포트 안맞는 애들 다 걸러냈는데 결국 남는게 없으면 더가도 답없으니까 그만둔다.
        if len(sub_dict) == 0:
            break

        # 서포트 맞는 애들만 이제 dict에 넣어줌. 왜냐면 이따가 컨피던스구할때 각 집합별로 개수 알아야해서 저장중
        keys = list(sub_dict.keys())
        values = list(sub_dict.values())
        for i in range(len(keys)):
            dict[keys[i]] = values[i]

        for key in keys:
            for i in range(1, element_count):
                subset = list(combinations(set(list(key)), i))
                for tuple_x_ns in subset:
                    set_x_ns = set(tuple_x_ns)
                    set_y_ns = set(key) - set_x_ns
                    x_list = list(set_x_ns)
                    x_list.sort()
                    tuple_x = tuple(x_list)
                    output.write(str(set_x_ns) + '\t' + str(set_y_ns) + '\t' + "{:.2f}".format(dict[key] / tran_count * 100)
                                 + '\t' + "{:.2f}".format(dict[key] / dict[tuple_x] * 100) + '\n')

                    # 채점프로그램의 normal round 와 scientific round의 차이가 어떤 숫자에서 나는지 확인하기 위해 작성했었음.
                    # if math.trunc((dict[key] / dict[tuple_x] * 100 * 1000) % 100) == 5:
                    #     print(str(set_x_ns) + '\t' + str(set_y_ns) + '\t' + "{:.2f}".format(dict[key] / dict[tuple_x] * 100))
                    # if math.trunc((dict[key] / dict[tuple_x] * 100 * 1000) % 100) == 25:
                    #     print(str(set_x_ns) + '\t' + str(set_y_ns) + '\t' + "{:.2f}".format(dict[key] / dict[tuple_x] * 100))
                    # if math.trunc((dict[key] / dict[tuple_x] * 100 * 1000) % 100) == 45:
                    #     print(str(set_x_ns) + '\t' + str(set_y_ns) + '\t' + "{:.2f}".format(dict[key] / dict[tuple_x] * 100))
                    # if math.trunc((dict[key] / dict[tuple_x] * 100 * 1000) % 100) == 65:
                    #     print(str(set_x_ns) + '\t' + str(set_y_ns) + '\t' + "{:.2f}".format(dict[key] / dict[tuple_x] * 100))
                    # if math.trunc((dict[key] / dict[tuple_x] * 100 * 1000) % 100) == 85:
                    #     print(str(set_x_ns) + '\t' + str(set_y_ns) + '\t' + "{:.2f}".format(dict[key] / dict[tuple_x] * 100))

        element_count += 1

    input.close()
    output.close()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
