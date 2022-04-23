import math

import numpy as np


def calc(df):
    data_size = len(df.values)
    dic = {}
    for data in df.values:
        if data[len(data) - 1] in dic:
            dic[data[len(data) - 1]] += 1
        else:
            dic[data[len(data) - 1]] = 1

    value = 0
    for i in range(len(dic)):
        p = list(dic.values())[i] / data_size
        log_2 = math.log(p, 2)
        value -= p * log_2

    return value


def find_test_attribute(df):
    attributes = df.columns
    data_set = df.values
    data_size = len(data_set)
    compare_value_list = []
    compare_df_list_list = []
    for i in range(len(attributes) - 1):

        attribute_values = []
        for value in df.values:
            if value[i] not in attribute_values:
                attribute_values.append(value[i])

        compare_df_list = []
        for value in attribute_values:
            compare_df_list.append(df.loc[df[attributes[i]] == value, :])

        calc_result = 0
        for compare_df in compare_df_list:
            calc_result += (len(compare_df.values) / data_size) * calc(compare_df)

        compare_value_list.append(calc_result)
        compare_df_list_list.append(compare_df_list)

    minindex = np.argmin(compare_value_list)

    # print(compare_value_list)
    # print(attributes[minindex])
    # print(compare_df_list_list[minindex])

    return attributes[minindex], compare_df_list_list[minindex]
