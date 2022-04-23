import numpy as np

import gini
import info
import gain_ratio

class Node:
    def __init__(self, test_attribute=None, cf_attribute_value=None, class_value=None):
        # 뭘로 나뉠 건가
        self.test_attribute = test_attribute
        # 나뉘고 난 뒤 해당하는 값이 뭔가
        self.cf_attribute_value = cf_attribute_value
        # 클래스 레이블의 값이 뭔가 (리프노드라면)
        self.class_value = class_value
        # 자식노드정보
        self.childnodes = []

    def make_tree(self, df, cf_attribute_value, measure):
        self.cf_attribute_value = cf_attribute_value
        if measure == 'gini':
            calc_value = gini.calc(df)
        elif measure == 'gain_ratio':
            calc_value = gain_ratio.calc(df)
        elif measure == 'info':
            calc_value = info.calc(df)
        dic = {}
        for data in df.values:
            if data[len(data) - 1] in dic:
                dic[data[len(data) - 1]] += 1
            else:
                dic[data[len(data) - 1]] = 1

        maxindex = np.argmax(list(dic.values()))
        self.class_value = list(dic.keys())[maxindex]
        if calc_value == 0 or len(df.columns) == 2:
            return self
        if measure == 'gini':
            test_attribute, cf_df_list = gini.find_test_attribute(df)
        elif measure == 'gain_ratio':
            test_attribute, cf_df_list = gain_ratio.find_test_attribute(df)
        elif measure == 'info':
            test_attribute, cf_df_list = info.find_test_attribute(df)
        self.test_attribute = test_attribute

        for cf_df in cf_df_list:
            cf_df_cf_attribute_value = cf_df[test_attribute].values[0]
            del cf_df[test_attribute]
            child_node = Node()
            child_node = child_node.make_tree(cf_df, cf_df_cf_attribute_value, measure)
            self.childnodes.append(child_node)
        return self

    def mining(self, test_data, attributes, measure):
        if len(self.childnodes) == 0:
            test_data[len(test_data)-1] = self.class_value
            return test_data
        else:
            for i in range(len(attributes)):
                if self.test_attribute == attributes[i]:
                    for child in self.childnodes:
                        if test_data[i] == child.cf_attribute_value:
                            if measure == 'gini':
                                test_data = child.mining(test_data, attributes, measure)
                            elif measure == 'gain_ratio':
                                test_data = child.mining(test_data, attributes, measure)
                            elif measure == 'info':
                                test_data = child.mining(test_data, attributes, measure)
                            return test_data
                    test_data[len(test_data) - 1] = self.class_value
                    return test_data

