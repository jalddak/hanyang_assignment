def calc(data_set_dic):
    data_set = []
    for data_dic in data_set_dic:
        data_set.append(list(data_dic.values()))
    dic = {}
    data_size = 0
    for data in data_set:
        if data[len(data)-1] in dic:
            dic[data[len(data)-1]] += 1
        else:
            dic[data[len(data) - 1]] = 1
        data_size += 1
    print(dic)

    value = 1
    for i in range(len(dic)):
        p = list(dic.values())[i]/data_size
        p_squared = p ** 2
        value -= p_squared
        print(p)
        print(p_squared)