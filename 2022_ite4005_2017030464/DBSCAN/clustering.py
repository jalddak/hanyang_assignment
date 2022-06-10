import sys
import numpy as np
import pandas as pd


def clustering(Eps, MinPts, data, cluster):
    index = 0
    while index < len(cluster):
        select_data_x = cluster.iloc[index, 1]
        select_data_y = cluster.iloc[index, 2]
        in_data_candidate = data.loc[((data['x_coordinate'] - select_data_x) ** 2) +
                                     ((data['y_coordinate'] - select_data_y) ** 2) <= Eps ** 2, :]
        in_cluster_candidate = cluster.loc[((cluster['x_coordinate'] - select_data_x) ** 2) +
                                           ((cluster['y_coordinate'] - select_data_y) ** 2) <= Eps ** 2, :]
        if len(in_data_candidate) != 0:
            cluster_candidate = pd.merge(in_data_candidate, in_cluster_candidate, how='outer',
                                         on=['object_id', 'x_coordinate', 'y_coordinate'])

            if len(cluster_candidate) >= MinPts:
                data = data.loc[((data['x_coordinate'] - select_data_x) ** 2) +
                                ((data['y_coordinate'] - select_data_y) ** 2) > Eps ** 2, :]
                cluster = pd.merge(cluster, in_data_candidate, how='outer',
                                   on=['object_id', 'x_coordinate', 'y_coordinate'])
        index += 1

    return cluster, data


def main():
    # 입력 받아오는 부분
    args = sys.argv[1:]
    input_name = args[0]
    n, Eps, MinPts = map(int, args[1:])

    data = pd.read_csv("./data_input/" + input_name, delimiter='\t', names=['object_id', 'x_coordinate', 'y_coordinate'])

    index = 0
    cluster_candidates = []
    while index < len(data):
        if len(data) < MinPts:
            break

        select_data_x = data.iloc[index, 1]
        select_data_y = data.iloc[index, 2]

        cluster_candidate = data.loc[((data['x_coordinate'] - select_data_x) ** 2) +
                                     ((data['y_coordinate'] - select_data_y) ** 2) <= Eps ** 2, :]

        if len(cluster_candidate) >= MinPts:
            index = 0
            data = data.loc[((data['x_coordinate'] - select_data_x) ** 2) +
                            ((data['y_coordinate'] - select_data_y) ** 2) > Eps ** 2, :]
            cluster_candidate, data = clustering(Eps, MinPts, data, cluster_candidate)
            cluster_candidates.append(cluster_candidate)
        else:
            index += 1

    cluster_candidates.sort(key=len, reverse=True)
    while len(cluster_candidates) != n:
        cluster_candidates.pop()

    index = 0
    for cluster in cluster_candidates:
        output = open('./data_output/' + input_name[:6] + "_cluster_" + str(index) + input_name[6:], 'w', encoding='utf-8')
        for i in range(len(cluster)):
            output.write(str(cluster.iloc[i, 0]) + '\n')
        output.close()
        index += 1


if __name__ == '__main__':
    main()
