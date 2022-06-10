import numpy as np
import pandas as pd
import sys
from sklearn.metrics import mean_squared_error


def get_rmse(R, P, Q, non_zeros):
    error = 0
    # 두개의 분해된 행렬 P와 Q.T의 내적으로 예측 R 행렬 생성
    full_pred_matrix = np.dot(P, Q.T)

    # 실제 R 행렬에서 널이 아닌 값의 위치 인덱스 추출하여 실제 R 행렬과 예측 행렬의 RMSE 추출
    x_non_zero_ind = [non_zero[0] for non_zero in non_zeros]
    y_non_zero_ind = [non_zero[1] for non_zero in non_zeros]
    R_non_zeros = R[x_non_zero_ind, y_non_zero_ind]
    full_pred_matrix_non_zeros = full_pred_matrix[x_non_zero_ind, y_non_zero_ind]

    mse = mean_squared_error(R_non_zeros, full_pred_matrix_non_zeros)
    rmse = np.sqrt(mse)

    return rmse


def after_matrix_factorization(table):
    R = table.values
    num_users, num_items = R.shape
    # K의 값이 클수록 실제 base_data에 있는 실제값과 SGD 방법을 이용한 MF 방식의
    # 예측값의 유사도는 비슷하였으나, PA4.exe 의 rmse는 오히려 더 떨어졌다.
    # K 값이 클 수록 프로그램 시간이 오래걸리는데, 굳이 크게 해줄 이유가 없어보인다.
    K = 3

    # P와 Q 매트릭스의 크기를 지정하고 정규분포를 가진 random한 값으로 입력합니다.
    np.random.seed(1)
    P = np.random.normal(scale=1. / K, size=(num_users, K))
    Q = np.random.normal(scale=1. / K, size=(num_items, K))

    # R > 0 인 행 위치, 열 위치, 값을 non_zeros 리스트에 저장.
    non_zeros = [(i, j, R[i, j]) for i in range(num_users) for j in range(num_items) if R[i, j] > 0]

    step = 0
    learning_rate = 0.01
    r_lambda = 0.01

    # SGD 기법으로 P와 Q 매트릭스를 계속 업데이트.
    count = 0
    before_rmse = 0
    while count < 3:
        step += 1
        for i, j, r in non_zeros:
            # 실제 값과 예측 값의 차이인 오류 값 구함
            eij = r - np.dot(P[i, :], Q[j, :].T)
            # Regularization을 반영한 SGD 업데이트 공식 적용
            P[i, :] = P[i, :] + learning_rate * (eij * Q[j, :] - r_lambda * P[i, :])
            Q[j, :] = Q[j, :] + learning_rate * (eij * P[i, :] - r_lambda * Q[j, :])

        rmse = get_rmse(R, P, Q, non_zeros)
        # 처음엔 그냥 한번만이라도 rmse 가 전과 0.001 이하로 차이나면, 비슷하다고 가정하고 그냥 멈췄는데,
        # zero_injection을 수행하는 과정에서 초반에 0.001 이하로 차이나는 경우가 생겨서 바로 끝나버려 제대로 된 예측을 시작하기전에
        # 끝나버려서 기회를 3번정도 더 주는 것으로 코드 수정함.
        if before_rmse != 0 and before_rmse - rmse < 0.001:
            count += 1
        before_rmse = rmse
    pred_matrix = np.dot(P, Q.T)

    return pred_matrix


def main():
    # 입력 받아오는 부분
    args = sys.argv[1:]
    base_name = args[0]
    test_name = args[1]

    # zero_injection 의 경우도 생각해보려했으나, 너무 오래걸리고, PA4.exe 의 rmse 결과 또한 더 나빠져서 그냥 안했다.
    # (주석 처리된 코드는 모두 zero_injection 관련 코드이다. 혹시몰라서 주석처리로 남겨뒀다.)
    base_data = pd.read_csv("./data/" + base_name, delimiter='\t', names=['user_id', 'item_id', 'rating', 'time_stamp'])
    test_data = pd.read_csv("./data/" + test_name, delimiter='\t', names=['user_id', 'item_id', 'rating', 'time_stamp'])
    # test_data2 = pd.read_csv("./data/" + test_name, delimiter='\t', names=['user_id', 'item_id', 'rating', 'time_stamp'])
    del base_data['time_stamp']
    del test_data['time_stamp']
    test_data['rating'] = 0
    # del test_data2['time_stamp']
    # del test_data2['rating']

    data = pd.merge(base_data, test_data, how='outer', on=['user_id', 'item_id', 'rating'])
    # zero_one_data = pd.merge(base_data, test_data2, how='outer', on=['user_id', 'item_id'])

    rating_table = data.pivot_table('rating', index='user_id', columns='item_id').replace(0, np.nan)
    # ratings = zero_one_data.groupby('rating').count().iloc[:, :1] / zero_one_data.count().user_id
    # zero_one_table = data.pivot_table('rating', index='user_id', columns='item_id').replace(ratings.index, 1)
    # zero_one_table = zero_one_table.fillna(0)

    # print("zero_injection_start")
    # zero_injection_pred_matrix = after_matrix_factorization(zero_one_table)
    # zero_injection_pred_matrix = np.round(zero_injection_pred_matrix, 3)
    # zi_result = pd.DataFrame(zero_injection_pred_matrix, columns=rating_table.columns, index=rating_table.index)
    # print("zero_injection_end")
    #
    # print("rating_table_change_start")
    # for i in rating_table.index:
    #     for j in rating_table.columns:
    #         if zi_result.loc[i, j] < 0.5:
    #             rating_table.loc[i, j] = 1
    # print("rating_table_change_end")

    print("cf_start")
    cf_pred_matrix = after_matrix_factorization(rating_table)
    cf_pred_matrix = np.round(cf_pred_matrix, 0).astype(int)
    cf_result = pd.DataFrame(cf_pred_matrix, columns=rating_table.columns, index=rating_table.index)
    print("cf_end")

    print("output_write_start")
    output = open('./data/' + base_name + "_prediction.txt", 'w', encoding='utf-8')
    for t in test_data.values:
        predict = cf_result.loc[t[0], t[1]]
        if predict > 5:
            predict = 5
        elif predict < 1 or predict == np.nan:
            # else의 경우는 아예 새로운 데이터 였을 경우라고 생각했다. 그럴 경우, 거의 모든 값이 0보다 작은 값인 것 같다.
            if (cf_result[t[1]].sum() / len(cf_result)) > 0.5:
                predict = 1
            else:
                select = base_data.loc[base_data['user_id'] == t[0]]
                predict = int((select['rating'].sum() / len(select)).round())
        output.write(str(t[0]) + '\t')
        output.write(str(t[1]) + '\t')
        output.write(str(predict) + '\n')
    output.close()
    print("output_write_end")


if __name__ == '__main__':
    main()