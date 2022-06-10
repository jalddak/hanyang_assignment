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


def main():
    # 입력 받아오는 부분
    args = sys.argv[1:]
    base_name = args[0]
    test_name = args[1]

    base_data = pd.read_csv("./data/" + base_name, delimiter='\t', names=['user_id', 'item_id', 'rating', 'time_stamp'])
    test_data = pd.read_csv("./data/" + test_name, delimiter='\t', names=['user_id', 'item_id', 'rating', 'time_stamp'])
    del base_data['time_stamp']
    del test_data['time_stamp']
    test_data['rating'] = 0

    data = pd.merge(base_data, test_data, how='outer', on=['user_id', 'item_id', 'rating'])

    data_rating = data.pivot_table('rating', index='user_id', columns='item_id').replace(0, np.nan)

    # 원본 행렬 R 생성, 분해 행렬 P와 Q 초기화, 잠재요인 차원 K는 3 설정.
    R = data_rating.values
    num_users, num_items = R.shape
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

    before_rmse = 0
    # SGD 기법으로 P와 Q 매트릭스를 계속 업데이트.
    while True:
        step += 1
        for i, j, r in non_zeros:
            # 실제 값과 예측 값의 차이인 오류 값 구함
            eij = r - np.dot(P[i, :], Q[j, :].T)
            # Regularization을 반영한 SGD 업데이트 공식 적용
            P[i, :] = P[i, :] + learning_rate * (eij * Q[j, :] - r_lambda * P[i, :])
            Q[j, :] = Q[j, :] + learning_rate * (eij * P[i, :] - r_lambda * Q[j, :])

        rmse = get_rmse(R, P, Q, non_zeros)
        if before_rmse != 0 and before_rmse - rmse < 0.001:
            break
        before_rmse = rmse
    pred_matrix = np.dot(P, Q.T)

    pred_matrix_int = np.round(pred_matrix, 0).astype(int)
    result = pd.DataFrame(pred_matrix_int, columns=data_rating.columns, index=data_rating.index)

    output = open('./data/' + base_name + "_prediction.txt", 'w', encoding='utf-8')
    for t in test_data.values:
        predict = result.loc[t[0], t[1]]
        if predict > 5:
            predict = 5
        elif predict < 1 or predict == np.nan:
            predict = 1
        output.write(str(t[0]) + '\t')
        output.write(str(t[1]) + '\t')
        output.write(str(predict) + '\n')
    output.close()


if __name__ == '__main__':
    main()
