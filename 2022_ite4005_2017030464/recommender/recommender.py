import numpy as np
import pandas as pd
import sys
from sklearn.metrics import mean_squared_error


def check_rmse(R, U, I, not_nans, b_U, b_I, mean):
    pred_R = mean + b_U[:, np.newaxis] + b_I[np.newaxis, :] + np.dot(U, I.T)

    users = [not_nan[0] for not_nan in not_nans]
    items = [not_nan[1] for not_nan in not_nans]
    R_not_nans = R[users, items]
    pred_R_not_nans = pred_R[users, items]

    mse = mean_squared_error(R_not_nans, pred_R_not_nans)
    rmse = np.sqrt(mse)

    return rmse


# SGD 방식의 MF 알고리즘
def SGD_matrix_factorization(table):
    R = table.values
    num_users, num_items = R.shape
    K = 3
    np.random.seed(2017030464)

    U = np.random.normal(0, 1/K, (num_users, K))
    I = np.random.normal(0, 1/K, (num_items, K))

    not_nans = [(i, j, R[i, j]) for i in range(num_users) for j in range(num_items) if not pd.isna(R[i, j])]

    # 편향 추가
    b_U = np.zeros(num_users)
    b_I = np.zeros(num_items)
    mean = np.nanmean(R)

    learning_rate = 0.01
    lambda_ = 0.01

    count = 0
    iter = 0
    before_rmse = 0
    while count < 3 and iter < 50:
        iter += 1
        for user, item, rating in not_nans:
            # 실제 값과 예측 값의 차이인 오류 값 구함
            pred_R = mean + b_U[user] + b_I[item] + np.dot(U[user, :], I[item, :].T)
            error = rating - pred_R

            b_U[user] += learning_rate * (error - lambda_ * b_U[user])
            b_I[item] += learning_rate * (error - lambda_ * b_I[item])
            U[user, :] += learning_rate * (error * I[item, :] - lambda_ * U[user, :])
            I[item, :] += learning_rate * (error * U[user, :] - lambda_ * I[item, :])

        rmse = check_rmse(R, U, I, not_nans, b_U, b_I, mean)
        print(str(iter) + " times MF's rmse: " + str(rmse))
        if before_rmse != 0 and before_rmse - rmse < 0.001:
            count += 1
        before_rmse = rmse
    print("MF was performed " + str(iter) + " times.")
    pred_matrix = mean + b_U[:, np.newaxis] + b_I[np.newaxis, :] + np.dot(U, I.T)

    return pred_matrix


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

    # merge 하는 이유: test 데이터에는 존재하고 base에 없는 데이터가 존재할 수도 있기 때문이다.
    data = pd.merge(base_data, test_data, how='outer', on=['user_id', 'item_id', 'rating'])
    rating_table = data.pivot_table('rating', index='user_id', columns='item_id').replace(0, np.nan)

    print("cf_start")
    cf_pred_matrix = SGD_matrix_factorization(rating_table)
    cf_result = pd.DataFrame(cf_pred_matrix, columns=rating_table.columns, index=rating_table.index)
    print("cf_end")

    print("output_write_start")
    output = open('./data/' + base_name + "_prediction.txt", 'w', encoding='utf-8')
    for t in test_data.values:
        predict = cf_result.loc[t[0], t[1]]
        output.write(str(t[0]) + '\t')
        output.write(str(t[1]) + '\t')
        output.write(str(predict) + '\n')
    output.close()
    print("output_write_end")
    return 0


if __name__ == '__main__':
    main()