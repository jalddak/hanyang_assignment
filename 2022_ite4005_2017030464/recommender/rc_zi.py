import numpy as np
import pandas as pd
import sys
from sklearn.metrics import mean_squared_error


def check_rmse(R, U, I, non_zeros):
    pred_R = np.dot(U, I.T)

    users = [non_zero[0] for non_zero in non_zeros]
    items = [non_zero[1] for non_zero in non_zeros]
    R_non_zeros = R[users, items]
    pred_R_non_zeros = pred_R[users, items]

    mse = mean_squared_error(R_non_zeros, pred_R_non_zeros)
    rmse = np.sqrt(mse)

    return rmse


# SGD 방식의 MF 알고리즘
def after_matrix_factorization(R, zero):
    num_users, num_items = R.shape
    # K의 값이 클수록 실제 base_data에 있는 실제값과 SGD 방법을 이용한 MF 방식의
    # 예측값의 유사도는 비슷하였으나, PA4.exe 의 rmse는 오히려 더 떨어졌다.
    # K 값이 클 수록 프로그램 시간이 오래걸리는데, 굳이 크게 해줄 이유가 없어보인다.
    K = 3

    # P와 Q 매트릭스의 크기를 지정하고 정규분포를 가진 random한 값으로 입력합니다.
    # 난수 값은 아무거나 넣어도 성능은 비슷한 것 같았다.
    # 테스트 결과 1-> 0.9599261 / 2017030464 -> 0.9559395
    np.random.seed(2017030464)
    U = np.random.normal(0, 1/K, (num_users, K))
    I = np.random.normal(0, 1/K, (num_items, K))

    # R > 0 인 행 위치, 열 위치, 값을 non_zeros 리스트에 저장.
    non_zeros = [(i, j, R[i, j]) for i in range(num_users) for j in range(num_items) if R[i, j] > 0]

    learning_rate = 0.01
    # 람다 값 1~0.001까지 해봤는데 1은 영 아니고, 0.1~0.001 까진 비슷한 결과나왔다. 그래서 중간인 0.01 정도로 세팅했다.
    lambda_ = 0.01

    # SGD 기법으로 P와 Q 매트릭스를 계속 업데이트.
    count = 0
    before_rmse = 0
    # 어차피 zero injection은 세타가 파라미터로, 결과로 나오는 것들중 세타퍼센트 이하의 것들만 0 or 실제값으로 바꿔줘야 하기 때문에
    if zero:
        for _ in range(10):
            for user, item, rating in non_zeros:
                # 실제 값과 예측 값의 차이인 오류 값 구함
                eui = rating - np.dot(U[user, :], I[item, :].T)
                U[user, :] = U[user, :] + learning_rate * (eui * I[item, :] - lambda_ * U[user, :])
                I[item, :] = I[item, :] + learning_rate * (eui * U[user, :] - lambda_ * I[item, :])

            rmse = check_rmse(R, U, I, non_zeros)
            # 처음엔 그냥 한번만이라도 rmse 가 전과 0.001 이하로 차이나면, 이후의 차이도 비슷하다고 가정하고 그냥 멈췄는데,
            # zero_injection을 수행하는 과정에서 초반에 0.001 이하로 차이나는 경우가 생겨서 바로 끝나버려 제대로 된 예측을 시작하기전에
            # 끝나버려서 기회를 3번정도 더 주는 것으로 코드 수정함.
            # 무조건 많이 하는게 좋은건가 하면 또 아니다. 내 조건을 했을때 보편적으로 35회 정도 MF를 수행하지만, 100번 수행으로 고정해놓고
            # 만들어도 PA4의 rmse는 별 차이가 없고, 오히려 더 증가하는 경우도 있었다.
            if before_rmse != 0 and before_rmse - rmse < 0.001:
                count += 1
            before_rmse = rmse
        pred_matrix = np.dot(U, I.T)

    else:
        while count < 3:
            for user, item, rating in non_zeros:
                # 실제 값과 예측 값의 차이인 오류 값 구함
                eui = rating - np.dot(U[user, :], I[item, :].T)
                U[user, :] = U[user, :] + learning_rate * (eui * I[item, :] - lambda_ * U[user, :])
                I[item, :] = I[item, :] + learning_rate * (eui * U[user, :] - lambda_ * I[item, :])

            rmse = check_rmse(R, U, I, non_zeros)
            # 처음엔 그냥 한번만이라도 rmse 가 전과 0.001 이하로 차이나면, 이후의 차이도 비슷하다고 가정하고 그냥 멈췄는데,
            # zero_injection을 수행하는 과정에서 초반에 0.001 이하로 차이나는 경우가 생겨서 바로 끝나버려 제대로 된 예측을 시작하기전에
            # 끝나버려서 기회를 3번정도 더 주는 것으로 코드 수정함.
            # 무조건 많이 하는게 좋은건가 하면 또 아니다. 내 조건을 했을때 보편적으로 35회 정도 MF를 수행하지만, 100번 수행으로 고정해놓고
            # 만들어도 PA4의 rmse는 별 차이가 없고, 오히려 더 증가하는 경우도 있었다.
            if before_rmse != 0 and before_rmse - rmse < 0.001:
                count += 1
            before_rmse = rmse
        pred_matrix = np.dot(U, I.T)

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
    rt_matrix = rating_table.values

    zero_one_data = pd.merge(base_data, test_data, how='outer', on=['user_id', 'item_id', 'rating'])
    rating = zero_one_data.groupby("rating").count().iloc[:, :1] / zero_one_data.count().user_id
    zero_one_table = data.pivot_table('rating', index='user_id', columns='item_id').replace(rating[1:].index, 1).replace(0, np.nan)
    zot_matrix = zero_one_table.values

    print("zero_injection_start")
    zero_injection_pred_matrix = after_matrix_factorization(zot_matrix, True)
    zi_result = pd.DataFrame(zero_injection_pred_matrix, columns=rating_table.columns, index=rating_table.index)
    print("zero_injection_end")

    print("rating_table_change_start")
    zero_injection_array = zero_injection_pred_matrix.reshape(-1)
    # zero injection의 세타값을 50퍼센트로 잡을 경우이다.
    zi_median = np.median(zero_injection_array)
    for i in range(len(zero_injection_pred_matrix)):
        for j in range(len(zero_injection_pred_matrix[i])):
            if zero_injection_pred_matrix[i][j] < zi_median:
                rt_matrix[i, j] = 2
    print("rating_table_change_end")

    print("cf_start")
    cf_pred_matrix = after_matrix_factorization(rt_matrix, False)
    # 정수형이냐 아니어도 되냐 그 분기점 아직 질문에 대한 답이 안올라왔지만, 아니여도 될듯
    # cf_pred_matrix = np.round(cf_pred_matrix, 0).astype(int)
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