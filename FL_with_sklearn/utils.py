from typing import Tuple, Union, List
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler

XY = Tuple[np.ndarray, np.ndarray]
Dataset = Tuple[XY, XY]
LogRegParams = Union[XY, Tuple[np.ndarray]]
XYList = List[XY]

columns = (['duration'
,'protocol_type'
,'service'
,'flag'
,'src_bytes'
,'dst_bytes'
,'land'
,'wrong_fragment'
,'urgent'
,'hot'
,'num_failed_logins'
,'logged_in'
,'num_compromised'
,'root_shell'
,'su_attempted'
,'num_root'
,'num_file_creations'
,'num_shells'
,'num_access_files'
,'num_outbound_cmds'
,'is_host_login'
,'is_guest_login'
,'count'
,'srv_count'
,'serror_rate'
,'srv_serror_rate'
,'rerror_rate'
,'srv_rerror_rate'
,'same_srv_rate'
,'diff_srv_rate'
,'srv_diff_host_rate'
,'dst_host_count'
,'dst_host_srv_count'
,'dst_host_same_srv_rate'
,'dst_host_diff_srv_rate'
,'dst_host_same_src_port_rate'
,'dst_host_srv_diff_host_rate'
,'dst_host_serror_rate'
,'dst_host_srv_serror_rate'
,'dst_host_rerror_rate'
,'dst_host_srv_rerror_rate'
,'outcome'
,'level'])

def get_model_params(model: LogisticRegression) -> LogRegParams:
    if model.fit_intercept:
        params = [
            model.coef_,
            model.intercept_,
        ]
    else:
        params = [
            model.coef_,
        ]
    return params

def set_model_params(
        model: LogisticRegression, params: LogRegParams
    ) -> LogisticRegression:
    model.coef_ = params[0]
    if model.fit_intercept:
        model.intercept_ = params[1]
    return model

def set_initial_params(model: LogisticRegression):
    n_classes = 2
    n_features = 48  # 先用全部48個
    model.classes_ = np.array([i for i in range(2)])
    model.coef_ = np.zeros((n_classes, n_features))
    if model.fit_intercept:
        model.intercept_ = np.zeros((n_classes))
    return


def scaling(df_num, cols):
    std_scaler = StandardScaler()  # 針對超異常的資料標準化成平均值為 0 ，標準差 1 的標準常態分佈
    # 降低異常值的干擾，但是也會減少可解釋性，但在這裡不這樣做模型扛不住過大的離群值
    std_scaler_temp = std_scaler.fit_transform(df_num)
    std_df = pd.DataFrame(std_scaler_temp, columns=cols)
    return std_df


def process_standardize(dataframe, cat_cols):
    df_num = dataframe.drop(cat_cols, axis=1)  # 移除種類資料欄位，只保留數值型欄位
    num_cols = df_num.columns  # 儲存數值型欄位名稱

    scaled_df = scaling(df_num, num_cols)
    # 把標準化後大於3的都換成3
    scaled_df[scaled_df > 3] = 3
    scaled_df[scaled_df < -3] = -3

    dataframe.drop(labels=num_cols, axis="columns", inplace=True)  # 移除原始的數值型欄位
    dataframe[num_cols] = scaled_df[num_cols]  # 將標準化後的資料加入 DataFrame 中

    # 將 "outcome" 欄位中的 "normal" 改為 0，其他值改為 1
    dataframe.loc[dataframe['outcome'] == "normal", "outcome"] = 0
    dataframe.loc[dataframe['outcome'] != 0, "outcome"] = 1

    dataframe = pd.get_dummies(dataframe, columns=['protocol_type', 'service', 'flag'])  # 對特定種類資料進行 one-hot 編碼
    return dataframe


def preprocessing_data(data):
    # 不為 "normal" 的資料，改成 'attack'
    data.loc[data['outcome'] != 'normal', 'outcome'] = 'attack'
    n_features = data.shape[1]

    # 合併多餘種類為other
    data['flag'] = np.where(
        (data['flag'] != 'SF') & (data['flag'] != 'S0') & (data['flag'] != 'REJ'), 'other',
        data['flag'])  # 將 "flag" 欄位中不符合條件的值改成 "other"
    valid_services = ["http", "private", "domain_u", "smtp", "ftp_data"]
    data['service'] = np.where(~data['service'].isin(valid_services), 'other', data['service'])

    # 資料標準化
    cat_cols = ['is_host_login', 'protocol_type', 'service', 'flag', 'land', 'logged_in', 'is_guest_login', 'level',
                'outcome']
    scaled_train = process_standardize(data, cat_cols)
    scaled_train = scaled_train.drop(['protocol_type_icmp', 'service_other', 'flag_other'], axis=1)
    # 去掉一個虛擬變數可能比較準
    return scaled_train


def preprocessing(file_path_train, file_path_test):
    data_train = pd.read_csv(file_path_train, header=None, names=columns)
    data_test = pd.read_csv(file_path_test, header=None, names=columns)

    df_train = preprocessing_data(data_train)
    df_test = preprocessing_data(data_test)
    df_train['outcome'] = df_train['outcome'].astype(float)
    df_test['outcome'] = df_test['outcome'].astype(float)

    x_train = df_train.drop(['outcome', 'level'], axis=1).values
    x_test = df_test.drop(['outcome', 'level'], axis=1).values

    y_train = torch.tensor(df_train['outcome'].values).float().unsqueeze(1)
    y_test = torch.tensor(df_test['outcome'].values).float().unsqueeze(1)

    return (x_train, y_train), (x_test, y_test)


def shuffle(X: np.ndarray, y: np.ndarray) -> XY:
    """Shuffle X and y."""
    rng = np.random.default_rng()
    idx = rng.permutation(len(X))
    return X[idx], y[idx]


def partition(X: np.ndarray, y: np.ndarray, num_partitions: int) -> XYList:
    """Split X and y into a number of partitions."""
    return list(
        zip(np.array_split(X, num_partitions), np.array_split(y, num_partitions))
    )
