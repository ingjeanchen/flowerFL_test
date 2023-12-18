import socket
from typing import Tuple, Union, List
import numpy as np
import pandas as pd
import torch
import pickle
import tenseal as ts
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


def get_model_params(model, encrypted=False, context=None):
    if encrypted and context:
        # Convert parameters to encrypted form
        print("def get_model_params - if/weight: ", model.lr.weight, type(model.lr.weight))

        weight = ts.ckks_vector(context, model.lr.weight.tolist()[0])
        bias = ts.ckks_vector(context, model.lr.bias.tolist())
        # print("def get_model_params - if/weight: ", weight, type(weight))
        return [weight, bias]
    else:
        # Return plain parameters
        print("def get_model_params - else/weight: ", weight, type(weight))
        return [model.lr.weight, model.lr.bias]


def set_model_params(model, params, encrypted=False, context=None):
    if encrypted and context:
        # Decrypt parameters and set them to the model
        weight = ts.ckks_vector_from(context, params[0])
        bias = ts.ckks_vector_from(context, params[1])
        print("def set_model_params - if/weight: ", weight, type(weight))
        model.lr.weight = torch.tensor(weight.decrypt())
        model.lr.bias = torch.tensor(bias.decrypt())
    else:
        # Set plain parameters to the model
        model.lr.weight = params[0]
        model.lr.bias = params[1]
        print("def set_model_params - params: ", params)
        print("def set_model_params - else/weight: ", model.lr.weight, type(model.lr.weight))


def set_initial_params(model: LogisticRegression):
    n_classes = 2
    n_features = 48  # 先用全部48個
    model.classes_ = np.array([i for i in range(2)])
    model.coef_ = np.zeros((n_classes, n_features))
    if model.fit_intercept:
        model.intercept_ = np.zeros((n_classes))
    return


class BasicLR(torch.nn.Module):
    def __init__(self, n_features):
        super(BasicLR, self).__init__()
        self.lr = torch.nn.Linear(n_features, 1)    # .to(device)  # 只有一個輸出值 (0,1)

    def forward(self, x):
        x = x   # .to(device)
        out = torch.sigmoid(self.lr(x))
        return out


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

    x_train = torch.tensor(x_train.astype(float), dtype=torch.float32)
    x_test = torch.tensor(x_test.astype(float), dtype=torch.float32)

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

def recv_size(sock: socket.socket):
    """
    Helper function to receive data size
    """
    size_data = sock.recv(4)
    if len(size_data) != 4:
        raise RuntimeError("Failed to receive data size")
    return int.from_bytes(size_data, 'big')

def recv_all(sock: socket.socket, size: int):
        data = b''
        while len(data) < size:
            packet = sock.recv(size - len(data))
            if not packet:
                raise RuntimeError("Failed to receive data")
            data += packet
        return data

def send_chunked_data(socket: socket.socket, data):
    CHUNK_SIZE = 1024
    chunks = [data[i:i + CHUNK_SIZE] for i in range(0, len(data), CHUNK_SIZE)]
    for chunk in chunks:
        socket.sendall(chunk)
    socket.sendall(b'END')  # 發送結束的信號

def receive_chunked_data(socket: socket.socket):
    data = b''
    while True:
        chunk = socket.recv(1024)
        if b'END' in chunk:
            data += chunk.split(b'END')[0]
            break
        data += chunk
    return data

def receive_public_context(socket: socket.socket) -> ts.Context:
    """ 
    Get context size first and receive the public context. 
    """
    ctx_size = recv_size(socket)
    print(f"Context size: {ctx_size}.\nReceiving context...")
    ctx_data = recv_all(socket, ctx_size)

    try:
        # 傳 ack 給對方表示收到 context 了
        socket.send(b'Received')
        ctx = ts.context_from(ctx_data)
        print("Context received!")
    except:
        socket.send(b'Fail')
        raise Exception("cannot deserialize context")
    return ctx

def receive_parameters(socket: socket.socket, who_id: str, client_params=None, context=None, server_params=None, kgc_params=None):
    got_weight = False
    got_bias = False
    try:
        while not(got_weight and got_bias):
            # 接收 prefix ，以區分 weight 和 bias
            prefix = socket.recv(7)  # 最長 prefix 是 "weight:"，所以接收 7 個字節
            if not prefix:
                continue
            prefix = prefix.decode()

            # 取得傳來的參數
            params_data = receive_chunked_data(socket)
            params = pickle.loads(params_data)

            if prefix.startswith('weight'):
                print(f"Received weight parameters from {who_id}.")
                
                # server 處理 client 加密後的參數
                if client_params and context:
                    ckks_weight = ts.ckks_vector_from(context, params)
                    client_params['weight'].append(ckks_weight) # 加入等待聚合的 weight list 中
                
                if server_params and context:
                    ckks_weight = ts.ckks_vector_from(context, params)
                    server_params['weight'] = ckks_weight.decrypt(context.secret_key())

                if kgc_params:
                    kgc_params['weight'] = params

                socket.sendall(b'ACK_W')
                got_weight = True

            elif prefix.startswith('bias'):
                print(f"Received bias parameters from {who_id}.")
                
                # server 處理 client 加密後的參數
                if client_params and context:
                    ckks_bias = ts.ckks_vector_from(context, params)
                    client_params['bias'].append(ckks_bias) # 加入等待聚合的 bias list 中
                
                if server_params and context:
                    ckks_bias = ts.ckks_vector_from(context, params)
                    server_params['bias'] = ckks_bias.decrypt(context.secret_key())

                if kgc_params:
                    kgc_params['bias'] = params

                socket.sendall(b'ACK_B')
                got_bias = True
            else:
                print("Unknown prefix:", prefix)
    
    except Exception as e:
        if who_id == 'server':
            print(f"Error handling Server: {e}")
        elif who_id == 'client':
            print(f"Error interacting with KGC: {e}")
        else:
            print(f"Error handling Client {who_id}: {e}")

def send_updates(socket: socket.socket, role: str, weight, bias, to_encrypt=True, context=None, cli_id=''):
        # 加密參數
        if to_encrypt and context:
            weight = ts.ckks_vector(context, weight)
            bias = ts.ckks_vector(context, bias)

        # 添加前綴以區分 weight 和 bias
        weight_prefix = b'weight:'
        bias_prefix = b'bias:'

        # 將 weight 和 bias 序列化為字節流
        if role != 'kgc':
            weight_data = pickle.dumps(weight.serialize())
            bias_data = pickle.dumps(bias.serialize())
        else:
            weight_data = pickle.dumps(weight)
            bias_data = pickle.dumps(bias)
        ack_msg = {weight_prefix: b'ACK_W', bias_prefix: b'ACK_B'}
        retry_limit = 10

        # 先傳送前綴和大小，再傳數據
        for prefix, data in [(weight_prefix, weight_data), (bias_prefix, bias_data)]:
            attempts = 0
            param_succ_msg = {'client': f"Server received {prefix.decode()[:-1]}.", 
                'server': f"KGC received {prefix.decode()[:-1]}.",
                'kgc': f"Client {cli_id} received updated {prefix.decode()[:-1]}."
                }
            while attempts < retry_limit:
                print(f"Attempt {attempts + 1}: Sending params {prefix.decode()[:-1]}...")
                socket.sendall(prefix)
                send_chunked_data(socket, data)
                ack = socket.recv(5)

                if ack == ack_msg[prefix]:
                    print(f"{param_succ_msg[role]}")
                    break
                else:
                    print(f"Warning: Incorrect ACK received for {prefix.decode()}. Retrying...")
                    attempts += 1

            if attempts == retry_limit:
                print(f"Error: Failed to send {prefix.decode()[:-1]} after {retry_limit} attempts.")
                return False  # Indicate failure after maximum attempts

        succ_msg = {'client': "Encrypted params sent to server.", 
                    'server': "Aggregated params sent to KGC.",
                    'kgc': f'Updated parameters sent to Client {cli_id}.'
                    }
        print(succ_msg[role])
        return True