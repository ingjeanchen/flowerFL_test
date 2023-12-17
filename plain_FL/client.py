import socket
import tenseal as ts
import numpy as np
import utils
import uuid
import pickle
import torch
import json
from typing import Dict

def train(model, X, y, epochs, optimizer, criterion, batch_size=32):
    dataset = torch.utils.data.TensorDataset(X, y)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for e in range(1, epochs + 1):
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            out = model(inputs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
        
        # 計算整個epoch的平均損失
        total_loss = 0.0
        num_batches = len(dataloader)
        with torch.no_grad():
            for inputs, labels in dataloader:
                out = model(inputs)
                loss = criterion(out, labels)
                total_loss += loss.item()
        average_loss = total_loss / num_batches
        print(f"Epoch [{e}/{epochs}], Average Loss: {average_loss}")

    return model

def test(model, X_test, y_test, criterion):
    loss = 0
    with torch.no_grad():
        pred = (model(X_test) > 0.5).float()
        loss = criterion(pred, y_test)
        correct = (pred == y_test).float()
        accuracy = correct.mean().item()
    # loss = float("{:.2f}".format(loss.item()))
    return loss.item(), accuracy

class Client:
    def __init__(self, serv_host: str, serv_port: int, kgc_host: str, kgc_port: int, num_rounds: int, num_clients: int, num_epochs: int):
        """
        Initialization and configuration of the client.

        serv_sock: server socket, to send encrypted params to the central server
        kgc_sock: key generation center socket, to receive public context and updated params
        config: training configuration data
        id: 8-digit unique client id
        context: public context to encrypt parameters
        model: training model
        """
        self.serv_host = serv_host
        self.serv_port = serv_port
        self.kgc_host = kgc_host
        self.kgc_port = kgc_port

        self.num_rounds = num_rounds
        self.num_clients = num_clients
        self.num_epochs = num_epochs

        self.id = str(uuid.uuid4().hex)[:8]
        self.context = None
        self.model = None
        self.init_sockets()

    def init_sockets(self):
        self.serv_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.serv_socket.connect((self.serv_host, self.serv_port))

        self.kgc_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.kgc_socket.connect((self.kgc_host, self.kgc_port))

        # Send unique id to KGC and server
        uid_msg = pickle.dumps(self.id)
        self.kgc_socket.sendall(uid_msg) # 傳送 unique id 給 key gen center
        self.serv_socket.sendall(uid_msg) # 傳送 unique id 給 server

    def get_data(self):
        """
        Get training and testing data.
        """
        file_paths = ['../nsl-kdd/KDDTrain+.txt', '../nsl-kdd/KDDTest+.txt']
        
        (X_train, y_train), (X_test, y_test) = utils.preprocessing(
            file_path_train=file_paths[0], file_path_test=file_paths[1])
        
        partition_id = np.random.choice(50)
        (X_train, y_train) = utils.partition(X_train, y_train, 50)[partition_id]
        return (X_train, y_train), (X_test, y_test)

    def fit(self, model, X, y, optimizer, criterion):
        """
        Train the model.
        """
        model = train(model, X, y, self.num_epochs, optimizer, criterion, 32)
        
        # Send params to the server after training
        self.send_encrypted_updates(model)
        return model

    def evaluate(self, model, X_test, y_test, criterion):
        """
        Evaluate the trained model.
        """
        test_loss, test_accuracy = test(model, X_test, y_test, criterion)
        print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

    def receive_public_context(self) -> ts.Context:
        """ 
        Get context size first and receive the public context from kgc. 
        """
        ctx_size = utils.recv_size(self.kgc_socket)
        print(f"Context size: {ctx_size}.\nReceiving context...")
        ctx_data = utils.recv_all(self.kgc_socket, ctx_size)

        try:
            # 傳 ack 給 kgc 表示收到 context 了
            self.kgc_socket.send(b'Received')
            ctx = ts.context_from(ctx_data)
            print("Context received!")
        except:
            self.kgc_socket.send(b'Fail')
            raise Exception("cannot deserialize context")
        self.context = ctx

    def send_encrypted_updates(self, model):
        # 加密參數
        weight = ts.ckks_vector(self.context, model.lr.weight.tolist()[0])
        bias = ts.ckks_vector(self.context, model.lr.bias.tolist())
        
        # 添加前綴以區分 weight 和 bias
        weight_prefix = b'weight:'
        bias_prefix = b'bias:'

        # 將 weight 和 bias 序列化為字節流
        weight_data = pickle.dumps(weight.serialize())
        bias_data = pickle.dumps(bias.serialize())
        
        # 先傳送 weight 的前綴和大小
        self.serv_socket.send(weight_prefix + len(weight_data).to_bytes(4, 'big'))
        # 然後傳送 weight 數據
        self.serv_socket.sendall(weight_data)

        # 再傳送 bias 的前綴和大小
        self.serv_socket.send(bias_prefix + len(bias_data).to_bytes(4, 'big'))
        # 然後傳送 bias 數據
        self.serv_socket.sendall(bias_data)

        print("Encrypted params sent to server")

    def receive_global_model(self):
        # 接收 prefix ，以區分 weight 和 bias
        prefix = self.kgc_socket.recv(7)  # 最長 prefix 是 "weight:"，所以接收 7 個字節
        prefix = prefix.decode()

        if prefix == 'weight:':
            # 取得傳來的 weight 參數
            params_size = utils.recv_size(self.kgc_socket)
            params_data = utils.recv_all(self.kgc_socket, params_size)
            weight_params = pickle.loads(params_data)
            print("weight", weight_params)

        elif prefix == 'bias:':
            # 取得傳來的 bias 參數
            params_size = self.kgc_socket.recv(4)
            params_size = int.from_bytes(params_size, 'big')
            params_data = self.kgc_socket.recv(params_size)
            bias_params = pickle.loads(params_data)
            print("bias", bias_params)

        else:
            print("Unknown prefix:", prefix)

if __name__ == "__main__":

    # fit 5 個 round 就會有 50 了
    with open('config.json', 'r') as config_file:
        config = json.load(config_file)
    
    client_config = config['client']
    train_config = config['train_config']


    
    client = Client(client_config['server_host'], client_config['server_port'], 
                client_config['kgc_host'], client_config['kgc_port'],
                train_config['num_rounds'], train_config['num_clients'], client_config['num_epochs'])

    (X_train, y_train), (X_test, y_test) = client.get_data()
    n_features = X_train.shape[1]
    model = utils.BasicLR(n_features)
    print(y_test.size(0))

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = torch.nn.BCELoss()

    client.receive_public_context()
    print("my context is public?", client.context.is_public())

    model = client.fit(model, X_train, y_train, optimizer, criterion)
    client.evaluate(model, X_test, y_test, criterion)

    
    client.kgc_socket.close()
    client.serv_socket.close()
