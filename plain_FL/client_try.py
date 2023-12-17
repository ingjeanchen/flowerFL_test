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
    def __init__(self, serv_host: str, serv_port: int, num_rounds: int, num_clients: int, num_epochs: int):
        """
        Initialization and configuration of the client.

        serv_sock: server socket, to send encrypted params to the central server
        config: training configuration data
        id: 8-digit unique client id
        context: public context to encrypt parameters
        model: training model
        """
        self.serv_host = serv_host
        self.serv_port = serv_port

        self.num_rounds = num_rounds
        self.num_clients = num_clients
        self.num_epochs = num_epochs

        self.id = str(uuid.uuid4().hex)[:8]
        self.model = None
        self.init_sockets()
        self.create_context()

    def init_sockets(self):
        self.serv_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.serv_socket.connect((self.serv_host, self.serv_port))

        # Send unique id to the server
        uid_msg = pickle.dumps(self.id)
        self.serv_socket.sendall(uid_msg) # 傳送 unique id 給 server

    def create_context(self) -> ts.Context:
        poly_mod_degree = 8192  # 定義要使用的多項式模數次數，必須是2的冪次方，會影響加密效率和安全性
        coeff_mod_bit_sizes = [30, 25, 25, 25, 25, 25, 25, 30]
        context = ts.context(
            scheme=ts.SCHEME_TYPE.CKKS, 
            poly_modulus_degree=poly_mod_degree, 
            coeff_mod_bit_sizes=coeff_mod_bit_sizes,
            encryption_type=ts.ENCRYPTION_TYPE.ASYMMETRIC
        )    
        context.global_scale = 2 ** 25
        context.generate_galois_keys()
        context.make_context_public()
        self.context = context

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

        ack_msg = {weight_prefix: b'ACK_W', bias_prefix: b'ACK_B'}
        # 先傳送前綴和大小，再傳數據
        for prefix, data in [(weight_prefix, weight_data), (bias_prefix, bias_data)]:
            print(f"Sending params {prefix.decode()} ...")
            self.serv_socket.send(prefix + len(data).to_bytes(4, 'big'))
            utils.send_chunked_data(self.serv_socket, data)
            ack = self.serv_socket.recv(5)
            if ack != ack_msg[prefix]:
                print(f"Error: Incorrect ACK received for {prefix.decode()}")
            else:
                print(f"Server received {prefix.decode()} params.")
        print("Encrypted params sent to server")


if __name__ == "__main__":

    # fit 5 個 round 就會有 50 了
    with open('config.json', 'r') as config_file:
        config = json.load(config_file)
    
    client_config = config['client']
    train_config = config['train_config']


    
    client = Client(client_config['server_host'], client_config['server_port'], 
                train_config['num_rounds'], train_config['num_clients'], client_config['num_epochs'])

    (X_train, y_train), (X_test, y_test) = client.get_data()
    n_features = X_train.shape[1]
    model = utils.BasicLR(n_features)
    print(y_test.size(0))

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = torch.nn.BCELoss()

    print("my context is public?", client.context.is_public())

    model = client.fit(model, X_train, y_train, optimizer, criterion)
    client.evaluate(model, X_test, y_test, criterion)

    client.serv_socket.close()
