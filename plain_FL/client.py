import socket
import time
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
        out = model(X_test)
        loss = criterion(out, y_test)
        pred = (out > 0.5).float()

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
        self.connect_kgc()

    def connect_kgc(self):
        self.kgc_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.kgc_socket.connect((self.kgc_host, self.kgc_port))

        # Send unique id to KGC
        uid_msg = pickle.dumps(self.id)
        self.kgc_socket.sendall(uid_msg) # 傳送 unique id 給 key gen center

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
        return model

    def evaluate(self, model, X_test, y_test, criterion):
        """
        Evaluate the trained model.
        """
        test_loss, test_accuracy = test(model, X_test, y_test, criterion)
        print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

    def fetch_context(self) -> ts.Context:
        """ 
        Get context size first and receive the public context from kgc. 
        """
        print("fetching context...")
        self.context = utils.receive_public_context(self.kgc_socket)

    def send_encrypted_updates(self, model):
        try:
            self.serv_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.serv_socket.connect((self.serv_host, self.serv_port))

            uid_msg = pickle.dumps(self.id)
            self.serv_socket.sendall(uid_msg)
            ack = self.serv_socket.recv(6)
            if ack == b'ACK_ID':
                print("server got id")
                utils.send_updates(self.serv_socket, 'client', model.lr.weight.tolist()[0], 
                            model.lr.bias.tolist(), to_encrypt=True, context=self.context)
            
        except Exception as e:
            print(f"Error sending updates to server: {e}")

        finally:
            self.serv_socket.close()

    def receive_global_model(self, model):
        kgc_params = {'weight': None, 'bias': None}
        utils.receive_parameters(self.kgc_socket, 'client', kgc_params=kgc_params)
        with torch.no_grad():
            model.lr.weight = torch.nn.Parameter(torch.tensor(kgc_params['weight']).float().unsqueeze(0))
            model.lr.bias = torch.nn.Parameter(torch.tensor(kgc_params['bias']).float())
        return model
    
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

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = torch.nn.BCELoss()

    print(model.lr.weight)
    print(model.lr.bias)

    for round in range(train_config['num_rounds']):
        print(f"Starting training round {round + 1}")
        client.fetch_context()
        model = client.fit(model, X_train, y_train, optimizer, criterion)   
        client.send_encrypted_updates(model)
        model = client.receive_global_model(model)
        client.evaluate(model, X_test, y_test, criterion)
    
    client.kgc_socket.close()
