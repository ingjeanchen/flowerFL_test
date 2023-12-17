import socket
import pickle
import threading
import tenseal as ts
import json
from typing import Dict, List

class Server:
    def __init__(self, cli_host: str, cli_port: int, kgc_host: str, kgc_port: int, num_rounds: int, num_clients: int):
        self.cli_host = cli_host
        self.cli_port = cli_port
        self.kgc_host = kgc_host
        self.kgc_port = kgc_port
        self.num_rounds = num_rounds
        self.num_clients = num_clients
        self.client_weights = []
        self.client_biases = []

    def init_sockets(self):
        self.cli_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.cli_socket.bind((self.cli_host, self.cli_port))
        self.cli_socket.listen(self.num_clients) # client 數量要指定, 不可以變動
        accept_timeout = 20
        self.cli_socket.settimeout(accept_timeout)

        self.kgc_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.kgc_socket.connect((self.kgc_host, self.kgc_port))  # 連接到 KGC   

    def aggregate_encrypted_parameters(self, parameters_list: List[ts.CKKSVector]) -> ts.CKKSVector:
        aggregated = parameters_list[0]
        for params in parameters_list[1:]:
            aggregated += params
        return aggregated


    def handle_client(self, cli_sock: socket.socket, cli_id: str):
        # 接收 prefix ，以區分 weight 和 bias
        prefix = cli_sock.recv(7)  # 最長 prefix 是 "weight:"，所以接收 7 個字節
        prefix = prefix.decode()

        if prefix == 'weight:':
            # 取得傳來的 weight 參數
            params_size = cli_sock.recv(4)
            params_size = int.from_bytes(params_size, 'big')
            params_data = cli_sock.recv(params_size)
            weight_params = pickle.loads(params_data)
            print(f"Received weight parameters from {cli_id}.")

            # 加入等待聚合的 weight list 中
            self.client_weights.append(weight_params)

        elif prefix == 'bias:':
            # 取得傳來的 bias 參數
            params_size = cli_sock.recv(4)
            params_size = int.from_bytes(params_size, 'big')
            params_data = cli_sock.recv(params_size)
            bias_params = pickle.loads(params_data)
            print(f"Received bias parameters from {cli_id}.")

            # 加入等待聚合的 bias list 中
            self.client_biases.append(bias_params)
        else:
            print("Unknown prefix:", prefix)

        print(f"Client {cli_id}'s parameters are successfully received.")


    def start(self):
        self.init_sockets()
        for round_num in range(self.num_rounds):
            self.client_weights = []
            self.client_biases = []
            client_threads = []

            print(f"Round {round_num + 1}: Waiting for clients to connect...")

            for _ in range(self.num_clients):
                try:
                    client_socket, client_address = self.cli_socket.accept()

                    # 得到 client 的 id
                    client_id = client_socket.recv(32)
                    client_id = pickle.loads(client_id)

                    print(f"Client {client_address} connected, ID: {client_id}")

                    cli_thread = threading.Thread(target=self.handle_client, args=(client_socket, client_id))
                    cli_thread.start()
                    client_threads.append(cli_thread)

                except socket.timeout:
                    print("Client accept timeout.")

                except Exception as e:
                    print(f"Error accepting client connection: {e}")

            # 等待所有客戶端線程完成
            for thread in client_threads:
                if thread != threading.current_thread():
                    thread.join()

            print(f"Round {round_num + 1}: Aggregating parameters...")
            aggregated_weight = self.aggregate_encrypted_parameters(self.client_weights)
            aggregatted_bias = self.aggregate_encrypted_parameters(self.client_biases)
            print(f"Round {round_num + 1}: Aggregation completed.")

            weight_prefix = b'weight:'
            bias_prefix = b'bias:'
            
            weight_data = pickle.dumps(aggregated_weight.serialize())
            bias_data = pickle.dumps(aggregatted_bias.serialize())
            
            # 先傳送 weight 的前綴和大小
            self.kgc_sock.send(weight_prefix + len(weight_data).to_bytes(4, 'big'))
            # 然後傳送 weight 數據
            self.kgc_sock.sendall(weight_data)

            # 再傳送 bias 的前綴和大小
            self.kgc_sock.send(bias_prefix + len(bias_data).to_bytes(4, 'big'))
            # 然後傳送 bias 數據
            self.kgc_sock.sendall(bias_data)

        # 回合結束後關閉 socket
        self.cli_socket.close()
        self.kgc_socket.close()

if __name__ == '__main__':
    with open('config.json', 'r') as config_file:
        config = json.load(config_file)
    
    server_config = config['server']
    train_config = config['train_config']

    server = Server(server_config['client_host'], server_config['client_port'], 
                    server_config['kgc_host'], server_config['kgc_port'],
                    train_config['num_rounds'], train_config['num_clients'])

    server.start()