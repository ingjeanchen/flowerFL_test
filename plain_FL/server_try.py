import socket
import pickle
import threading
import utils
import tenseal as ts
import json
from typing import Dict, List

class Server:
    def __init__(self, cli_host: str, cli_port: int, num_rounds: int, num_clients: int):
        self.cli_host = cli_host
        self.cli_port = cli_port
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

    def aggregate_encrypted_parameters(self, parameters_list: List[ts.CKKSVector]) -> ts.CKKSVector:
        aggregated = parameters_list[0]
        for params in parameters_list[1:]:
            aggregated += params
        return aggregated

    def handle_client(self, cli_sock: socket.socket, cli_id: str):
        got_weight = False
        got_bias = False
        try:
            while not(got_weight and got_bias):
                # 接收 prefix ，以區分 weight 和 bias
                prefix = cli_sock.recv(7).decode()  # 最長 prefix 是 "weight:"，所以接收 7 個字節
                print("Got prefix", prefix, len(prefix))
                if not prefix:
                    continue

                # 取得傳來的參數
                params_size = int.from_bytes(cli_sock.recv(4), 'big')
                params_data = utils.receive_chunked_data(cli_sock)
                params = pickle.loads(params_data)

                if prefix.startswith('weight:'):
                    print(f"Received weight parameters (len: {params_size}) from {cli_id}.")
                    # ckks_params = ts.ckks_vector_from(, params)
                    self.client_weights.append(params) # 加入等待聚合的 weight list 中
                    cli_sock.sendall(b'ACK_W')
                    got_weight = True
                elif prefix.startswith('bias:'):
                    print(f"Received bias parameters (len: {params_size}) from {cli_id}.")
                    self.client_biases.append(params) # 加入等待聚合的 bias list 中
                    cli_sock.sendall(b'ACK_B')
                    got_bias = True
                else:
                    print("Unknown prefix:", prefix)
        
        except Exception as e:
            print(f"Error handling client {cli_id}: {e}")

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


        # 回合結束後關閉 socket
        self.cli_socket.close()

if __name__ == '__main__':
    with open('config.json', 'r') as config_file:
        config = json.load(config_file)
    
    server_config = config['server']
    train_config = config['train_config']

    server = Server(server_config['client_host'], server_config['client_port'],
                    train_config['num_rounds'], train_config['num_clients'])

    server.start()