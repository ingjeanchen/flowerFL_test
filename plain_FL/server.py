import socket
import pickle
import threading
import tenseal as ts
import json
import utils
from typing import Dict, List

class Server:
    def __init__(self, cli_host: str, cli_port: int, kgc_host: str, kgc_port: int, num_rounds: int, num_clients: int):
        self.cli_host = cli_host
        self.cli_port = cli_port
        self.kgc_host = kgc_host
        self.kgc_port = kgc_port
        self.num_rounds = num_rounds
        self.num_clients = num_clients

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

    def send_encrypted_updates(self, weight, bias):
        utils.send_updates(self.kgc_socket, 'server', weight, bias, to_encrypt=False)

    def fetch_context(self) -> ts.Context:
        """ 
        Get context size first and receive the public context from kgc. 
        """
        self.context = utils.receive_public_context(self.kgc_socket)

    def handle_client(self, cli_sock: socket.socket, cli_id: str):
        utils.receive_parameters(cli_sock, cli_id, self.client_params, self.context)

    def start(self):
        self.init_sockets()
        self.fetch_context()
        client_count = 0

        for round_num in range(self.num_rounds):
            self.client_params = {'weight': [], 'bias': []}
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
                thread.join()
                client_count += 1

                if client_count == self.num_clients:
                    break

            if client_count == self.num_clients:
                print(f"Round {round_num + 1}: Aggregating parameters...")
                aggregated_weight = self.aggregate_encrypted_parameters(self.client_params['weight'])
                aggregatted_bias = self.aggregate_encrypted_parameters(self.client_params['bias'])
                print(aggregated_weight, aggregatted_bias)
                print(f"Round {round_num + 1}: Aggregation completed.")
                self.send_encrypted_updates(aggregated_weight, aggregatted_bias)
            else:
                print("Not all clients sent their parameters. Skipping aggregation.")
                
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