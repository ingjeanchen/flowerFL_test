import logging
import time
import tenseal as ts
import socket
import pickle
import utils
import threading
import json

class KeyGenCenter():
    def __init__(self, serv_host: str, serv_port: int, cli_host: str, cli_port: int, num_rounds: int, num_clients: int):
        self.serv_host = serv_host
        self.serv_port = serv_port
        self.cli_host = cli_host
        self.cli_port = cli_port

        self.num_rounds = num_rounds
        self.num_clients = num_clients
        self.cli_sockets = {}
        self.server_params = {'weight': None, 'bias': None}
        self.sock = None
        self.clients_completed = 0
        self.server_completed = False

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
        return context

    def send_public_context(self, conn: socket.socket, cli_id: str):
        ctx = self.context
        ctx_data = ctx.serialize(save_public_key=True)
        ctx_size = len(ctx_data)
        print(f"Sending context size: {ctx_size}...")
        conn.send(ctx_size.to_bytes(4, 'big'))
        print(f"Sending context to {cli_id}...")

        conn.sendall(ctx_data)
        ack = conn.recv(16)
        ack = ack.decode()
        if(ack == 'Failed'):
            raise("send context failed")
        else:
            if cli_id == 'server':
                print("Server received the context")
            else:
                print(f"Client {cli_id} received the context")
            return 1
        
    def distribute_context(self):
        print("Distributing context...")
        # Create new context for the round
        self.context = self.create_context()

        # Send context to all clients
        for client_id, socket in self.cli_sockets.items():
            self.send_public_context(socket, client_id)

        # Send context to the server
        self.send_public_context(self.server_conn, 'server')

    def send_updated_params_to_clients(self):
        for client_id, socket in self.cli_sockets.items():

            print(f"Sending updated params to Client {client_id}...", self.server_params)
            utils.send_updates(socket, 'kgc', self.server_params['weight'], 
                               self.server_params['bias'], to_encrypt=False, cli_id=client_id)        
    
    def reset_round(self):
        self.clients_completed = 0
        self.server_completed = False

    def start_rounds(self):
        for round_number in range(1, self.num_rounds + 1):
            print(f"Round {round_number} started")
            self.reset_round()

            self.distribute_context()

            while len(self.cli_sockets) < self.num_clients:
                time.sleep(0.5)
            
            utils.receive_parameters(self.server_conn, 'server', context=self.context, server_params=self.server_params)

            self.send_updated_params_to_clients()
            
            print(f"Round {round_number} completed")

    def start_client_listener(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.bind((self.cli_host, self.cli_port))
        self.sock.listen(self.num_clients)

        print("Key Generation Center (KGC) is running...")
        print("Waiting for clients to connect...")

        while len(self.cli_sockets) < self.num_clients:
            cli_socket, _ = self.sock.accept()
            client_id = cli_socket.recv(32)
            client_id = pickle.loads(client_id)
            self.cli_sockets[client_id] = cli_socket
            print(f"Client ID [{client_id}] connected.")

    def start_server_listener(self):
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((self.serv_host, self.serv_port))
        self.server_socket.listen(self.num_clients)

        print("KGC listening for Server connection...")

        self.server_conn, _ = self.server_socket.accept()
        print("Server connected!")

    def start(self):
        self.start_server_listener()
        self.start_client_listener()

        self.start_rounds()

        # Stopping the Key Gen Center
        self.stop()

    def stop(self):
        self.server_socket.close()
        self.sock.close()
        for _, sock in self.cli_sockets.items():
            sock.close()

if __name__ == '__main__':
    with open('config.json', 'r') as config_file:
        config = json.load(config_file)
    
    kgc_config = config['kgc']
    train_config = config['train_config']

    kgc = KeyGenCenter(kgc_config['server_host'], kgc_config['server_port'],
                       kgc_config['client_host'], kgc_config['client_port'],
                       train_config['num_rounds'], train_config['num_clients'])
    kgc.start()