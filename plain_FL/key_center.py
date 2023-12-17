import logging
import time
import tenseal as ts
import socket
import pickle
import threading

class KeyGenCenter():
    def __init__(self):
        self.context = None
        self.public_key = None
        self.secret_key = None
        self.cli_sockets = {}
        self.lock = threading.Lock()
        self.sock = None

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
        self.context = context
        self.public_key = context.public_key()
        self.secret_key = context.secret_key()

    def send_public_context(self, conn: socket.socket, cli_id: str):
        ctx = self.context
        ctx_data = ctx.serialize(save_public_key=True)
        ctx_size = len(ctx_data)
        print(f"Sending context size: {ctx_size}...")
        conn.send(ctx_size.to_bytes(4, 'big'))
        print(f"Sending context to {cli_id}...")
        print("Context sent!")
        conn.sendall(ctx_data)
        ack = conn.recv(16)
        ack = ack.decode()
        if(ack == 'Failed'):
            raise("send context failed")
        else:
            print(f"Client {cli_id} received the context")

    def handle_client(self, cli_socket):
        with self.lock:
            try:
                # 得到 client 的 id
                client_id = cli_socket.recv(32)
                client_id = pickle.loads(client_id)
                self.cli_sockets[client_id] = cli_socket
                print(f"Client ID [{client_id}] connected.")

                self.send_public_context(cli_socket, client_id)
            except EOFError as e:
                logging.error(f"EOFError: {e}")
            except Exception as e:
                logging.error(f"Error in handle_client: {e}")

    def handle_server(self, serv_sock: socket.socket):
        # 接收 prefix ，以區分 weight 和 bias
        prefix = serv_sock.recv(7)  # 最長 prefix 是 "weight:"，所以接收 7 個字節
        prefix = prefix.decode()

        if prefix == 'weight:':
            # 取得傳來的 weight 參數
            params_size = serv_sock.recv(4)
            params_size = int.from_bytes(params_size, 'big')
            params_data = serv_sock.recv(params_size)
            weight_params = pickle.loads(params_data)

            # 解密 weight 參數
            weight_params.decrypt()
            print("Received weight parameters:", weight_params)
            # TODO: 將參數除以 client 數量

        elif prefix == 'bias:':
            # 取得傳來的 bias 參數
            params_size = serv_sock.recv(4)
            params_size = int.from_bytes(params_size, 'big')
            params_data = serv_sock.recv(params_size)
            bias_params = pickle.loads(params_data)

            # 解密 bias 參數
            bias_params.decrypt()
            print("Received bias parameters:", bias_params)
            # TODO: 將參數除以 client 數量
            
        else:
            print("Unknown prefix:", prefix)

    def start_client_listener(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.bind(('localhost', 8081))
        self.sock.listen()

        print("Key Generation Center (KGC) is running...")
        print("Waiting for clients to connect...")

        while True:
            cli_socket, _ = self.sock.accept()            

            # 使用多線程處理客戶端的連接
            cli_thread = threading.Thread(target=self.handle_client, args=(cli_socket,))
            cli_thread.start()

    def start_server_listener(self):
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind(('localhost', 8082))
        self.server_socket.listen()

        print("KGC listening for Server connection...")

        while True:
            server_socket, _ = self.server_socket.accept()
            print("Server connected!")
            serv_thread = threading.Thread(target=self.handle_server, args=(server_socket,))
            serv_thread.start()

    def stop(self):
        self.server_socket.close()
        self.sock.close()
        for _, sock in self.cli_sockets.items():
            sock.close()

if __name__ == '__main__':
    kgc = KeyGenCenter()
    kgc.create_context()

    # 用 thread 啟動 Client 和 Server 平行監聽
    client_listener_thread = threading.Thread(target=kgc.start_client_listener)
    server_listener_thread = threading.Thread(target=kgc.start_server_listener)

    client_listener_thread.start()
    server_listener_thread.start()
    
    num_rounds = 10 # TODO: 需要再用 config 設定
    num_clients = 1 # TODO: 需要再用 config 設定

    for round in range(num_rounds):
        print(f"Round {round + 1}")
        kgc.create_context()  # Generate new keys for each round
        time.sleep(1)  # Time for clients to connect and receive the context

    # 結束 Key Gen Center
    kgc.stop()

    client_listener_thread.join()
    server_listener_thread.join()