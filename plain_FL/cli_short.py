import socket


serv_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
serv_socket.connect(('localhost', 8057))

msg = b'hello'
serv_socket.sendall(msg)