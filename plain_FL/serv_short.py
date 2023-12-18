import socket

cli_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
cli_socket.bind(('localhost', 8057))
cli_socket.listen()

conn, _ = cli_socket.accept()

buf = conn.recv(1024)

print(buf)