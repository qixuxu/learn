import socket
import cv2
import numpy as np

# 服务器地址和端口
server_address = ('localhost', 8888)

# 创建套接字
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect(server_address)

try:
    while True:
        # 接收图像长度
        length_bytes = client_socket.recv(4)
        length = int.from_bytes(length_bytes, byteorder='big')

        # 接收图像数据
        data = b""
        while len(data) < length:
            packet = client_socket.recv(length - len(data))
            if not packet:
                break
            data += packet

        # 将接收到的数据转换为图像
        nparr = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # 显示图像
        cv2.imshow('Drone Camera', img)

        # 按 'q' 键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # 关闭套接字和窗口
    client_socket.close()
    cv2.destroyAllWindows()