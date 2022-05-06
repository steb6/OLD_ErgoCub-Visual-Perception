import atexit
import copy
import io
import pickle
import socket
import cv2
import numpy as np


def draw_mask(rgb, mask):
    overlay = copy.deepcopy(rgb)
    if np.any(mask == 1):
        overlay[mask == 1] = np.array([0, 0, 128])
    res1 = cv2.addWeighted(rgb, 1, overlay, 0.5, 0)

    res2 = copy.deepcopy(rgb)
    res2[mask == 0] = np.array([0, 0, 0])

    return cv2.cvtColor(res1, cv2.COLOR_RGB2BGR), cv2.cvtColor(res2, cv2.COLOR_RGB2BGR)


def main():
    in_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    in_sock.bind(("172.30.160.1", 5051))  # 172.30.160.1
    in_sock.listen()

    # Accept client.
    print('Wainting for connections...')
    client, addr = in_sock.accept()
    print('Connection received')


    print('Connecting to process...')
    while True:
        try:
            out_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            out_sock.connect(("127.0.0.1", 5050)) #no longer throws error
            break
        except socket.error:
            pass
    print('Connected to process')

    def close_socket():
        in_sock.close(), out_sock.close()
    atexit.register(close_socket)

    camera = cv2.VideoCapture(0)

    while True:
        _, image = camera.read()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        data = {'image': image,
                'index': 0}

        out = pickle.dumps(data)
        out_sock.sendall(len(out).to_bytes(24, 'big') + pickle.dumps(data))

        data_length = int.from_bytes(client.recv(24), 'big')
        stream = io.BytesIO()
        while (data_length - stream.tell()) > 0:
            stream.write(client.recv(data_length - stream.tell()))

        data = pickle.loads(stream.getbuffer())

        res1, res2 = draw_mask(image, data['image'])
        cv2.imshow('res1', res1)
        cv2.imshow('res2', res2)
        cv2.waitKey(1)


if __name__ == '__main__':
    main()

