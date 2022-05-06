import cv2
import pycuda.autoinit
import socket
import numpy as np
# from src.inference import Runner
from segmentation.container.src.inference import Runner


def main():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("0.0.0.0", 5050))
    sock.listen()

    model = Runner('./segmentation/tensorrt/assets/seg_int8.engine')

    # Accept client.
    print('Waiting for connections...')
    client, addr = sock.accept()
    print('Connection received')

    data = np.zeros([480, 640, 4], dtype=np.uint16)

    while True:
        client.recv_into(data.data, data.nbytes)

        rgb, depth = data[..., 0:3].astype(np.uint8), data[..., 3]

        mask = model(rgb)[0]

        if np.any(mask == 1):
            print('Object detected')


if __name__ == '__main__':
    main()