import pycuda.autoinit
# import pycuda.driver as cuda
# cuda.init()
# cuda.Device(0).make_context()
import atexit
import gc
import io
import pickle
import time
import socket

import torch

from shape_reconstruction.tensorrt.utils.inference import Infer
torch.tensor([0]).cuda()
backbone = Infer('./shape_reconstruction/tensorrt/assets/pcr.engine')


def main():
    print('Connecting to process...')
    while True:
        try:  # moved this line here
            out_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            out_sock.connect(("127.0.0.1", 5053))  # no longer throws error - 172.30.160.1
            break
        except socket.error:
            pass
    print('Connected to process')

    in_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    in_sock.bind(("172.30.160.1", 5052))
    in_sock.listen()

    def close_socket():
        in_sock.close(), out_sock.close()
    atexit.register(close_socket)

    # Accept client.
    print('Waiting for connections...')
    client, addr = in_sock.accept()
    print('Connection received')

    i, avg = 1, 0
    while True:
        start = time.perf_counter()

        data_length = int.from_bytes(client.recv(24), 'big')

        stream = io.BytesIO()
        while (data_length - stream.tell()) > 0:
            stream.write(client.recv(data_length - stream.tell()))

        inp = stream.getbuffer()
        gc.disable()
        data = pickle.loads(inp)
        gc.enable()

        normalized_pc = data['normalized_pc']

        if normalized_pc is not None:
            fast_weights = backbone(normalized_pc)
            data['weights'] = fast_weights
        else:
            data['weights'] = None

        gc.disable()
        out = pickle.dumps(data)
        gc.enable()

        out_sock.sendall(len(out).to_bytes(24, 'big') + out)

        fps = 1 / (time.perf_counter() - start)
        avg += (fps - avg) / i
        i += 1
        print(f'fps={avg:.2f} count={i}', end='\r')

        # print(np.unique(mask))
        # if np.any(mask == 1):
        #     print('Object detected')

if __name__ == '__main__':
    main()
