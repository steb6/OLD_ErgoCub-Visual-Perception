# import pycuda.driver as cuda
# cuda.init()
# cuda.Device(0).make_context()
import atexit
import gc
import io
import pickle
import time
import socket

from shape_reconstruction.tensorrt.utils.decoder import Decoder
decoder = Decoder()


def main():
    print('Connecting to process...')
    while True:
        try:  # moved this line here
            out_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            out_sock.connect(("127.0.0.1", 5054))  # no longer throws error - 172.30.160.1
            break
        except socket.error:
            pass
    print('Connected to process')

    in_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    in_sock.bind(("127.0.0.1", 5053))
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

        weights = data['weights']

        if weights is not None:
            reconstruction = decoder(weights)
            data['reconstruction'] = reconstruction
        else:
            data['reconstruction'] = None

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
