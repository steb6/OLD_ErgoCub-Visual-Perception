import numpy as np

from denoising.src.denoise import denoise
import atexit
import gc
import io
import pickle
import time
import socket


def main():
    print('Connecting to process...')
    while True:
        try:  # moved this line here
            out_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            out_sock.connect(("172.18.128.1", 5052))  # no longer throws error - 172.30.160.1
            break
        except socket.error:
            pass
    print('Connected to process')

    in_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    in_sock.bind(("0.0.0.0", 5051))
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

        downsampled_pc = data['pc']

        if downsampled_pc is not None:
            denoised_pc = denoise(downsampled_pc)
            if denoised_pc.shape[0] > 2024:
                idx = np.random.choice(denoised_pc.shape[0], 2024, replace=False)
                size_pc = denoised_pc[idx]
            else:
                print('Info: Partial Point Cloud padded')
                diff = 2024 - denoised_pc.shape[0]
                pad = np.zeros([diff, 3])
                pad[:] = denoised_pc[0]
                size_pc = np.vstack((denoised_pc, pad))

            mean = np.mean(size_pc, axis=0)
            var = np.sqrt(np.max(np.sum((size_pc - mean) ** 2, axis=1)))
            normalized_pc = (size_pc - mean) / (var * 2)
            normalized_pc[..., -1] = -normalized_pc[..., -1]

            data['normalized_pc'] = normalized_pc
        else:
            data['normalized_pc'] = None

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
