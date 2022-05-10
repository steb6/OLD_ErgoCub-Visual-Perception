import subprocess

import numpy as np

from denoising.src.denoise import denoise
import atexit
import gc
import io
import pickle
import time
import socket


def get_ip():

    cmd1 = ['cat', '/etc/resolv.conf']
    cmd2 = ['grep', 'nameserver']

    ps1 = subprocess.Popen(cmd1, stdout=subprocess.PIPE)
    ps2 = subprocess.Popen(cmd2, stdin=ps1.stdout, stdout=subprocess.PIPE)

    output = ps2.communicate()[0]

    ip = output.strip()[11:].decode("utf-8")

    return ip

def main():
    print('Started')
    print('Connecting to process...')
    while True:
        try:  # moved this line here
            out_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            out_sock.connect((get_ip(), 5052))  # no longer throws error - 172.30.160.1
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

        downsampled_pc = data['downsampled_pc']

        if downsampled_pc is not None:
            denoised_pc = denoise(downsampled_pc)

            data['denoised_pc'] = denoised_pc

        gc.disable()
        out = pickle.dumps(data)
        gc.enable()

        out_sock.sendall(len(out).to_bytes(24, 'big') + out)

        fps = 1 / (time.perf_counter() - start)
        avg += (fps - avg) / i
        i += 1
        print(f'fps={avg:.2f} count={i}', end='\r')


if __name__ == '__main__':
    main()
