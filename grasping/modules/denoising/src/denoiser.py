import gc
import io
import pickle
import subprocess
import socket

from loguru import logger

def get_ip():
    cmd = ['wsl', 'cat', '/etc/resolv.conf']
    ps = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    output = ps.communicate()[0]

    ip = output.strip()[185:].decode("utf-8")
    return ip


class Denoising:

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.client = None
        self.out_sock = None
        self.in_sock = None
        self.popen = None

        self.startup()

    def startup(self):
        import os
        from pathlib import Path
        # p = (Path("/mnt/") / Path(os.getcwd()).as_posix().replace("C:", "c") / Path("grasping/denoising.bash")).as_posix()
        # cmd = ['wsl', p]
        # print(cmd)
        cmd = ['wsl', '-d', 'Ubuntu-18.04', '.', '/home/arosasco/grasping/denoise2.sh']
        popen = subprocess.Popen(cmd, stdout=subprocess.PIPE, universal_newlines=True)
        for stdout_line in iter(popen.stdout.readline, ""):
            print(stdout_line)
            if stdout_line.strip() == 'Started':
                break
        self.popen = popen

        logger.info('Waiting incoming connections...')
        in_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        in_sock.bind((get_ip(), 5052))
        in_sock.listen()

        self.in_sock = in_sock
        self.client, _ = in_sock.accept()
        logger.info('Client connected.')

        logger.info('Connecting to server...')
        while True:
            try:
                out_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                out_sock.connect(("127.0.0.1", 5051))
                self.out_sock = out_sock
                break
            except socket.error:
                pass
        logger.info('Server connected.')

    def __call__(self, input_pc):

        out = pickle.dumps({'downsampled_pc': input_pc})
        self.out_sock.sendall(len(out).to_bytes(24, 'big') + out)

        data_length = int.from_bytes(self.client.recv(24), 'big')

        stream = io.BytesIO()
        while (data_length - stream.tell()) > 0:
            stream.write(self.client.recv(data_length - stream.tell()))

        gc.disable()
        inp = pickle.loads(stream.getbuffer())
        gc.enable()

        return inp['denoised_pc']
