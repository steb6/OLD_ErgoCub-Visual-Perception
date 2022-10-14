import gc
import io
import pickle
import subprocess
import socket

from loguru import logger

from grasping.modules.denoising.src.denoise import denoise


def get_ip():
    cmd = ['wsl', 'cat', '/etc/resolv.conf']
    ps = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    output = ps.communicate()[0]

    ip = output.strip()[185:].decode("utf-8")
    return ip


class Denoising:

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.startup()

    def startup(self):
        pass

    def __call__(self, input_pc):

        if input_pc is not None:
            denoised_pc = denoise(input_pc)

        return denoised_pc
