import gc
import pickle
import socket
import cv2

from utils.input import RealSense



def main():
    ip = "127.0.0.1"
    port = 5050

    print('Connecting to process...')
    while True:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((ip, port)) #no longer throws error
            break
        except socket.error:
            pass
    print('Connected to process')

    # camera = cv2.VideoCapture(0)
    camera = RealSense()

    while True:
        # _, image = camera.read()

        data = {'image': image,
                'depth': depth,
                'index': 0}

        image, depth = camera.read()

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        gc.disable()
        out = pickle.dumps(data, protocol=-1)
        gc.enable()

        sock.sendall(len(out).to_bytes(24, 'big') + out)

        cv2.imshow('test1', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)


if __name__ == '__main__':
    main()

