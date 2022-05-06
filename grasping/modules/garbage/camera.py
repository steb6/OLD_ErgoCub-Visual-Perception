import cv2

if __name__ == '__main__':
    camera = cv2.VideoCapture(0)

    while True:
        _, image = camera.read()
        cv2.imshow('', image)
        cv2.waitKey(1)