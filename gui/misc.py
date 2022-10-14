import copy
import cv2
import numpy as np


def project_pc(points, k=None):
    if k is None:
        intrinsics = {'fx': 612.7910766601562, 'fy': 611.8779296875, 'ppx': 321.7364196777344,
                      'ppy': 245.0658416748047,
                      'width': 640, 'height': 480}

        k = np.eye(3)
        k[0, :] = np.array([intrinsics['fx'], 0, intrinsics['ppx']])
        k[1, 1:] = np.array([intrinsics['fy'], intrinsics['ppy']])

    points = np.array(points)
    uv = k @ points.T
    uv = uv[0:2] / uv[2, :]

    uv = np.round(uv, 0).astype(int)

    return uv.T


def project_hands(rgb, right_t, left_t):
    right_hand = np.concatenate([np.zeros([1, 3]), np.eye(3)])
    left_hand = np.concatenate([np.zeros([1, 3]), np.eye(3)])

    right_hand = (np.block([right_hand, np.ones([4, 1])]) @ right_t)[:, :3]
    left_hand = (np.block([left_hand, np.ones([4, 1])]) @ left_t)[:, :3]

    points2d = project_pc(right_hand)

    res = copy.deepcopy(rgb)
    for i in range(3):
        res = cv2.line(res, points2d[0], points2d[i + 1], color=np.eye(3)[i] * 255, thickness=10)

    points2d = project_pc(left_hand)
    for i in range(3):
        res = cv2.line(res, points2d[0], points2d[i + 1], color=np.eye(3)[i] * 255, thickness=10)

    res = cv2.addWeighted(rgb, 0.7, res, 0.3, 0)
    # res = cv2.cvtColor(res, cv2.COLOR_RGB2BGR)

    return res
