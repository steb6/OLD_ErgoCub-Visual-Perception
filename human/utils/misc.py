import numpy as np


def nms_cpu(boxes, confs, nms_thresh=0.7, min_mode=False):
    # print(boxes.shape)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = confs.argsort()[::-1]

    keep = []
    while order.size > 0:
        idx_self = order[0]
        idx_other = order[1:]

        keep.append(idx_self)

        xx1 = np.maximum(x1[idx_self], x1[idx_other])
        yy1 = np.maximum(y1[idx_self], y1[idx_other])
        xx2 = np.minimum(x2[idx_self], x2[idx_other])
        yy2 = np.minimum(y2[idx_self], y2[idx_other])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h

        if min_mode:
            over = inter / np.minimum(areas[order[0]], areas[order[1:]])
        else:
            over = inter / (areas[order[0]] + areas[order[1:]] - inter)

        inds = np.where(over <= nms_thresh)[0]
        order = order[inds + 1]

    return np.array(keep)


def postprocess_yolo_output(boxes, confidences, conf_thresh=0.3, nms_thresh=0.7, num_classes=1):
    # Reshape output
    boxes = boxes.reshape(1, -1, 1, 4)  # 1 4032 1 4
    boxes = boxes[:, :, 0]  # 1 4032 4
    confidences = confidences.reshape(1, -1, 80)  # 1 4032 80

    # Get maximum confidences
    max_conf = np.max(confidences, axis=2)
    max_id = np.argmax(confidences, axis=2)

    # Loop over batches to get results
    bboxes_batch = []
    for i in range(boxes.shape[0]):  # 1

        # Get boxes, confidences and ids of the detection in this image of the batch
        argwhere = max_conf[i] > conf_thresh
        l_box_array = boxes[i, argwhere, :]
        l_max_conf = max_conf[i, argwhere]
        l_max_id = max_id[i, argwhere]

        # Loop over the possible classes
        bboxes = []
        for j in range(num_classes):

            # Get boxes, confidences and ids of the classes in this image of the batch
            cls_argwhere = l_max_id == j
            ll_box_array = l_box_array[cls_argwhere, :]
            ll_max_conf = l_max_conf[cls_argwhere]
            ll_max_id = l_max_id[cls_argwhere]

            # Keep only detection which has certain criterion
            keep = nms_cpu(ll_box_array, ll_max_conf, nms_thresh=nms_thresh)
            if keep.size > 0:
                ll_box_array = ll_box_array[keep, :]
                ll_max_conf = ll_max_conf[keep]
                ll_max_id = ll_max_id[keep]
                for k in range(ll_box_array.shape[0]):
                    bboxes.append(
                        [ll_box_array[k, 0], ll_box_array[k, 1], ll_box_array[k, 2], ll_box_array[k, 3],
                         ll_max_conf[k], ll_max_id[k]])

        bboxes_batch.append(bboxes)
    return bboxes_batch


def to_homogeneous(x):
    return np.concatenate([x, np.ones_like(x[..., :1])], axis=-1)


def is_within_fov(imcoords):
    lower = np.array(18).astype(np.float32)
    upper = np.array(256 - 18).astype(np.float32)
    return np.all(np.logical_and(imcoords >= lower, imcoords <= upper), axis=-1)


def get_new_rotation_matrix(forward_vector, up_vector):
    # Z will point forwards, towards the box center
    new_z = forward_vector / np.linalg.norm(forward_vector, axis=-1, keepdims=True)
    # Get the X (right direction) as the cross of forward and up.
    new_x = np.cross(new_z, up_vector)
    # Get alternative X by rotating the new Z around the old Y by 90 degrees
    # in case lookdir happens to align with the up vector and the above cross product is zero.
    new_x_alt = np.stack([new_z[:, 2], np.zeros_like(new_z[:, 2]), -new_z[:, 0]], axis=1)
    new_x = np.where(np.linalg.norm(new_x, axis=-1, keepdims=True) == 0, new_x_alt, new_x)
    new_x = new_x / np.linalg.norm(new_x, axis=-1, keepdims=True)
    # Complete the right-handed coordinate system to get Y
    new_y = np.cross(new_z, new_x)
    # Stack the axis vectors to get the rotation matrix
    return np.stack([new_x, new_y, new_z], axis=1)


def project(points):
    return points[..., :2] / points[..., 2:3]


def homography(x1, x2, y1, y2, K, out_dim):
    # We get the center of the bound box, the top, right, down and left points
    boxpoints_homog = to_homogeneous(np.array([[[(x1 + x2) / 2, (y1 + y2) / 2],  # 1 5 2
                                                [(x1 + x2) / 2, y1],
                                                [x2, (y1 + y2) / 2],
                                                [(x1 + x2) / 2, y2],
                                                [x1, (y1 + y2) / 2]]]))
    # We use the inverse of the matrix K to project bb points from image space to cam space
    # boxpoints_camspace = np.einsum('bpc,bCc->bpC', boxpoints_homog, np.linalg.inv(K[None, ...])
    boxpoints_camspace = boxpoints_homog @ np.linalg.inv(K[None, ...]).transpose((0, 2, 1))
    # Z values are useless
    boxpoints_camspace = to_homogeneous(boxpoints_camspace[..., :2])
    # Get center
    box_center_camspace = boxpoints_camspace[:, 0]
    # Rotate camera to point the center of the bounding box
    R_noaug = get_new_rotation_matrix(forward_vector=box_center_camspace, up_vector=np.array([[0, -1, 0]]))
    # COMPUTE ZOOM
    # Get lateral points
    sidepoints_camspace = boxpoints_camspace[:, 1:5]
    # Put again the bb point from camspace to image space and apply the rotation that centers the bounding box
    # sidepoints_new = project(np.einsum(
    #     'bpc,bCc->bpC', sidepoints_camspace, K[None, ...] @ R_noaug))
    sidepoints_new = project(sidepoints_camspace @ (K[None, ...] @ R_noaug).transpose((0, 2, 1)))

    # Measure the size of the reprojected boxes
    vertical_size = np.linalg.norm(sidepoints_new[:, 0] - sidepoints_new[:, 2], axis=-1)
    horiz_size = np.linalg.norm(sidepoints_new[:, 1] - sidepoints_new[:, 3], axis=-1)
    box_size_new = np.maximum(vertical_size, horiz_size)

    # How much we need to scale (zoom) to have the boxes fill out the final crop
    box_aug_scales = out_dim / box_size_new

    # This matrix add to the classic K matrix the action to zoom
    new_intrinsic_matrix = np.concatenate([
        np.concatenate([
            # Top-left of original intrinsic matrix gets scaled
            K[:2, :2] * box_aug_scales,
            # Principal point is the middle of the new image size
            np.full((2, 1), out_dim / 2, dtype=K.dtype)], axis=1),
        np.concatenate([
            # [0, 0, 1] as the last row of the intrinsic matrix:
            np.zeros((1, 2), np.float32),
            np.ones((1, 1), np.float32)], axis=1)], axis=0)

    return new_intrinsic_matrix, R_noaug
