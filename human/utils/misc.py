import numpy as np
import einops


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


def is_within_fov(imcoords, offset=16):
    lower = np.array(offset).astype(np.float32)
    upper = np.array(256 - offset).astype(np.float32)
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


def reconstruct_ref_fullpersp(normalized_2d, coords3d_rel, validity_mask):
    """Reconstructs the reference point location.

    Args:
      normalized_2d: normalized image coordinates of the joints
         (without intrinsics applied), shape [batch_size, n_points, 2]
      coords3d_rel: 3D camera coordinate offsets relative to the unknown reference
         point which we want to reconstruct, shape [batch_size, n_points, 3]
      validity_mask: boolean mask of shape [batch_size, n_points] containing True
         where the point is reliable and should be used in the reconstruction

    Returns:
      The 3D reference point in camera coordinates, shape [batch_size, 3]
    """

    def rms_normalize(x):  # It makes the norm of the vector equal to one
        scale = np.sqrt(np.mean(np.square(x), axis=1))
        normalized = (x[..., 0] / scale)[..., None]
        return scale, normalized

    n_batch = np.shape(normalized_2d)[0]
    n_points = normalized_2d.shape[1]
    eyes = np.tile(np.expand_dims(np.eye(2, 2), 0), [n_batch, n_points, 1])
    scale2d, reshaped2d = rms_normalize(np.reshape(normalized_2d, [-1, n_points * 2, 1]))
    A = np.concatenate([eyes, -reshaped2d], axis=2)
    # A: 1, 0, x1; 0, 1, y1; 1, 0, x2; 0, 1, y2; ... its the A matrix of linear system
    rel_backproj = normalized_2d * coords3d_rel[:, :, 2:] - coords3d_rel[:, :, :2]
    scale_rel_backproj, b = rms_normalize(np.reshape(rel_backproj, [-1, n_points * 2, 1]))

    weights = validity_mask.astype(np.float32) + np.float32(1e-4)
    weights = einops.repeat(weights, 'b j -> b (j c) 1', c=2)

    i = 0  # TODO we just select one element!
    ref = np.linalg.lstsq((A * weights)[i], (b * weights)[i], rcond=None)[0].T
    ref = np.concatenate([ref[:, :2], ref[:, 2:] / scale2d[i]], axis=1) * scale_rel_backproj[i]
    return ref


def reconstruct_absolute(coords2d, coords3d_rel, intrinsics, is_predicted_to_be_in_fov, weak_perspective=False):
    # coords2d = tf.convert_to_tensor(coords2d)
    inv_intrinsics = np.linalg.inv(intrinsics.astype(np.float32))
    coords2d_normalized = (to_homogeneous(coords2d) @ inv_intrinsics.swapaxes(1, 2))[..., :2]
    # coords2d_normalized = np.matmul(
    #     to_homogeneous(coords2d), inv_intrinsics, transpose_b=True)[..., :2]
    reconstruct_ref_fn = reconstruct_ref_fullpersp
    # is_predicted_to_be_in_fov = is_within_fov(coords2d)

    ref = reconstruct_ref_fn(coords2d_normalized, coords3d_rel, is_predicted_to_be_in_fov)

    # Joints that wasn't in FOV
    coords_abs_3d_based = coords3d_rel + np.expand_dims(ref, 1)

    # Joints that was in FOV
    reference_depth = ref[:, 2]
    relative_depths = coords3d_rel[..., 2]
    coords_abs_2d_based = back_project(coords2d_normalized, relative_depths, reference_depth)

    return np.where(
        is_predicted_to_be_in_fov[..., np.newaxis], coords_abs_2d_based, coords_abs_3d_based)


def back_project(camcoords2d, delta_z, z_offset):
    return to_homogeneous(camcoords2d) * np.expand_dims(delta_z + np.expand_dims(z_offset, -1), -1)


def is_pose_consistent_with_box(pose2d, box):
    """Check if pose prediction is consistent with the original box it was based on.
    Concretely, check if the intersection between the pose's bounding box and the detection has
    at least half the area of the detection box. This is like IoU but the denominator is the
    area of the detection box, so that truncated poses are handled correctly.
    """

    # TODO MINE
    pose2d = np.concatenate((np.clip(pose2d[:, 0], 0, 640)[..., None],
                             np.clip(pose2d[:, 1], 0, 480)[..., None]), axis=-1)
    posebox_start = np.concatenate((np.array(np.min(pose2d[:, 0]))[None, ...],
                                    np.array(np.min(pose2d[:, 1]))[None, ...]))
    posebox_end = np.concatenate((np.array(np.max(pose2d[:, 0]))[None, ...],
                                  np.array(np.max(pose2d[:, 1]))[None, ...]))
    box_start = np.array([box[0], box[1]])
    box_end = np.array([box[2], box[3]])
    box_area = (box_end - box_start)[0] * (box_end - box_start)[1]

    intersection_start = np.concatenate((np.array(np.max([box_start[0], posebox_start[0]]))[..., None],
                                        np.array(np.max([box_start[1], posebox_start[1]]))[..., None]))
    intersection_end = np.concatenate((np.array(np.min([box_end[0], posebox_end[0]]))[..., None],
                                      np.array(np.min([box_end[1], posebox_end[1]]))[..., None]))
    intersection_area = (intersection_end - intersection_start)[0] * \
                        (intersection_end - intersection_start)[1]
    return intersection_area > box_area * 0.5
