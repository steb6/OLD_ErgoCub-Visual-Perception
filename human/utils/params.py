class MetrabsTRTConfig(object):
    def __init__(self):
        self.yolo_engine_path = 'human/engines/yolo.engine'
        self.image_transformation_path = 'human/engines/image_transformation1.engine'
        self.bbone_engine_path = 'human/engines/bbone1.engine'
        self.heads_engine_path = 'human/engines/heads1.engine'
        self.expand_joints_path = 'human/assets/32_to_122.npy'
        self.skeleton_types_path = 'human/assets/skeleton_types.pkl'
        self.skeleton = 'smpl+head_30'
        self.yolo_thresh = 0.3
        self.nms_thresh = 0.7
        self.num_aug = 0  # if zero, disables test time augmentation


class RealSenseIntrinsics(object):
    def __init__(self):
        self.fx = 384.025146484375
        self.fy = 384.025146484375
        self.ppx = 319.09661865234375
        self.ppy = 237.75723266601562
        self.width = 640
        self.height = 480
        import numpy as np
        self.K = np.array([[self.fx, 0, self.ppx],
                           [0, self.fy, self.ppy],
                           [0, 0, 1]])


class TRXConfig(object):
    def __init__(self):
        self.trt_path = 'human/engines/trx.engine'
        self.way = 5
        self.seq_len = 16
        self.n_joints = 30
        self.device = 'cuda'


class FocusConfig:
    class FocusModelConfig:
        def __init__(self):
            self.name = 'resnet18'

    class FaceDetectorConfig:
        def __init__(self):
            self.mode = 'mediapipe'
            self.mediapipe_max_num_faces = 1
            self.mediapipe_static_image_mode = False

    class GazeEstimatorConfig:
        def __init__(self):
            self.camera_params = 'human/assets/camera_params.yaml'
            self.normalized_camera_params = 'human/assets/ptgaze/data/normalized_camera_params/eth-xgaze.yaml'
            self.normalized_camera_distance = 0.6
            self.checkpoint = 'human/engines/eth-xgaze_resnet18.pth'
            self.image_size = [224, 224]

    def __init__(self):
        # GAZE ESTIMATION
        self.face_detector = FocusConfig.FaceDetectorConfig()
        self.gaze_estimator = FocusConfig.GazeEstimatorConfig()
        self.model = FocusConfig.FocusModelConfig()
        self.mode = 'ETH-XGaze'
        self.device = 'cuda'
        self.area_thr = 0.03  # head bounding box must be over this value to be close
        self.close_thr = -0.95  # When close, z value over this thr is considered focus
        self.dist_thr = 0.3  # when distant, roll under this thr is considered focus
        self.foc_rot_thr = 0.7  # when close, roll above this thr is considered not focus
        self.patience = 1  # result is based on the majority of previous observations
        self.sample_params_path = "human/assets/ptgaze/data/calib/sample_params.yaml"
