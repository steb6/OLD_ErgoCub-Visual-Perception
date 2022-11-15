from logging import INFO
import os
from ISBFSAR.modules.hpe.hpe import HumanPoseEstimator
from utils.confort import BaseConfig
import platform


input_type = "skeleton"  # rgb, skeleton or hybrid
docker = os.environ.get('AM_I_IN_A_DOCKER_CONTAINER', False)
seq_len = 8 if input_type != "skeleton" else 16
ubuntu = platform.system() == "Linux"


class Logging(BaseConfig):
    class Logger:
        class Params:
            level = INFO  # Minimum logging level or list of logging levels
            recurring = False

    debug = True
    # options: rgb depth mask 'fps center hands partial scene reconstruction transform
    keys = ['rgb', 'hands', 'mask', 'fps', 'reconstruction', 'planes', 'lines', 'vertices']


class Network(BaseConfig):
    ip = 'host.docker.internal'  # 'host.docker.internal'
    port = 50000
    in_queue = 'source_human'
    out_queues = ['sink']
    # make the output queue blocking (can be used to put a breakpoint in the sink and debug the process output)
    blocking = False


class HumanPoseEstimation(BaseConfig):
    model = HumanPoseEstimator

    class Args:
        base_dir = "ISBFSAR"
        engine_dir = "engines" if not docker else os.path.join("engines", "docker")
        yolo_engine_path = os.path.join(base_dir, 'modules', 'hpe', 'weights', engine_dir, 'yolo.engine')
        image_transformation_path = os.path.join(base_dir, 'modules', 'hpe', 'weights', engine_dir,
                                                 'image_transformation1.engine')
        bbone_engine_path = os.path.join(base_dir, 'modules', 'hpe', 'weights', engine_dir, 'bbone1.engine')
        heads_engine_path = os.path.join(base_dir, 'modules', 'hpe', 'weights', engine_dir, 'heads1.engine')
        expand_joints_path = os.path.join(base_dir, "assets", '32_to_122.npy')
        skeleton_types_path = os.path.join(base_dir, "assets", "skeleton_types.pkl")
        skeleton = 'smpl+head_30'
        yolo_thresh = 0.3
        nms_thresh = 0.7
        num_aug = 0  # if zero, disables test time augmentation
        just_box = input_type == "rgb"


class MainConfig(object):
    def __init__(self):
        self.input_type = input_type  # rgb or skeleton
        self.cam = "realsense"  # webcam or realsense
        self.cam_width = 640
        self.cam_height = 480
        self.window_size = seq_len
        self.skeleton_scale = 2200.
        self.acquisition_time = 3  # Seconds


class RealSenseIntrinsics(object):
    def __init__(self):
        self.fx = 384.025146484375
        self.fy = 384.025146484375
        self.ppx = 319.09661865234375
        self.ppy = 237.75723266601562
        self.width = 640
        self.height = 480


class TRXConfig(object):
    def __init__(self):
        # MAIN
        self.model = "DISC"  # DISC or EXP
        self.input_type = input_type  # skeleton or rgb
        self.way = 5
        self.shot = 1
        self.device = 'cuda'
        self.skeleton_type = skeleton_type

        # CHOICE DATASET
        data_name = "NTURGBD_to_YOLO_METRO_122"
        self.data_path = f"D:\\datasets\\{data_name}" if not ubuntu else f"../datasets/{data_name}"
        self.n_joints = 30

        # TRAINING
        self.initial_lr = 1e-2 if self.input_type == "skeleton" else 3e-4
        self.n_task = (100 if self.input_type == "skeleton" else 30) if not ubuntu else (10000 if self.input_type == "skeleton" else 500)
        self.optimize_every = 1  # Put to 1 if not used, not 0 or -1!
        self.batch_size = 1 if not ubuntu else (32 if self.input_type == "skeleton" else 4)
        self.n_epochs = 10000
        self.start_discriminator_after_epoch = 0  # self.n_epochs  # TODO CAREFUL
        self.first_mile = self.n_epochs  # 15 TODO CAREFUL
        self.second_mile = self.n_epochs  # 1500 TODO CAREFUL
        self.n_workers = 0 if not ubuntu else 16
        self.log_every = 10 if not ubuntu else 1000
        self.eval_every_n_epoch = 10

        # MODEL
        self.trans_linear_in_dim = 256 if self.input_type == "skeleton" else 1000 if self.input_type == "rgb" else 512
        self.trans_linear_out_dim = 128
        self.query_per_class = 1
        self.trans_dropout = 0.
        self.num_gpus = 4
        self.temp_set = [2]
        self.checkpoints_path = "checkpoints"

        # DEPLOYMENT
        if input_type == "rgb":
            self.final_ckpt_path = os.path.join(base_dir, "modules", "ar", "modules", "raws", "rgb", "3000.pth")
        elif input_type == "skeleton":
            self.final_ckpt_path = os.path.join(base_dir, "modules", "ar", "modules", "raws", "DISC.pth")
        elif input_type == "hybrid":
            self.final_ckpt_path = os.path.join(base_dir, "modules", "ar", "modules", "raws", "hybrid",
                                                "1714_truncated_resnet.pth")
        self.trt_path = os.path.join(base_dir, "modules", "ar", engine_dir, "trx.engine")
        self.seq_len = seq_len


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
        self.camera_params = os.path.join(base_dir, "assets", "camera_params.yaml")
        self.normalized_camera_params = os.path.join(base_dir, 'assets', 'eth-xgaze.yaml')
        self.normalized_camera_distance = 0.6
        self.checkpoint = os.path.join(base_dir, 'modules', 'focus', 'gaze_estimation', 'modules', 'raw', 'eth-xgaze_resnet18.pth')
        self.image_size = [224, 224]


class FocusConfig:
    def __init__(self):
        # GAZE ESTIMATION
        self.face_detector = FaceDetectorConfig()
        self.gaze_estimator = GazeEstimatorConfig()
        self.model = FocusModelConfig()
        self.mode = 'ETH-XGaze'
        self.device = 'cuda'
        self.area_thr = 0.03  # head bounding box must be over this value to be close
        self.close_thr = -0.95  # When close, z value over this thr is considered focus
        self.dist_thr = 0.3  # when distant, roll under this thr is considered focus
        self.foc_rot_thr = 0.7  # when close, roll above this thr is considered not focus
        self.patience = 3  # result is based on the majority of previous observations
        self.sample_params_path = os.path.join(base_dir, "assets", "sample_params.yaml")


class MutualGazeConfig:
    def __init__(self):
        self.data_path = 'D:/datasets/mutualGaze_dataset' if not ubuntu else "/home/IIT.LOCAL/sberti/datasets/mutualGaze_dataset"
        self.head_model = os.path.join(base_dir, "modules", "focus", "mutual_gaze", "head_detection", "epoch_0.pth")
        self.focus_model = os.path.join(base_dir, "modules", "focus", "mutual_gaze", "focus_detection", "checkpoints", "sess_0_f1_1.00.pth")
        self.ckpts_path = os.path.join(base_dir, "modules", "focus", "mutual_gaze", "focus_detection", "checkpoints")

        self.augmentation_size = 0.8
        self.dataset = "focus_dataset_heads"
        self.model = "facenet"  # facenet, resnet

        self.batch_size = 8
        self.lr = 1e-6
        self.log_every = 10
        self.pretrained = True
        self.n_epochs = 1000


# TODO THEIR ###########################################################################################################


# class Segmentation(BaseConfig):
#     model = FcnSegmentatorTRT
#
#     class Args:
#         engine_path = './grasping/segmentation/fcn/trt/assets/seg_fp16_docker.engine'
#
#
# class Denoiser(BaseConfig):
#     model = DbscanDenoiser
#
#     class Args:
#         # DBSCAN parameters
#         eps = 0.05
#         min_samples = 10
#
#
# class ShapeCompletion(BaseConfig):
#     class Encoder:
#         model = ConfidencePCRDecoderTRT
#
#         class Args:
#             engine_path = 'grasping/shape_completion/confidence_pcr/trt/assets/pcr_docker.engine'
#
#     class Decoder:
#         model = ConfidencePCRDecoder
#
#         class Args:
#             no_points = 10_000
#             steps = 20
#             thr = 0.5
#
#
# class GraspDetection(BaseConfig):
#     model = RansacGraspDetectorTRT
#
#     class Args:
#         engine_path = './grasping/grasp_detection/ransac_gd/trt/assets/ransac200_10000_docker.engine'
#         # RANSAC parameters
#         tolerance = 0.001
#         iterations = 10000
