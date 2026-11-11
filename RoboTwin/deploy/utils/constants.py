import numpy as np

# imagenet statistics for image normalization
IMG_MEAN = np.array([0.485, 0.456, 0.406])
IMG_STD = np.array([0.229, 0.224, 0.225])

HOME_POSE = [-0.024598978459835052, 0.4009386897087097, 0.3017459809780121, -3.1358962059020996, -0.00013843314081896096, 1.5719070434570312]

ROBOT_IP = "10.12.41.48"
LOCAL_IP = "10.12.41.219"
ROBOT_PORT = 7003
FORCE_PORT = 7004

# tcp normalization and gripper width normalization
TRANS_MIN, TRANS_MAX = np.array([-0.5, -0.5, 0]), np.array([0.5, 0.5, 1.0]) 
MAX_GRIPPER_WIDTH = 0.11 # meter

# workspace in camera coordinate
# WORKSPACE_MIN = np.array([-2, -2, -2])
# WORKSPACE_MAX = np.array([2, 2, 2])
#WORKSPACE_MIN = np.array([-0.6, -0.2, -2])
#WORKSPACE_MAX = np.array([0.3, 1, 2])
WORKSPACE_MIN = np.array([-0.32, -0.34, 0.16])
WORKSPACE_MAX = np.array([ 0.34,  0.18, 0.62])
# safe workspace in base coordinate
SAFE_EPS = 0.002
SAFE_WORKSPACE_MIN = np.array([-0.4, 0.1, 0.0])
SAFE_WORKSPACE_MAX = np.array([0.4, 0.48, 0.7])

# gripper threshold (to avoid gripper action too frequently)
GRIPPER_THRESHOLD = 0.02 # meter
