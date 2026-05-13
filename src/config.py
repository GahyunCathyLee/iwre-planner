'''
Configuration values for B1: IDM-only baseline.
'''

# Simulation
DEFAULT_MAP = "Town04"
DEFAULT_FPS = 20.0
DEFAULT_STEPS = 500
DEFAULT_NUM_NPC = 10

# Traffic Manager
TRAFFIC_MANAGER_PORT = 8000
GLOBAL_DISTANCE_TO_LEADING_VEHICLE = 2.5

# Ego vehicle
EGO_VEHICLE_FILTER = "vehicle.tesla.model3"

# IDM parameters
IDM_DESIRED_SPEED = 13.9      # m/s, about 50 km/h
IDM_MIN_DISTANCE = 2.0        # s0, minimum gap [m]
IDM_TIME_HEADWAY = 1.5        # T, desired time headway [s]
IDM_MAX_ACCEL = 1.5           # a_max [m/s^2]
IDM_COMFORT_DECEL = 2.0       # b, comfortable braking [m/s^2]
IDM_DELTA = 4.0               # acceleration exponent
IDM_MAX_DETECTION_DISTANCE = 80.0  # front vehicle search range [m]

# Vehicle control conversion
MAX_THROTTLE = 0.75
MAX_BRAKE = 0.8
ACCEL_TO_THROTTLE_SCALE = 1.5
DECEL_TO_BRAKE_SCALE = 4.0

# Logging
LOG_DIR = "outputs/logs"