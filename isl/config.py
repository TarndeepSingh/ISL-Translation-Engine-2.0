"""Single source of truth for shapes and layout. Change here, everything follows."""
T = 32                       # canonical temporal length
IDLE_LABEL = "idle"

# Raw landmark cache layout per frame: (75, 3)
#   [0:33] pose, [33:54] left hand, [54:75] right hand.  Missing -> NaN.
N_RAW = 75
POSE_SLICE = slice(0, 33)
LH_SLICE = slice(33, 54)
RH_SLICE = slice(54, 75)

# Upper-body pose landmarks used (MediaPipe indices 11..22):
# shoulders, elbows, wrists, pinky, index, thumb  (L,R pairs)
POSE_IDX = list(range(11, 23))

# Feature vector layout (170 dims)
#   pose 36 | left hand 63 | right hand 63 | wrist traj 6 | hand-present flags 2
F_POSE = slice(0, 36)
F_LH = slice(36, 99)
F_RH = slice(99, 162)
F_WRIST = slice(162, 168)
F_FLAGS = slice(168, 170)
FEATURE_DIM = 170
