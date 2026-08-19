LOGDIR = "."

global DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
# Model Constants
IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200 #151656 #151655 #-200
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"

# For Qwen2_5_VL
QWEN2_5_VL_IMAGE_TOKEN = "<|image_pad|>"
QWEN2_5_VL_IMAGE_TOKEN_INDEX = 151655

# For Qwen3_5. The image pad token keeps the same surface form, but never
# trust inherited ids across tokenizer generations: the real id is read from
# the checkpoint config (config.image_token_id) at the splice, and chat ids
# are looked up from the tokenizer in mm_utils.
QWEN3_5_IMAGE_TOKEN = "<|image_pad|>"

# For regions
DEFAULT_REGION_TOKEN = "<region<i>>"
DEFAULT_REGION_FEATURE_TOKEN = "<regionfeat>"
DEFAULT_REGION_INDEX = -300 #151654 #151654 #-300

# For Grounding
DEFAULT_GROUNDING_START = "<ground>"
DEFAULT_GROUNDING_END = "</ground>"
DEFAULT_GROUNDING_OBJECTS_START = "<objects>"
DEFAULT_GROUNDING_OBJECTS_END = "</objects>"

# For Think
DEFAULT_THINK_START = "<think>"
DEFAULT_THINK_END = "</think>"
