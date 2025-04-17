# hofnet version 2.0.0
import os
from hofnet import __root_dir__

DEFAULT_PRETRAIN_MODEL_PATH = os.path.join(
    __root_dir__, "database/"
)

DEFAULT_hofnet_PATH = os.path.join(__root_dir__, "database/hofnet.ckpt")
DEFAULT_PMTRANSFORMER_PATH = os.path.join(__root_dir__, "database/pmtransformer.ckpt")

DEFAULT_FINETUNED_MODEL_PATH = os.path.join(__root_dir__, "database/finetuned/")
DEFAULT_COREMOF_PATH = os.path.join(__root_dir__, "database/coremof/")
DEFAULT_QMOF_PATH = os.path.join(__root_dir__, "database/qmof/")
DEFAULT_HMOF_PATH = os.path.join(__root_dir__, "database/hmof/")
