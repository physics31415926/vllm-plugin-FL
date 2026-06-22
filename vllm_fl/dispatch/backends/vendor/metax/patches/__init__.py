from . import accelerator_compat  # noqa: F401 — must be first: fixes torch.accelerator.empty_cache
from . import fix_standalone_compile  # noqa: F401
from . import fix_compilation_backend  # noqa: F401 — fixes torch._functorch.config.autograd_cache_normalize_inputs missing in torch 2.8
from . import gdn_linear_attn  # noqa: F401 — fixes hasattr dynamo guard crash on torch 2.8
from . import pynccl_wrapper  # noqa: F401
from . import cuda_wrapper  # noqa: F401
from . import utils_patch  # noqa: F401
from . import chunk_delta_h  # noqa: F401
from . import model_executor  # noqa: F401
