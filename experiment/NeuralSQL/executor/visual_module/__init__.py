from .vqa_module import VQAModule
from .yes_vqa_module import YesVQAModule
from .m3ae_vqa_module import M3AEVQAModule
from .custom_vqa_module import CustomVQAModule

import os


def get_vqa_module(vqa_module_type: str) -> VQAModule:
    """Get the VQA module based on the specified type."""
    if vqa_module_type == "yes" or vqa_module_type == "debug":  # NOTE: this could be a baseline model
        return YesVQAModule()
    elif vqa_module_type == "m3ae":
        # NOTE: hardcoded model path
        model_path = os.path.join(os.path.dirname(__file__), "./checkpoints/m3ae/seed42.ckpt")
        return M3AEVQAModule(model_path=model_path)
    elif vqa_module_type == "custom":
        return CustomVQAModule()
    else:
        raise NotImplementedError("Only support debug model")
