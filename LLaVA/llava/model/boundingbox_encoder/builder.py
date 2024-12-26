import os
from .boundingbox_encoder import CLIPBoundingBoxEncoder

def build_bbox_tower(bb_encoder_cfg, **kwargs):
    bb_input_dim = getattr(bb_encoder_cfg, "bb_input_dim", 35)
    bb_output_dim = getattr(bb_encoder_cfg, "bb_output_dim", 1024)
    output_dim = getattr(bb_encoder_cfg, "hidden_size", 4096)   
    bb_projector_type = getattr(bb_encoder_cfg, "bb_projector_type", "linear")
    return CLIPBoundingBoxEncoder(bb_input_dim, output_dim, hidden_dim=bb_output_dim, bb_projector_type=bb_projector_type, **kwargs)
