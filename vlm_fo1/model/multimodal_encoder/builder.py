# Builders for different vision tower backbones (MM encoder visual modules)
# Tower imports happen inside the dispatch branches: the Qwen2.5 wrapper pulls
# in the vendored transformers-4.50 code and the Qwen3.5 wrapper needs
# transformers >= 5.2, so importing both at module scope would tie the whole
# package to one transformers generation.

def build_vision_tower(vision_tower_cfg, **kwargs):
    """
    Use model config to construct the main vision tower.

    vision_tower_cfg: should have attribute mm_vision_tower
    Returns: instance of configured vision backbone
    """
    vision_tower_name = getattr(vision_tower_cfg, 'mm_vision_tower', None)

    # Check for the Qwen3.5 vision model first (its name also contains "qwen")
    if "qwen3.5" in vision_tower_name.lower() or "qwen3_5" in vision_tower_name.lower():
        from .qwen3_5_encoder import Qwen3_5VisionTower
        return Qwen3_5VisionTower(vision_tower_name, args=vision_tower_cfg, **kwargs)

    # Check for the Qwen2.5-VL vision model in tower name
    if "qwen2.5-vl" in vision_tower_name.lower():
        from .qwen2_5_vl_encoder import Qwen2_5_VlVisionTower
        return Qwen2_5_VlVisionTower(vision_tower_name, args=vision_tower_cfg, **kwargs)

    # Raise a clear error for unknown towers
    raise ValueError(f'Unknown vision tower: {vision_tower_name}')

def build_vision_tower_aux(vision_tower_cfg, **kwargs):
    """
    Use model config to construct the auxiliary (helper) vision tower.

    vision_tower_cfg: should have attribute mm_vision_tower_aux
    Returns: instance of configured auxiliary vision backbone
    """
    vision_tower_aux = getattr(vision_tower_cfg, 'mm_vision_tower_aux', None)

    # Check for the DaViT auxiliary vision model in tower name
    if 'davit' in vision_tower_aux.lower():
        from .davit_aux_encoder import DavitVisionTower as DavitVisionTowerAux
        return DavitVisionTowerAux(vision_tower_aux, args=vision_tower_cfg, **kwargs)

    # Raise a clear error if tower type is unknown
    raise ValueError(f'Unknown aux vision tower: {vision_tower_aux}')
