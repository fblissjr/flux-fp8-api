import torch
from loguru import logger
from safetensors.torch import load_file
from tqdm import tqdm
from typing import Literal, Optional, Tuple, Dict

try:
    from cublas_ops import CublasLinear
except Exception as e:
    CublasLinear = type(None)
from float8_quantize import F8Linear
from modules.flux_model import Flux


def swap_scale_shift(weight):
    scale, shift = weight.chunk(2, dim=0)
    new_weight = torch.cat([shift, scale], dim=0)
    return new_weight


def check_if_lora_exists(state_dict, lora_name):
    subkey = lora_name.split(".lora_A")[0].split(".lora_B")[0].split(".weight")[0]
    for key in state_dict.keys():
        if subkey in key:
            return subkey
    return False


def convert_if_lora_exists(new_state_dict, state_dict, lora_name, flux_layer_name):
    if (original_stubkey := check_if_lora_exists(state_dict, lora_name)) != False:
        weights_to_pop = [k for k in state_dict.keys() if original_stubkey in k]
        for key in weights_to_pop:
            key_replacement = key.replace(
                original_stubkey, flux_layer_name.replace(".weight", "")
            )
            new_state_dict[key_replacement] = state_dict.pop(key)
        return new_state_dict, state_dict
    else:
        return new_state_dict, state_dict


def convert_diffusers_to_flux_transformer_checkpoint(
    diffusers_state_dict: Dict[str, torch.Tensor],
    num_layers: int,
    num_single_layers: int,
    has_guidance: bool = True,
    prefix: str = "",
) -> Dict[str, torch.Tensor]:
    """
    Convert diffusers-style LoRA weights to Flux Transformer checkpoint format.

    Args:
        diffusers_state_dict (Dict[str, torch.Tensor]): The original diffusers LoRA state dict.
        num_layers (int): Number of transformer layers.
        num_single_layers (int): Number of single transformer layers.
        has_guidance (bool): Whether the model has guidance embedding.
        prefix (str): Prefix for the diffusers keys.

    Returns:
        Dict[str, torch.Tensor]: Converted Flux-style LoRA weights.
    """
    original_state_dict = {}

    # Convert time_text_embed
    original_state_dict, diffusers_state_dict = convert_if_lora_exists(
        original_state_dict,
        diffusers_state_dict,
        f"{prefix}time_text_embed.timestep_embedder.linear_1.weight",
        "time_in.in_layer.weight",
    )
    original_state_dict, diffusers_state_dict = convert_if_lora_exists(
        original_state_dict,
        diffusers_state_dict,
        f"{prefix}time_text_embed.text_embedder.linear_1.weight",
        "vector_in.in_layer.weight",
    )
    original_state_dict, diffusers_state_dict = convert_if_lora_exists(
        original_state_dict,
        diffusers_state_dict,
        f"{prefix}time_text_embed.text_embedder.linear_2.weight",
        "vector_in.out_layer.weight",
    )

    if has_guidance:
        original_state_dict, diffusers_state_dict = convert_if_lora_exists(
            original_state_dict,
            diffusers_state_dict,
            f"{prefix}time_text_embed.guidance_embedder.linear_1.weight",
            "guidance_in.in_layer.weight",
        )
        original_state_dict, diffusers_state_dict = convert_if_lora_exists(
            original_state_dict,
            diffusers_state_dict,
            f"{prefix}time_text_embed.guidance_embedder.linear_2.weight",
            "guidance_in.out_layer.weight",
        )

    # Convert context_embedder and x_embedder
    original_state_dict, diffusers_state_dict = convert_if_lora_exists(
        original_state_dict,
        diffusers_state_dict,
        f"{prefix}context_embedder.weight",
        "txt_in.weight",
    )
    original_state_dict, diffusers_state_dict = convert_if_lora_exists(
        original_state_dict,
        diffusers_state_dict,
        f"{prefix}x_embedder.weight",
        "img_in.weight",
    )

    # Convert transformer blocks
    for i in range(num_layers):
        block_prefix = f"{prefix}transformer_blocks.{i}."
        # Convert norms
        original_state_dict, diffusers_state_dict = convert_if_lora_exists(
            original_state_dict,
            diffusers_state_dict,
            f"{block_prefix}norm1.linear.weight",
            f"double_blocks.{i}.img_mod.lin.weight",
        )

        # Convert Q, K, V
        for component in ["q", "k", "v"]:
            sample_key_A = f"{block_prefix}attn.to_{component}.lora_A.weight"
            sample_key_B = f"{block_prefix}attn.to_{component}.lora_B.weight"
            if (
                sample_key_A in diffusers_state_dict
                and sample_key_B in diffusers_state_dict
            ):
                sample_A = diffusers_state_dict.pop(sample_key_A)
                sample_B = diffusers_state_dict.pop(sample_key_B)
                original_state_dict[f"double_blocks.{i}.img_attn.qkv.lora_A.weight"] = (
                    torch.cat([sample_A, sample_B], dim=0)
                )
                original_state_dict[f"double_blocks.{i}.img_attn.qkv.lora_B.weight"] = (
                    torch.cat([sample_A, sample_B], dim=0)
                )
            else:
                logger.debug(
                    f"Skipping {component} for layer {i} since no LoRA weight is available"
                )

        # Convert context Q, K, V
        for component in ["q", "k", "v"]:
            context_key_A = f"{block_prefix}attn.add_{component}_proj.lora_A.weight"
            context_key_B = f"{block_prefix}attn.add_{component}_proj.lora_B.weight"
            if (
                context_key_A in diffusers_state_dict
                and context_key_B in diffusers_state_dict
            ):
                context_A = diffusers_state_dict.pop(context_key_A)
                context_B = diffusers_state_dict.pop(context_key_B)
                original_state_dict[f"double_blocks.{i}.txt_attn.qkv.lora_A.weight"] = (
                    torch.cat([context_A, context_B], dim=0)
                )
                original_state_dict[f"double_blocks.{i}.txt_attn.qkv.lora_B.weight"] = (
                    torch.cat([context_A, context_B], dim=0)
                )
            else:
                logger.debug(
                    f"Skipping context {component} for layer {i} since no LoRA weight is available"
                )

        # Convert qk_norm
        for norm_type in ["q", "k", "added_q", "added_k"]:
            original_state_dict, diffusers_state_dict = convert_if_lora_exists(
                original_state_dict,
                diffusers_state_dict,
                f"{block_prefix}attn.norm_{norm_type}.weight",
                f"double_blocks.{i}.{'img' if 'added' not in norm_type else 'txt'}_attn.norm.{norm_type.replace('added_', '')}_norm.scale",
            )

        # Convert ff layers
        for ff_type in ["ff", "ff_context"]:
            for layer in [0, 2]:
                original_state_dict, diffusers_state_dict = convert_if_lora_exists(
                    original_state_dict,
                    diffusers_state_dict,
                    f"{block_prefix}{ff_type}.net.{layer}.{'proj.' if layer == 0 else ''}weight",
                    f"double_blocks.{i}.{'img' if ff_type == 'ff' else 'txt'}_mlp.{layer}.weight",
                )

        # Convert output projections
        original_state_dict, diffusers_state_dict = convert_if_lora_exists(
            original_state_dict,
            diffusers_state_dict,
            f"{block_prefix}attn.to_out.0.weight",
            f"double_blocks.{i}.img_attn.proj.weight",
        )
        original_state_dict, diffusers_state_dict = convert_if_lora_exists(
            original_state_dict,
            diffusers_state_dict,
            f"{block_prefix}attn.to_add_out.weight",
            f"double_blocks.{i}.txt_attn.proj.weight",
        )

    # Convert single transformer blocks
    for i in range(num_single_layers):
        block_prefix = f"{prefix}single_transformer_blocks.{i}."
        original_state_dict, diffusers_state_dict = convert_if_lora_exists(
            original_state_dict,
            diffusers_state_dict,
            f"{block_prefix}proj_out.weight",
            f"single_blocks.{i}.linear2.weight",
        )
        original_state_dict, diffusers_state_dict = convert_if_lora_exists(
            original_state_dict,
            diffusers_state_dict,
            f"{block_prefix}proj_out.bias",
            f"single_blocks.{i}.linear2.bias",
        )

        # Convert Q, K, V, mlp for single blocks
        for component in ["q", "k", "v"]:
            original_state_dict, diffusers_state_dict = convert_if_lora_exists(
                original_state_dict,
                diffusers_state_dict,
                f"{block_prefix}attn.to_{component}.lora_A.weight",
                f"single_blocks.{i}.attn.to_{component}.weight",
            )
            original_state_dict, diffusers_state_dict = convert_if_lora_exists(
                original_state_dict,
                diffusers_state_dict,
                f"{block_prefix}attn.to_{component}.lora_B.weight",
                f"single_blocks.{i}.attn.to_{component}.bias",
            )

        original_state_dict, diffusers_state_dict = convert_if_lora_exists(
            original_state_dict,
            diffusers_state_dict,
            f"{block_prefix}proj_mlp.lora_A.weight",
            f"single_blocks.{i}.linear1.weight",
        )
        original_state_dict, diffusers_state_dict = convert_if_lora_exists(
            original_state_dict,
            diffusers_state_dict,
            f"{block_prefix}proj_mlp.lora_B.weight",
            f"single_blocks.{i}.linear1.bias",
        )

    # Convert final layers
    original_state_dict, diffusers_state_dict = convert_if_lora_exists(
        original_state_dict,
        diffusers_state_dict,
        f"{prefix}proj_out.weight",
        "final_layer.linear.weight",
    )
    original_state_dict, diffusers_state_dict = convert_if_lora_exists(
        original_state_dict,
        diffusers_state_dict,
        f"{prefix}proj_out.bias",
        "final_layer.linear.bias",
    )
    original_state_dict, diffusers_state_dict = convert_if_lora_exists(
        original_state_dict,
        diffusers_state_dict,
        f"{prefix}norm_out.linear.weight",
        "final_layer.adaLN_modulation.1.weight",
    )

    if len(list(diffusers_state_dict.keys())) > 0:
        logger.warning(f"Unexpected keys: {diffusers_state_dict.keys()}")

    return original_state_dict


def convert_from_original_flux_checkpoint(
    original_state_dict: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    """
    Convert the state dict from the original Flux checkpoint format to the new format.

    Args:
        original_state_dict (Dict[str, torch.Tensor]): The original Flux checkpoint state dict.

    Returns:
        Dict[str, torch.Tensor]: The converted state dict in the new format.
    """
    sd = {
        k.replace("lora_unet_", "")
        .replace("double_blocks_", "double_blocks.")
        .replace("single_blocks_", "single_blocks.")
        .replace("_img_attn_", ".img_attn.")
        .replace("_txt_attn_", ".txt_attn.")
        .replace("_img_mod_", ".img_mod.")
        .replace("_txt_mod_", ".txt_mod.")
        .replace("_img_mlp_", ".img_mlp.")
        .replace("_txt_mlp_", ".txt_mlp.")
        .replace("_linear1", ".linear1")
        .replace("_linear2", ".linear2")
        .replace("_modulation_", ".modulation.")
        .replace("lora_up", "lora_B")
        .replace("lora_down", "lora_A"): v
        for k, v in original_state_dict.items()
        if "lora" in k
    }
    return sd


def get_module_for_key(
    key: str, model: Flux
) -> F8Linear | torch.nn.Linear | CublasLinear:
    parts = key.split(".")
    module = model
    for part in parts:
        module = getattr(module, part)
    return module


def get_lora_for_key(
    key: str, lora_weights: dict
) -> Optional[Tuple[torch.Tensor, torch.Tensor, Optional[float]]]:
    """
    Get LoRA weights for a specific key.

    Args:
        key (str): The key to look up in the LoRA weights.
        lora_weights (dict): Dictionary containing LoRA weights.

    Returns:
        Optional[Tuple[torch.Tensor, torch.Tensor, Optional[float]]]: A tuple containing lora_A, lora_B, and alpha if found, None otherwise.
    """
    prefix = key.split(".lora")[0]
    lora_A = lora_weights.get(f"{prefix}.lora_A.weight")
    lora_B = lora_weights.get(f"{prefix}.lora_B.weight")
    alpha = lora_weights.get(f"{prefix}.alpha")

    if lora_A is None or lora_B is None:
        return None
    return lora_A, lora_B, alpha


@torch.inference_mode()
def apply_lora_weight_to_module(
    module_weight: torch.Tensor,
    lora_weights: Tuple[torch.Tensor, torch.Tensor, Optional[float]],
    rank: Optional[int] = None,
    lora_scale: float = 1.0,
) -> torch.Tensor:
    """
    Apply LoRA weights to a module's weight tensor.

    Args:
        module_weight (torch.Tensor): The original weight tensor of the module.
        lora_weights (Tuple[torch.Tensor, torch.Tensor, Optional[float]]): LoRA weights (A, B) and alpha.
        rank (Optional[int]): The rank of the LoRA approximation.
        lora_scale (float): Scaling factor for the LoRA weights.

    Returns:
        torch.Tensor: The updated weight tensor after applying LoRA.
    """
    lora_A, lora_B, alpha = lora_weights

    uneven_rank = lora_B.shape[1] != lora_A.shape[0]
    rank_diff = lora_A.shape[0] / lora_B.shape[1]

    if rank is None:
        rank = lora_B.shape[1]
    if alpha is None:
        alpha = rank

    w_dtype = module_weight.dtype
    dtype = torch.float32
    device = module_weight.device
    w_orig = module_weight.to(dtype=dtype, device=device)
    w_up = lora_A.to(dtype=dtype, device=device)
    w_down = lora_B.to(dtype=dtype, device=device)

    if alpha != rank:
        w_up = w_up * alpha / rank
    if uneven_rank:
        fused_lora = lora_scale * torch.mm(
            w_down.repeat_interleave(int(rank_diff), dim=1), w_up
        )
    else:
        fused_lora = lora_scale * torch.mm(w_down, w_up)
    fused_weight = w_orig + fused_lora
    return fused_weight.to(dtype=w_dtype, device=device)


@torch.inference_mode()
def apply_lora_to_model(
    model: "Flux", lora_path: str, lora_scale: float = 1.0, debug: bool = False
) -> "Flux":
    """
    Apply LoRA weights to the Flux model.

    Args:
        model (Flux): The Flux model to apply LoRA weights to.
        lora_path (str): Path to the LoRA weights file.
        lora_scale (float): Scaling factor for the LoRA weights.

    Returns:
        Flux: The Flux model with LoRA weights applied.
    """
    # Set the logging level based on the debug flag
    has_guidance = model.params.guidance_embed
    logger.info(f"Loading LoRA weights for {lora_path}")
    lora_weights = load_file(lora_path)

    logger.debug("Original lora_weights:")
    logger.debug(lora_weights.keys())

    from_original_flux = False
    check_if_starts_with_transformer = [
        k for k in lora_weights.keys() if k.startswith("transformer.")
    ]
    if len(check_if_starts_with_transformer) > 0:
        logger.debug("Using prefix 'transformer.'")
        # hardcoded 19 single layers and 23 double
        logger.debug("Converting lora_weights...")
        lora_weights = convert_diffusers_to_flux_transformer_checkpoint(
            lora_weights, 19, 23, has_guidance=has_guidance, prefix="transformer."
        )
        logger.debug("Converted lora_weights:")
        logger.debug(lora_weights.keys())
    else:
        logger.debug("Not using prefix 'transformer.'")
        from_original_flux = True
        lora_weights = convert_from_original_flux_checkpoint(lora_weights)
        logger.debug("Converted lora_weights:")
        logger.debug(lora_weights.keys())

    logger.info("LoRA weights loaded")
    logger.debug("Extracting keys")
    keys_without_ab = [
        key.replace(".lora_A.weight", "")
        .replace(".lora_B.weight", "")
        .replace(".alpha", "")
        for key in lora_weights.keys()
    ]
    logger.debug("Keys extracted")
    keys_without_ab = list(set(keys_without_ab))

    for key in tqdm(keys_without_ab, desc="Applying LoRA", total=len(keys_without_ab)):
        logger.debug(f"Processing key: {key}")
        try:
            module = get_module_for_key(key, model)
            logger.debug(f"Retrieved module for key: {key}")
            dtype = model.dtype
            weight_is_f8 = False
            if isinstance(module, F8Linear):
                weight_is_f8 = True
                weight_f16 = (
                    module.float8_data.clone()
                    .detach()
                    .float()
                    .mul(module.scale_reciprocal)
                    .to(module.weight.device)
                )
            elif isinstance(module, torch.nn.Linear):
                weight_f16 = module.weight.clone().detach().float()
            elif isinstance(module, CublasLinear):
                weight_f16 = module.weight.clone().detach().float()
            lora_sd = get_lora_for_key(key, lora_weights)
            if lora_sd is None:
                logger.debug(f"Skipping layer {key} since no LoRA weight is available")
                continue

            weight_f16 = apply_lora_weight_to_module(
                weight_f16, lora_sd, lora_scale=lora_scale
            )
            logger.debug(f"Updated weight for key: {key}")
            if weight_is_f8:
                module.set_weight_tensor(weight_f16.type(dtype))
            else:
                module.weight.data = weight_f16.type(dtype)
        except Exception as e:
            logger.error(f"Error applying LoRA weight for key: {key} - {str(e)}")
            if debug:
                logger.exception("Detailed traceback:")
    logger.info(
        "LoRA applied successfully"
    )  # Changed from logger.success to logger.info
    return model
