"""GroundingDINO <-> transformers 5.x compatibility shim (auto-loaded).

Why this exists
---------------
The Grounded-SAM-2 vendored GroundingDINO text encoder (``BertModelWarper``)
was written against the transformers 4.x ``BertModel`` API.  The ``svlr`` conda
env ships transformers 5.x (required by gradio 6.x / huggingface-hub 1.x), which
removed two ModuleUtilsMixin methods GroundingDINO relies on:

  * ``get_head_mask`` (removed entirely)
  * ``get_extended_attention_mask`` (kept, but its 3rd positional parameter is
    now ``dtype`` instead of ``device``; GroundingDINO passes ``device`` there,
    which corrupts the mask dtype)

Downgrading transformers is not possible here: transformers 4.x pins
huggingface-hub < 1.0, but gradio 6.20 (the SVLR web/API server) requires
huggingface-hub >= 1.2.  So we restore the exact 4.x behaviour of just these two
methods, scoped so normal transformers 5.x callers are unaffected.

This module is loaded automatically at interpreter startup via the sibling
``zz_gdino_tf5_compat.pth`` file.  Every operation is guarded so a failure can
never prevent the Python interpreter in this env from starting.
"""

def _install():  # pragma: no cover - environment glue
    try:
        import torch
        from transformers.modeling_utils import ModuleUtilsMixin
    except Exception:
        return

    # --- 1) restore ModuleUtilsMixin.get_head_mask (removed in transformers 5.x)
    if not hasattr(ModuleUtilsMixin, "get_head_mask"):
        def _convert_head_mask_to_5d(self, head_mask, num_hidden_layers):
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
            return head_mask.to(dtype=self.dtype)

        def get_head_mask(self, head_mask, num_hidden_layers, is_attention_chunked=False):
            if head_mask is not None:
                head_mask = self._convert_head_mask_to_5d(head_mask, num_hidden_layers)
                if is_attention_chunked is True:
                    head_mask = head_mask.unsqueeze(-1)
            else:
                head_mask = [None] * num_hidden_layers
            return head_mask

        if not hasattr(ModuleUtilsMixin, "_convert_head_mask_to_5d"):
            ModuleUtilsMixin._convert_head_mask_to_5d = _convert_head_mask_to_5d
        ModuleUtilsMixin.get_head_mask = get_head_mask

    # --- 2) make get_extended_attention_mask tolerate GroundingDINO's
    #        (attention_mask, input_shape, device) positional call, while
    #        delegating genuine transformers 5.x callers to the original.
    if not getattr(ModuleUtilsMixin.get_extended_attention_mask, "_gdino_tf5_wrapped", False):
        _orig = ModuleUtilsMixin.get_extended_attention_mask

        def _classic(self, attention_mask, input_shape, dtype):
            if attention_mask.dim() == 3:
                extended = attention_mask[:, None, :, :]
            elif attention_mask.dim() == 2:
                extended = attention_mask[:, None, None, :]
            else:
                raise ValueError(
                    f"Wrong shape for attention_mask (shape {tuple(attention_mask.shape)})"
                )
            extended = extended.to(dtype=dtype)
            return (1.0 - extended) * torch.finfo(dtype).min

        def get_extended_attention_mask(self, attention_mask, input_shape, device=None, dtype=None):
            # GroundingDINO (transformers 4.x style) passes a torch.device as the
            # 3rd positional argument. Detect that and run the classic 4.x math,
            # which correctly handles the custom 3D per-phrase attention mask.
            if isinstance(device, torch.device):
                use_dtype = dtype if dtype is not None else self.dtype
                return _classic(self, attention_mask, input_shape, use_dtype)
            # Otherwise behave like the installed transformers version. In 5.x the
            # 3rd positional parameter is `dtype`.
            effective_dtype = dtype if dtype is not None else device
            try:
                if effective_dtype is None:
                    return _orig(self, attention_mask, input_shape)
                return _orig(self, attention_mask, input_shape, effective_dtype)
            except TypeError:
                use_dtype = effective_dtype if effective_dtype is not None else self.dtype
                return _classic(self, attention_mask, input_shape, use_dtype)

        get_extended_attention_mask._gdino_tf5_wrapped = True
        ModuleUtilsMixin.get_extended_attention_mask = get_extended_attention_mask


try:
    _install()
except Exception:
    # Never let a compatibility shim break interpreter startup.
    pass
