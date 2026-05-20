from typing import Any


_HINT = (
    "Neither Int8WeightOnlyConfig nor int8_weight_only is importable from "
    "torchao.quantization. Upgrade torchao or use the int8_bnb variant."
)


def _resolve_quantize_():
    try:
        from torchao.quantization import quantize_
        return quantize_
    except ImportError as e:
        raise ImportError(
            "torchao is not installed; install torchao or use int8_bnb."
        ) from e


quantize_ = _resolve_quantize_()


def torchao_int8_weight_only_config() -> Any:
    """Return a config object compatible with the installed torchao version."""
    try:
        from torchao.quantization import Int8WeightOnlyConfig
        return Int8WeightOnlyConfig()
    except ImportError:
        pass

    try:
        from torchao.quantization import int8_weight_only
        return int8_weight_only()
    except ImportError:
        pass

    try:
        from torchao.quantization.quant_api import int8_weight_only
        return int8_weight_only()
    except ImportError:
        pass

    raise ImportError(_HINT)
