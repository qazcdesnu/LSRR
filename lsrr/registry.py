from typing import Dict, Any, Type, Callable
import inspect

class Registry:
    """A generic modular registry for extensible components."""
    def __init__(self, name: str):
        self._name = name
        self._registry: Dict[str, Any] = {}

    def register(self, name: str = None) -> Callable:
        def decorator(cls_or_fn: Any) -> Any:
            key = name if name is not None else cls_or_fn.__name__
            if key in self._registry:
                # Allow re-registering for hot reloads / testing
                pass
            self._registry[key] = cls_or_fn
            return cls_or_fn
        return decorator

    def get(self, name: str) -> Any:
        if name not in self._registry:
            raise KeyError(f"'{name}' is not registered in '{self._name}' registry. Available: {list(self._registry.keys())}")
        return self._registry[name]

    def build(self, cfg: Any, **kwargs) -> Any:
        """Build an instance from a dict or OmegaConf DictConfig with 'type' key."""
        from omegaconf import DictConfig, OmegaConf
        if isinstance(cfg, DictConfig):
            cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        elif isinstance(cfg, dict):
            cfg_dict = dict(cfg)
        else:
            raise TypeError(f"Expected dict or DictConfig for build, got {type(cfg)}")

        if "type" not in cfg_dict:
            raise KeyError(f"Configuration must contain 'type' field to build from {self._name} registry. Got: {cfg_dict}")
        
        comp_type = cfg_dict.pop("type")
        target_cls = self.get(comp_type)
        
        # Merge kwargs
        merged_kwargs = {**cfg_dict, **kwargs}
        
        # If class/callable, check if it accepts kwargs or filter by signature if needed
        sig = inspect.signature(target_cls)
        # Check if **kwargs is in signature
        has_var_keyword = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
        if has_var_keyword:
            return target_cls(**merged_kwargs)
        else:
            # Only pass valid parameters
            valid_kwargs = {k: v for k, v in merged_kwargs.items() if k in sig.parameters}
            return target_cls(**valid_kwargs)

    def list_keys(self):
        return list(self._registry.keys())

# Registries for all slots
BACKBONE_REGISTRY = Registry("backbone")
ADAPTER_REGISTRY = Registry("adapter")
ENGINE_REGISTRY = Registry("engine")
TERMINATION_REGISTRY = Registry("termination")
FUSION_REGISTRY = Registry("fusion")
DECODER_REGISTRY = Registry("decoder")
LOSS_REGISTRY = Registry("loss")
DATA_REGISTRY = Registry("data")
ANALYSIS_REGISTRY = Registry("analysis")
