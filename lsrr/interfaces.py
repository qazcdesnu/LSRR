from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
import torch
import torch.nn as nn

@dataclass
class StepDiagnostics:
    """Per-cycle diagnostic indicators recorded during recursive refinement."""
    delta_state: float
    kl_div: Optional[float] = None
    entropy: Optional[float] = None
    extra: Optional[Dict[str, Any]] = None

@dataclass
class DataSample:
    """Unified data schema: {question, answer, cot_steps, meta}"""
    question: str
    answer: str
    cot_steps: List[str]
    meta: Dict[str, Any]

class BaseBackboneExtractor(ABC):
    """Slot 3.1: Extracts all layer hidden states H ∈ R^{L x d_in} from frozen backbone."""
    @abstractmethod
    def extract_hidden_states(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_rule: str = "last_token"
    ) -> torch.Tensor:
        """Extract H.
        Returns:
            Tensor of shape [batch_size, num_layers, hidden_dim]
        """
        pass

    @property
    @abstractmethod
    def num_layers(self) -> int:
        pass

    @property
    @abstractmethod
    def hidden_dim(self) -> int:
        pass

class BaseLayerAdapter(nn.Module, ABC):
    """Slot 3.2: Layer-wise affine transformation + normalization + layer pos emb."""
    @abstractmethod
    def forward(self, H: torch.Tensor) -> torch.Tensor:
        """Args:
            H: [B, L, d_in]
        Returns:
            R0: [B, L, d_model]
        """
        pass

class BaseRefinementEngine(nn.Module, ABC):
    """Slot 3.3: Recursive Refinement Engine."""
    @abstractmethod
    def forward_step(
        self,
        R_m: torch.Tensor,
        R0: torch.Tensor,
        m: int
    ) -> torch.Tensor:
        """One refinement cycle: R_m -> R_{m+1}.
        Args:
            R_m: [B, L, d_model] current state
            R0: [B, L, d_model] initial state
            m: current cycle index (0-indexed)
        Returns:
            R_{m+1}: [B, L, d_model] updated state
        """
        pass

class BaseTerminationRule(ABC):
    """Slot 3.5: Convergence and early exit rule."""
    @abstractmethod
    def reset(self, batch_size: int, device: torch.device):
        pass

    @abstractmethod
    def should_stop(
        self,
        R_m: torch.Tensor,
        R_next: torch.Tensor,
        logits_m: Optional[torch.Tensor] = None,
        logits_next: Optional[torch.Tensor] = None,
        m: int = 0
    ) -> Tuple[torch.Tensor, StepDiagnostics]:
        """Returns:
            stop_mask: BoolTensor[B] - samples that meet termination criteria
            diagnostics: StepDiagnostics for current cycle
        """
        pass

class BaseFusionHead(nn.Module, ABC):
    """Slot 3.6a: Representation Fusion (Attention pooling over layers + residual)."""
    @abstractmethod
    def forward(
        self,
        R_star: torch.Tensor,
        h_orig_L: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Args:
            R_star: [B, L, d_model] final refined state
            h_orig_L: Optional [B, d_model or d_in] context representation from original layer L
        Returns:
            h_fusion: [B, d_fusion]
            alpha_weights: [B, L] pooling attention weights across layers
        """
        pass

class BaseAnswerDecoder(nn.Module, ABC):
    """Slot 3.6b: Answer Token Generation."""
    @abstractmethod
    def forward(
        self,
        h_fusion: torch.Tensor,
        target_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Teacher-forcing forward pass.
        Args:
            h_fusion: [B, d_fusion]
            target_ids: [B, seq_len] answer token ids
        Returns:
            logits: [B, seq_len, vocab_size]
        """
        pass

    @abstractmethod
    def generate(
        self,
        h_fusion: torch.Tensor,
        max_new_tokens: int = 64,
        eos_token_id: Optional[int] = None
    ) -> torch.Tensor:
        """Autoregressive generation for inference/eval."""
        pass

class BaseLoss(nn.Module, ABC):
    """Slot 3.7: Loss function module."""
    @abstractmethod
    def forward(
        self,
        model_outputs: Dict[str, Any],
        batch: Dict[str, Any]
    ) -> Dict[str, torch.Tensor]:
        """Returns dict containing at least {"loss": total_loss_tensor}."""
        pass

class BaseDataModule(ABC):
    """Slot 3.8: Unified data source with dataset-specific evaluation & scoring."""
    @abstractmethod
    def get_split(self, split: str) -> List[DataSample]:
        pass

    @abstractmethod
    def evaluate_answer(self, prediction: str, target: str, meta: Dict[str, Any]) -> bool:
        """Dataset-specific normalization and exact match evaluation."""
        pass

class BaseAnalysisPlugin(ABC):
    """Slot 3.10: Post-hoc analysis suite consuming recorded diagnostic traces."""
    @abstractmethod
    def analyze(self, diagnostics_data: List[Dict[str, Any]], run_dir: str) -> Dict[str, Any]:
        pass
