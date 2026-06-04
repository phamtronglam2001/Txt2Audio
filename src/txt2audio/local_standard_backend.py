"""Custom standard backend to force local GGUF path loading."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from txt2audio.local_assets import NEUCODEC_ONNX_DIR, NEUCODEC_ONNX_FILE
from txt2audio.model_paths import bundle_dir_for_gguf_file
from vieneu.standard import VieNeuTTS

# ONNX decoder: CPU-friendly, no PyTorch distill-neucodec weights (see vieneu.base._load_codec).
STANDARD_ONNX_CODEC = "neuphonic/neucodec-onnx-decoder-int8"


class LocalPathStandardVieNeuTTS(VieNeuTTS):
    """VieNeu standard backend that supports direct local .gguf path."""

    def __init__(
        self,
        backbone_repo: str,
        backbone_device: str = "cpu",
        codec_repo: str = STANDARD_ONNX_CODEC,
        codec_device: str = "cpu",
        hf_token: Optional[str] = None,
        voices_repo_id: Optional[str] = None,
        assets_dir: Optional[str] = None,
    ) -> None:
        self._voices_repo_id = voices_repo_id
        self._assets_dir = (
            Path(assets_dir)
            if assets_dir
            else bundle_dir_for_gguf_file(Path(backbone_repo))
        )
        super().__init__(
            backbone_repo=backbone_repo,
            backbone_device=backbone_device,
            codec_repo=codec_repo,
            codec_device=codec_device,
            hf_token=hf_token,
        )

    def _load_voices(
        self,
        backbone_repo: Optional[str],
        hf_token: Optional[str] = None,
        clear_existing: bool = False,
    ) -> None:
        """Preset voices từ voices.json trong thư mục bundle (offline)."""
        local_voices = self._assets_dir / "voices.json"
        if local_voices.is_file():
            self._load_voices_from_file(local_voices, clear_existing=clear_existing)
            return
        if self._voices_repo_id:
            super()._load_voices(self._voices_repo_id, hf_token, clear_existing)
            return
        p = Path(backbone_repo) if backbone_repo else None
        if p and p.is_file() and p.suffix.lower() == ".gguf":
            return
        super()._load_voices(backbone_repo, hf_token, clear_existing)

    def _load_codec(self, codec_repo: str, codec_device: str) -> None:
        local_onnx = self._assets_dir / NEUCODEC_ONNX_DIR / NEUCODEC_ONNX_FILE
        if local_onnx.is_file():
            from neucodec import NeuCodecOnnxDecoder

            self.codec = NeuCodecOnnxDecoder(str(local_onnx))
            self._is_onnx_codec = True
            return
        super()._load_codec(codec_repo, codec_device)

    def _load_backbone(self, backbone_repo: str, backbone_device: str, hf_token: Optional[str] = None) -> None:
        p = Path(backbone_repo)
        if p.is_file() and p.suffix.lower() == ".gguf":
            try:
                from llama_cpp import Llama
            except ImportError as e:
                raise ImportError(
                    "Failed to import `llama_cpp`. Please install llama-cpp-python version >= 0.3.16."
                ) from e

            use_gpu = backbone_device in ("gpu", "cuda")
            self.backbone = Llama(
                model_path=str(p),
                verbose=False,
                n_gpu_layers=-1 if use_gpu else 0,
                n_ctx=self.max_context,
                mlock=True,
                flash_attn=True if use_gpu else False,
            )
            self._is_quantized_model = True
            return

        super()._load_backbone(backbone_repo, backbone_device, hf_token)
