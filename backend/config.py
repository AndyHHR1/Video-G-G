"""Carga de la configuración de ejecución (`configs/app.yaml`).

La configuración vive en YAML para poder cambiar umbrales, cadencias y modelos
sin tocar el código (RQNF21). Este módulo la convierte en dataclasses tipadas
y admite sobrescrituras puntuales desde la línea de comandos (`--set a.b=v`).
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "configs" / "app.yaml"


@dataclass
class ServerCfg:
    host: str = "127.0.0.1"
    port: int = 5000
    debug: bool = False
    # Solo para los módulos aparte (fotografía de Imagen, archivo de Audio):
    # el vídeo es únicamente en vivo y no acepta subidas.
    max_upload_mb: int = 64


@dataclass
class ModelsCfg:
    detector: str = "models/detector/escaleras_s7_v4.pt"
    coco: str = "models/coco/yolov8s.pt"
    sr_dir: str = "models/sr"


@dataclass
class InferenceCfg:
    conf: float = 0.25
    imgsz: int = 640


@dataclass
class EngineCfg:
    pose_every: int = 8
    people_every: int = 6
    stairs_every: int = 13
    obstacle_every: int = 11
    classify_every: int = 6
    person_conf: float = 0.20
    person_imgsz: int = 512
    max_stairs_boxes: int = 2


@dataclass
class ThresholdsCfg:
    prefilter: float = 0.75
    confirm: float = 0.85
    obstacle: float = 0.25
    alert: float = 0.35
    fall_crop_conf: float = 0.50
    fall_confirm_frames: int = 3


@dataclass
class PersistenceCfg:
    alto_seconds: float = 1.2
    medio_seconds: float = 0.6
    default_seconds: float = 1.0
    grace_seconds: float = 1.0
    occlusion_seconds: float = 1.0
    cooldown_seconds: float = 15.0
    iou_match: float = 0.25


@dataclass
class AlertsCfg:
    dir: str = "outputs/alerts"
    keep_days: int = 7
    max_images: int = 200


@dataclass
class PrivacyCfg:
    anonymize_faces: bool = True


@dataclass
class LiveSRCfg:
    model: str = "general_x4v3"
    input_max: int = 480
    output_width: int = 960
    jpeg_quality: int = 80


@dataclass
class EnhancementCfg:
    enabled: Any = "auto"          # True / False / "auto"
    half: bool = True
    live: LiveSRCfg = field(default_factory=LiveSRCfg)


@dataclass
class CameraCfg:
    source: Any = 0                # índice de cámara o URL (rtsp://, http://)
    reconnect_attempts: int = 5


def _baja_vram_default() -> dict:
    return {"enhancement.half": False,
            "enhancement.live.input_max": 320}


@dataclass
class HardwareCfg:
    profile: str = "auto"          # auto | normal | baja_vram
    low_vram_gb: float = 4.5       # auto: <= esta VRAM -> baja_vram
    baja_vram: dict = field(default_factory=_baja_vram_default)
    # Rellenados en el arranque por backend.hardware.apply_profile
    active: str = "normal"
    gpu: dict = field(default_factory=dict)


@dataclass
class ModuleCfg:
    """Módulos aparte (modulos/audio, modulos/imagen). Se cargan sin tocar el
    vídeo: si fallan, el vídeo en vivo sigue funcionando."""
    enabled: bool = True
    options: dict = field(default_factory=dict)


@dataclass
class ModulesCfg:
    audio: ModuleCfg = field(default_factory=ModuleCfg)
    imagen: ModuleCfg = field(default_factory=ModuleCfg)


@dataclass
class Settings:
    server: ServerCfg = field(default_factory=ServerCfg)
    device: Any = "auto"
    models: ModelsCfg = field(default_factory=ModelsCfg)
    inference: InferenceCfg = field(default_factory=InferenceCfg)
    engine: EngineCfg = field(default_factory=EngineCfg)
    thresholds: ThresholdsCfg = field(default_factory=ThresholdsCfg)
    persistence: PersistenceCfg = field(default_factory=PersistenceCfg)
    alerts: AlertsCfg = field(default_factory=AlertsCfg)
    privacy: PrivacyCfg = field(default_factory=PrivacyCfg)
    enhancement: EnhancementCfg = field(default_factory=EnhancementCfg)
    camera: CameraCfg = field(default_factory=CameraCfg)
    hardware: HardwareCfg = field(default_factory=HardwareCfg)
    modules: ModulesCfg = field(default_factory=ModulesCfg)

    # --- rutas absolutas ------------------------------------------------
    def path(self, rel: str | Path) -> Path:
        p = Path(rel)
        return p if p.is_absolute() else ROOT / p

    @property
    def detector_path(self) -> Path:
        return self.path(self.models.detector)

    @property
    def coco_path(self) -> Path:
        return self.path(self.models.coco)

    @property
    def sr_dir(self) -> Path:
        return self.path(self.models.sr_dir)

    @property
    def alerts_dir(self) -> Path:
        return self.path(self.alerts.dir)


# --------------------------------------------------------------------------- #
def _build(cls, data: dict | None):
    """Construye una dataclass a partir de un dict, ignorando claves sobrantes."""
    obj = cls()
    for f in fields(cls):
        if not data or f.name not in data:
            continue
        current = getattr(obj, f.name)
        value = data[f.name]
        if is_dataclass(current) and isinstance(value, dict):
            value = _build(type(current), value)
        setattr(obj, f.name, value)
    return obj


def _parse_scalar(raw: str) -> Any:
    """Convierte el texto de `--set` al tipo natural (int, float, bool...)."""
    return yaml.safe_load(raw)


def _apply_override(data: dict, dotted: str) -> None:
    if "=" not in dotted:
        raise ValueError(f"--set espera clave=valor, recibido: {dotted!r}")
    key, raw = dotted.split("=", 1)
    node = data
    parts = key.strip().split(".")
    for p in parts[:-1]:
        node = node.setdefault(p, {})
    node[parts[-1]] = _parse_scalar(raw)


def load_settings(path: str | Path | None = None,
                  overrides: list[str] | None = None) -> Settings:
    """Lee el YAML (si existe) y aplica las sobrescrituras `clave.sub=valor`."""
    cfg_path = Path(path) if path else DEFAULT_CONFIG
    data: dict = {}
    if cfg_path.is_file():
        data = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    for item in overrides or []:
        _apply_override(data, item)
    return _build(Settings, data)
