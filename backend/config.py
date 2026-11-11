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
    # Modelo v5: detector de escalera (1 clase) + clasificador de postura
    # sobre el recorte de cada persona (3 clases). Ver training/prepare_v5.py.
    detector: str = "models/detector/escalera_v5.pt"
    posture: str = "models/detector/postura_v5.pt"
    coco: str = "models/coco/yolov8s.pt"
    sr_dir: str = "models/sr"


@dataclass
class InferenceCfg:
    # Sensibilidad: confianza mínima de PERSONA (barra del panel, ver engine.conf)
    conf: float = 0.20
    imgsz: int = 640


@dataclass
class EngineCfg:
    pose_every: int = 8
    people_every: int = 6
    stairs_every: int = 13
    obstacle_every: int = 11
    classify_every: int = 6
    person_conf: float = 0.20    # obsoleto: la controla inference.conf
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


@dataclass
class SkeletonKeypointsCfg:
    conf_min: float = 0.45         # puntos por debajo se descartan
    hold_s: float = 0.6            # mantener un punto tapado (pasamanos)
    max_jump_torso: float = 1.2    # salto imposible entre fotogramas


@dataclass
class SkeletonSmoothingCfg:        # filtro One Euro + derivadas
    min_cutoff: float = 1.0
    beta: float = 1.5
    d_cutoff: float = 1.0
    velocity_alpha: float = 0.5


@dataclass
class SkeletonPerspectiveCfg:
    nadir: Any = None              # [x, y] normalizados: donde convergen las verticales


@dataclass
class BalanceCfg:
    trunk_abs_deg: float = 35.0
    trunk_dev_deg: float = 25.0
    trunk_extreme_deg: float = 55.0
    baseline_s: float = 3.0
    baseline_min_samples: int = 8
    hip_down_speed: float = 2.0    # torsos/s
    hip_accel: float = 8.0         # torsos/s²
    head_accel: float = 10.0       # torsos/s²
    com_margin: float = 0.35       # torsos fuera de la base de apoyo
    feet_pad: float = 0.15         # ancho de medio pie a cada lado del tobillo
    min_votes: int = 2
    persist_s: float = 0.25
    cooldown_s: float = 10.0
    min_torso_px: float = 24.0


@dataclass
class PostureRulesCfg:
    """Seguro de postura por reglas del esqueleto (skeleton/posture_rules.py).
    Alturas en torsos de la persona (de pie: cabeza 1.4 sobre la cadera,
    cadera 1.65 sobre los tobillos y 0.8 sobre las rodillas)."""
    enabled: bool = True
    # caído
    hands_feet_torso: float = 0.5   # muñecas a esta altura de los tobillos o menos
    head_hip_torso: float = 0.35    # cabeza como mucho esto por encima de la cadera
    lying_ratio: float = 1.3        # ancho / alto de los puntos visibles
    axis_deg: float = 60.0          # eje hombros->tobillos respecto a la vertical
    min_points: int = 8             # puntos visibles para medir la extensión
    # postura incorrecta (sentado / agachado)
    hip_knee_torso: float = 0.35
    knee_angle_deg: float = 110.0
    compact_torso: float = 1.0
    min_votes: int = 2
    # de pie con claridad (veta un «caído» del modelo)
    upright_head_torso: float = 0.9
    upright_hip_ankle_torso: float = 1.2
    # persistencia antes de confirmar
    persist_caido_s: float = 0.6
    persist_postura_s: float = 1.0


@dataclass
class SkeletonSignalsCfg:
    """Señales de conducta por esqueleto (skeleton/signals.py), en torsos."""
    hand_face_torso: float = 0.6       # muñeca así de cerca de la nariz
    phone_face_torso: float = 1.2      # teléfono (COCO) así de cerca de la nariz
    rail_lateral_torso: float = 0.5    # mano separada del cuerpo hacia la barandilla
    rail_above_shoulder_torso: float = 0.2
    rail_below_hip_torso: float = 0.6
    # con el pasamanos DIBUJADO en el panel: mano a esta distancia o menos
    rail_touch_torso: float = 0.45


@dataclass
class SkeletonCfg:
    enabled: bool = True
    model: str = "models/pose/yolov8s-pose.pt"
    imgsz: int = 640
    conf: float = 0.30
    iou: float = 0.6
    every: int = 2
    tracker: str = "configs/tracker_esqueletos.yaml"
    min_keypoints: int = 5
    forget_s: float = 3.0
    morphology: str = "off"        # off | mascara
    keypoints: SkeletonKeypointsCfg = field(default_factory=SkeletonKeypointsCfg)
    smoothing: SkeletonSmoothingCfg = field(default_factory=SkeletonSmoothingCfg)
    perspective: SkeletonPerspectiveCfg = field(default_factory=SkeletonPerspectiveCfg)
    balance: BalanceCfg = field(default_factory=BalanceCfg)
    postura: PostureRulesCfg = field(default_factory=PostureRulesCfg)
    senales: SkeletonSignalsCfg = field(default_factory=SkeletonSignalsCfg)
    # rutas absolutas, rellenadas por Settings
    model_path: Any = None
    tracker_path: Any = None


def _baja_vram_default() -> dict:
    return {"enhancement.half": False,
            "enhancement.live.input_max": 320,
            "skeleton.every": 3}


def _cpu_default() -> dict:
    # Sin GPU, el esqueleto YOLOv8s-Pose era el 50 % del coste del motor
    # (61.6 ms/frame medidos con tools/bench_engine.py --set device=cpu).
    return {"skeleton.model": "models/pose/yolov8n-pose.pt",
            "skeleton.imgsz": 480,
            "skeleton.every": 4,
            "engine.obstacle_every": 17}


@dataclass
class HardwareCfg:
    profile: str = "auto"          # auto | normal | baja_vram | cpu
    low_vram_gb: float = 4.5       # auto: <= esta VRAM -> baja_vram
    baja_vram: dict = field(default_factory=_baja_vram_default)
    cpu: dict = field(default_factory=_cpu_default)   # auto: sin GPU -> cpu
    # En CPU, exportar y usar los modelos en OpenVINO (perception/accel.py)
    openvino_cpu: bool = True
    # En CPU, el COCO que localiza personas/objetos en INT8 (~2x más rápido)
    openvino_int8_coco: bool = True
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
    skeleton: SkeletonCfg = field(default_factory=SkeletonCfg)
    modules: ModulesCfg = field(default_factory=ModulesCfg)

    # --- rutas absolutas ------------------------------------------------
    def path(self, rel: str | Path) -> Path:
        p = Path(rel)
        return p if p.is_absolute() else ROOT / p

    @property
    def detector_path(self) -> Path:
        return self.path(self.models.detector)

    @property
    def posture_path(self) -> Path:
        return self.path(self.models.posture)

    @property
    def coco_path(self) -> Path:
        return self.path(self.models.coco)

    @property
    def sr_dir(self) -> Path:
        return self.path(self.models.sr_dir)

    @property
    def alerts_dir(self) -> Path:
        return self.path(self.alerts.dir)

    def __post_init__(self):
        self.resolve_paths()

    def resolve_paths(self) -> None:
        sk = self.skeleton
        sk.model_path = self.path(sk.model)
        tr = Path(sk.tracker)
        # «bytetrack.yaml» / «botsort.yaml» sin carpeta: los de Ultralytics
        sk.tracker_path = self.path(tr) if tr.parent != Path(".") else tr


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
    settings = _build(Settings, data)
    settings.resolve_paths()
    return settings
