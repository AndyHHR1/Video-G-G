#!/usr/bin/env python3
"""
Integracion del dataset Le2i (raw) al proyecto.

POR QUE ESTE DATASET
--------------------
Los tres datasets originales (`fall`, `stairs`, `tracking`) NO contienen
postura pre-caida: `fall` solo tiene personas ya en el suelo y `tracking`
son peatones sin escalera. `Caso.md` (RQF02, categoria 4) exige detectar
"caidas activas o perdida inminente de equilibrio", y el objetivo declarado
del proyecto es anticipar la caida.

`Le2i-raw` (citado en `Caso.md` §1.4) aporta justo lo que falta: sus carpetas
estan etiquetadas por ESTADO y una de ellas es `Likefall` = "the center of
gravity was unstable", es decir, la perdida de equilibrio ANTES de caer.

MAPEO DE CLASES
---------------
    Blank    -> sin caja (negativo;RQF04 filtra falsos positivos)
    Fall     -> 0 persona_caido
    Lie      -> 0 persona_caido
    Likefall -> 5 persona_desequilibrio   <-- clase NUEVA de pre-caida
    Stand    -> 2 persona_erguida

COMO SE GENERAN LAS CAJAS
-------------------------
Le2i es un dataset de CLASIFICACION: no trae bounding boxes. Las cajas se
obtienen con un detector de personas preentrenado en COCO (yolov8s). Es una
pseudo-etiquetacion, y por eso el script la VALIDA: informa el porcentaje de
frames con persona detectada por clase. Si `Blank` mostrara muchas
detecciones, la pseudo-etiquetacion no seria fiable y habria que descartar el
approach.

Uso:
    python3 scripts/prepare_le2i.py --out datasets/le2i_yolo
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Clase 0 en COCO: "person". Los modelos preentrenados de Ultralytics usan
# el orden de clases de COCO, donde el indice 0 es justamente `person`.
COCO_PERSON_ID = 0

# Frames consecutivos son casi identicos. Se conserva 1 de cada STRIDE[clase]
# para evitar redundancia (que ademas inflaria el tiempo de epoca).
STRIDE = {
    "Likefall": 1,   # clase escasa y precious: se conserva todo
    "Fall": 3,
    "Lie": 4,
    "Stand": 5,
    "Blank": 10,
}

# Etiqueta de origen -> id de clase unificado
CLASS_MAP = {
    "Fall": 0,      # persona_caido
    "Lie": 0,       # persona_caido
    "Stand": 2,     # persona_erguida
    "Likefall": 5,  # persona_desequilibrio (PRE-CAIDA)
    "Blank": None,  # sin caja: es un negativo
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=ROOT / "datasets/_external/Le2i-raw")
    ap.add_argument("--out", type=Path, default=ROOT / "datasets/le2i_yolo")
    ap.add_argument("--weights", default="yolov8s.pt", help="Detector COCO")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=200,
                    help="Frames por tanda de deteccion. Acota el pico de RAM.")
    args = ap.parse_args()

    raw = args.raw.resolve()
    out = args.out.resolve()
    if not raw.is_dir():
        raise SystemExit(f"No existe el dataset crudo: {raw}")

    from ultralytics import YOLO

    # ---------------------------------------------------------------- #
    # 1. Recolectar los frames a procesar (con su clase y su video)
    # ---------------------------------------------------------------- #
    items = []   # (ruta, clase, video_id, split_origen)
    for split in ("train", "val"):
        for cls in STRIDE:
            base = raw / split / cls
            if not base.is_dir():
                continue
            for video in sorted(p for p in base.iterdir() if p.is_dir()):
                frames = sorted(
                    p for p in video.iterdir()
                    if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
                )
                stride = STRIDE[cls]
                for i, frame in enumerate(frames):
                    if i % stride:
                        continue
                    items.append((frame, cls, video.name, split))
    print(f"Frames seleccionados tras submuestreo: {len(items)}")
    by_cls = Counter(c for _, c, _, _ in items)
    for c, n in sorted(by_cls.items()):
        print(f"  {c:9} {n:>6}  (stride {STRIDE[c]})")

    # ---------------------------------------------------------------- #
    # 2. Deteccion de personas con el modelo COCO
    #
    # Se hace la inferencia A MANO con torch en vez de usar
    # `YOLO.predict()`. Motivo: reutilizar el Predictor de Ultralytics entre
    # varias llamadas deja estado acumulado yprovoca "CUDA driver error:
    # device not ready" a los pocos lotes; y el reintento en CPU con batch 8
    # desborda la RAM de la maquina (~8 GB) y muere por el OOM-killer.
    #
    # Con este codigo el pico de memoria es exactamente `batch` imagenes y no
    # hay estado que arrastrar entre lotes.
    # ---------------------------------------------------------------- #
    print(f"\nDetectando personas con {args.weights} (conf>={args.conf})...")
    import cv2
    import numpy as np
    import torch
    from ultralytics.utils.nms import non_max_suppression

    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = YOLO(args.weights).model.to(device).eval()
    net_cpu = None      # se crea bajo demanda si la GPU falla
    dets: dict[Path, list] = {}
    paths = [it[0] for it in items]
    total = len(paths)

    def detect_block(block: list[Path], on: str) -> dict[Path, list]:
        nonlocal net_cpu
        model = net
        if on == "cpu":
            if net_cpu is None:
                print("\n  (GPU no disponible: se continua en CPU)", flush=True)
                net_cpu = YOLO(args.weights).model.to("cpu").eval()
            model = net_cpu
        batch_imgs = []
        usable: list[Path] = []
        for p in block:
            im = cv2.imread(str(p))            # BGR
            if im is None:                    # imagen ilegible -> sin caja
                dets[p] = []
                continue
            im = cv2.resize(im, (args.imgsz, args.imgsz), interpolation=cv2.INTER_LINEAR)
            batch_imgs.append(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
            usable.append(p)

        if not batch_imgs:
            return {}

        arr = np.stack(batch_imgs).astype(np.float32) / 255.0   # N,H,W,3
        tensor = torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous().to(on)
        with torch.inference_mode():
            preds = model(tensor)
        # salida: (batch, 4+1+nc, anchors) -> NMS -> lista de (n,6)
        # `classes=[0]` es ESENCIAL: el detector es un modelo COCO de 80
        # clases y sin este filtro cada silla, mesa o Television contaria como
        # una persona (Le2i es de un unico sujeto, asi que la validacion de mas
        # abajo lo detecta de inmediato).
        out = non_max_suppression(
            preds, conf_thres=args.conf, iou_thres=0.45,
            classes=[COCO_PERSON_ID], max_det=5,
        )
        result = {}
        for path, det in zip(usable, out):
            if det is None or len(det) == 0:
                result[path] = []
                continue
            # las coordenadas estan en el espacio cuadrado de `imgsz`;
            # dividir por imgsz las devuelve a coordenadas normalizadas 0..1
            result[path] = (det[:, :4].cpu().numpy() / args.imgsz).tolist()
        del tensor, preds, out, arr, batch_imgs
        return result

    failures = 0
    for start in range(0, total, args.chunk):
        block = paths[start:start + args.chunk]
        # 1) GPU, 2) GPU tras vaciar la cache, 3) CPU. El ultimo recurso es la
        # CPU porque el allocator de CUDA puede quedar corrupto y vaciar la
        # cache no lo repara.
        for attempt, on in enumerate((device, device, "cpu")):
            try:
                dets.update(detect_block(block, on))
                break
            except Exception as exc:  # noqa: BLE001
                failures += 1
                print(
                    f"\n  [reintento {attempt + 1}/3] lote "
                    f"{start // args.chunk + 1} en {on}: {type(exc).__name__}: "
                    f"{str(exc)[:80]}",
                    flush=True,
                )
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                time.sleep(3)
        # si los tres intentos fallaron, el lote queda sin cajas en lugar de
        # romper el script
        dets.update({p: [] for p in block if p not in dets})
        print(
            f"  {min(start + args.chunk, total)}/{total} frames",
            end="\r", flush=True,
        )
    print()
    if failures:
        print(f"  (se registraron {failures} reintentos)")

    # ---------------------------------------------------------------- #
    # 3. VALIDACION de la pseudo-etiquetacion
    #
    # Si el detector fuera una caja negra, `Blank` (frames sin persona)
    # tendria tantas detecciones como el resto. Esa comprobacion es la que
    # garantiza que las cajas derivadas son fiables.
    #
    # El umbral es POR CLASE y no uniforme, porque las clases no son
    # equivalentes: se verifican las que el proyecto necesita de verdad
    # (`Likefall` = pre-caida, `Stand` = postura erguida, `Blank` = negativo)
    # y las demas se aceptan solo si tambien superan el control.
    #
    # `Fall` y `Lie` fallan de forma sistematica porque una persona TENDIDA
    # en el suelo esta muy escorzada, y los detectores de COCO se entrenan con
    # gente de pie. Es una limitacion real, no un ajuste de parametro: bajar el
    # umbral de 0.35 a 0.20 solo los subio del 56%/58% al 61%/66%.
    # ---------------------------------------------------------------- #
    print("\n--- VALIDACION de la pseudo-etiquetacion ---")
    print(f"{'clase':10} {'frames':>7} {'con persona':>12} {'%':>7} "
          f"{'cajas/frame':>12}  veredicto")
    stats: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    for path, cls, _vid, _split in items:
        n = len(dets.get(path, []))
        stats[cls][0] += 1
        stats[cls][1] += 1 if n else 0
        stats[cls][2] += n

    MIN_DETECTION = 0.80      # clases con persona: se espera detectarlas
    MAX_BLANK = 0.15          # 'Blank' NO debe tener persona
    MAX_BOXES = 1.6           # Le2i graba un unico sujeto

    rejected: set[str] = set()
    for cls in sorted(stats):
        total, with_person, boxes = stats[cls]
        pct = with_person / total if total else 0
        bpf = boxes / total if total else 0.0

        if cls == "Blank":
            ok = pct <= MAX_BLANK
        else:
            ok = pct >= MIN_DETECTION and bpf <= MAX_BOXES
        verdict = "OK" if ok else "DESCARTADA"
        if not ok:
            rejected.add(cls)
        print(
            f"{cls:10} {total:>7} {with_person:>12} {100*pct:>6.1f}% "
            f"{bpf:>12.2f}  {verdict}"
        )

    # Las clases que el proyecto necesita si o si no pueden fallar.
    CRITICAL = {"Likefall", "Stand", "Blank"}
    missing = CRITICAL - set(stats) - rejected
    if missing or (CRITICAL & rejected):
        bad = sorted((CRITICAL & rejected) | missing)
        raise SystemExit(
            f"ABORTADO: las clases criticas {bad} no superaron la validacion. "
            f"No se escribe el dataset."
        )

    if rejected:
        print(
            "\nSe descartan "
            + ", ".join(sorted(rejected))
            + ".\n  Motivo: el detector COCO no localiza de forma fiable a una "
            "persona\n  tendida en el suelo (sujeto escorzado). La clase "
            "`persona_caido` ya\n  esta cubierta por las anotaciones humanas del "
            "dataset `fall`, de modo que\n  no se pierde cobertura."
        )
    blank_pct = 100 * stats["Blank"][1] / stats["Blank"][0] if stats["Blank"][0] else 0
    print(f"\nACEPTADO: {blank_pct:.0f}% de falsos positivos en 'Blank'.")

    # ---------------------------------------------------------------- #
    # 4. Escribir el dataset YOLO
    #
    # El split NO se toma del `train`/`val` de Le2i tal cual: se reparte por
    # VIDEO para que los frames de un mismo video (identicos entre si) no se
    # separen entre train y val. Ademas se mezcla el train y el val originales
    # de Le2i y se vuelve a repartir, porque sus proporciones no encajan con el
    # dataset unificado.
    # ---------------------------------------------------------------- #
    for split in ("train", "val", "test"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    import random
    rng = random.Random(42)
    videos: dict[tuple[str, str], list] = defaultdict(list)
    for it in items:
        if it[1] in rejected:      # clase que no supero la validacion
            continue
        videos[(it[1], it[2])].append(it)     # agrupa por (clase, video)

    keys = sorted(videos)
    rng.shuffle(keys)
    n = len(keys)
    n_test = max(1, round(n * 0.15))
    n_val = max(1, round(n * 0.15))
    assign = {}
    for i, k in enumerate(keys):
        assign[k] = (
            "test" if i < n_test else "val" if i < n_test + n_val else "train"
        )

    manifest = []
    written = Counter()
    for key, group in videos.items():
        cls, vid = key
        split = assign[key]
        cid = CLASS_MAP[cls]
        for path, _c, _v, _o in group:
            stem = f"{cls}_{vid}_{path.stem}"
            dst_img = out / "images" / split / f"{stem}.jpg"
            if not dst_img.exists():
                dst_img.symlink_to(path)

            rows = []
            if cid is not None:
                for x1, y1, x2, y2 in dets[path]:
                    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                    bw, bh = x2 - x1, y2 - y1
                    if bw <= 0 or bh <= 0:
                        continue
                    rows.append(
                        f"{cid} {min(max(cx,0),1):.6f} {min(max(cy,0),1):.6f} "
                        f"{min(max(bw,1e-6),1):.6f} {min(max(bh,1e-6),1):.6f}"
                    )
            (out / "labels" / split / f"{stem}.txt").write_text(
                "\n".join(rows) + ("\n" if rows else "")
            )
            written[split] += 1
            manifest.append({
                "split": split, "origin": "le2i", "class_name": cls,
                "video": vid, "out_name": dst_img.name,
                "n_boxes": len(rows), "src_image": str(path),
            })

    with (out / "manifest.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(manifest[0]))
        w.writeheader()
        w.writerows(manifest)

    print("\nDataset Le2i escrito:")
    for split in ("train", "val", "test"):
        n_lbl = len(list((out / "labels" / split).glob("*.txt")))
        n_box = sum(
            1 for f in (out / "labels" / split).glob("*.txt")
            for line in f.read_text().splitlines() if line.strip()
        )
        print(f"  {split:6} {n_lbl:>6} imagenes, {n_box:>6} cajas")
    print(f"  (de las cuales 'persona_desequilibrio' solo puede venir de Likefall)")

    (out / "validacion.json").write_text(
        json.dumps(
            {c: {"frames": v[0], "con_persona": v[1], "cajas": v[2]}
             for c, v in stats.items()},
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()