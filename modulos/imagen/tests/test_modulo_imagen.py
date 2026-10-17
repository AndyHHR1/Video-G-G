import numpy as np
from modulos.imagen.modulo_imagen.features import quality_metrics, build_features


def test_quality_black():
    metrics = quality_metrics(np.zeros((40, 60, 3), dtype=np.uint8))
    assert metrics["luminosidad_media_0_255"] == 0
    assert metrics["contraste_desviacion"] == 0


def test_real_detections_only():
    frame = np.zeros((40, 60, 3), dtype=np.uint8)
    obj = build_features(frame, {"detecciones": [
        {"class_name": "persona", "confidence": 0.9, "bbox": [1, 2, 20, 30], "track_id": 7},
        {"class_name": "sin_caja", "confidence": 1.0},
    ], "postura": [], "obstaculos": [], "riesgo": "BAJO"}, timestamp="2026-01-01T00:00:00Z")
    assert len(obj["vector_caracteristicas"]["entidades"]) == 1
    assert obj["vector_caracteristicas"]["entidades"][0]["track_id"] == 7
    assert obj["metadata_captura"]["ancho_px"] == 60
    assert "score_prediccion" not in obj["diagnostico_descriptivo"]
