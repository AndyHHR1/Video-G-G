"""Desde raíz: python -m scripts.probar_clip_imagen RUTA_MP4 --intervalo 1"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))  # raíz de Video-G-G
import argparse
import json
from pathlib import Path

from modulos.imagen.modulo_imagen.analizador_clip import analizar_clip


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('clip', help='Archivo de video de hasta 10 segundos')
    parser.add_argument('--intervalo', type=float, default=1.0, help='Segundos entre fotogramas (por defecto 1)')
    parser.add_argument('--modo', choices=['original', 'adaptativo', 'gamma', 'clahe'], default='adaptativo')
    parser.add_argument('--salida', default='runs/imagen/prueba_clip')
    args = parser.parse_args()
    salida = Path(args.salida)
    resumen, _ = analizar_clip(args.clip, modo=args.modo, intervalo_s=args.intervalo,
                             guardar_en=salida)
    salida.mkdir(parents=True, exist_ok=True)
    (salida / 'resultados_clip.json').write_text(json.dumps(resumen, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"Clip: {resumen['clip']}; duración {resumen['duracion_s']}s; analizados {resumen['fotogramas_analizados']}/{resumen['fotogramas_planificados']}; total {resumen['tiempo_total_ms']}ms")
    print(f'Resultados: {salida.resolve()}')

if __name__ == '__main__':
    main()
