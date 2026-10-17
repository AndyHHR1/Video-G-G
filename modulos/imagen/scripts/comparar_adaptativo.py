"""Ejemplo: python -m scripts.comparar_adaptativo datasets/escaleras/frames/video_06_t006.jpg"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))  # raíz de Video-G-G
import argparse, json
from pathlib import Path
import cv2
from modulos.imagen.modulo_imagen.preprocesamiento import mejorar_imagen


def main():
    pa=argparse.ArgumentParser()
    pa.add_argument('imagen')
    pa.add_argument('--salida',default='runs/imagen/validacion_adaptativa')
    args=pa.parse_args()
    origen=Path(args.imagen)
    frame=cv2.imread(str(origen))
    if frame is None: raise SystemExit(f'No se pudo abrir: {origen}')
    out=Path(args.salida);out.mkdir(parents=True,exist_ok=True)
    cv2.imwrite(str(out/'original.jpg'),frame)
    resumen={}
    for modo in ['adaptativo','gamma','clahe']:
        result, meta=mejorar_imagen(frame,modo=modo)
        cv2.imwrite(str(out/f'{modo}.jpg'),result)
        resumen[modo]=meta
        print(f'{modo}: {meta["operaciones"]} | brillo {meta["antes"]["luminosidad"]} -> {meta["despues"]["luminosidad"]}')
    (out/'metricas.json').write_text(json.dumps(resumen,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Resultados:',out.resolve())

if __name__=='__main__':main()
