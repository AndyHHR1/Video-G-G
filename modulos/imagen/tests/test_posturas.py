"""Pruebas unitarias sin cargar modelos YOLO."""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / 'modulo_imagen'

def obtener_funcion(nombre):
    tree = ast.parse((ROOT / 'analisis_tiempo_real.py').read_text(encoding='utf-8'))
    funcion = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == nombre)
    espacio = {}
    exec(compile(ast.Module(body=[funcion], type_ignores=[]), '<prueba>', 'exec'), espacio)
    return espacio[nombre]

def test_pose_horizontal():
    evaluar = obtener_funcion('_evaluar_postura')
    assert evaluar({'bbox':[0,0,120,50], 'inclinacion_torso':77}) == 'postura_horizontal_posible'

def test_pose_solo_inclinacion():
    evaluar = obtener_funcion('_evaluar_postura')
    assert evaluar({'bbox':[0,0,30,120], 'inclinacion_torso':41}) == 'inclinacion_para_revision'

def test_pose_normal_y_no_evaluable():
    evaluar = obtener_funcion('_evaluar_postura')
    assert evaluar({'bbox':[0,0,30,120], 'inclinacion_torso':8}) == 'sin_indicio_suficiente'
    assert evaluar({'bbox':[], 'inclinacion_torso':None}) == 'no_evaluable'
