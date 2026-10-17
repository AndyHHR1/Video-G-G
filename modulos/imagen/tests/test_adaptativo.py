import numpy as np
import pytest
from modulos.imagen.modulo_imagen.calidad import medir_calidad
from modulos.imagen.modulo_imagen.preprocesamiento import mejorar_imagen

def test_imagen_correcta_permanece_igual():
    img=np.full((128,128,3),135,dtype=np.uint8)
    salida,meta=mejorar_imagen(img)
    assert np.array_equal(img,salida)
    assert not meta['operaciones']

def test_oscura_se_aclara_sin_mutacion():
    original=np.full((100,100,3),35,dtype=np.uint8)
    copia=original.copy()
    salida,meta=mejorar_imagen(original)
    assert salida.mean()>original.mean()
    assert np.array_equal(original,copia)
    assert meta['color_salida']=='BGR'

def test_entrada_invalida():
    with pytest.raises(ValueError): medir_calidad(np.zeros((10,10),dtype=np.uint8))
