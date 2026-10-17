from modulos.imagen.modulo_imagen.analizador_clip import tiempos_muestreo


def test_intervalos():
    assert tiempos_muestreo(5, 1) == [0,1,2,3,4]


def test_tiempos_interes():
    assert tiempos_muestreo(10, tiempos_interes=[3.5, 1, 3.5]) == [1.0,3.5]


def test_limites():
    import pytest
    with pytest.raises(ValueError):
        tiempos_muestreo(5, tiempos_interes=[6])
