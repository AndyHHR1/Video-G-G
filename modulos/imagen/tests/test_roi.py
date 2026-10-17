from modulos.imagen.modulo_imagen.recuperacion_roi import crear_roi

def test_roi_dentro_de_imagen():
    x1,y1,x2,y2 = crear_roi([250,210,360,300], (478,848,3))
    assert 0 <= x1 < x2 <= 848 and 0 <= y1 < y2 <= 478
    assert x1 <= 250 and x2 >= 360 and y1 <= 210 and y2 >= 300

def test_roi_borde():
    x1,y1,x2,y2 = crear_roi([810,0,848,100], (478,848,3))
    assert 0 <= x1 < x2 <= 848 and 0 <= y1 < y2 <= 478
