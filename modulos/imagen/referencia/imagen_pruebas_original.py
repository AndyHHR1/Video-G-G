"""Blueprint autónomo. Registro opcional en Flask existente, sin alterar Video."""
from flask import Blueprint, request, jsonify, render_template
import cv2
import numpy as np

imagen_bp = Blueprint('imagen_pruebas', __name__, template_folder='templates')

@imagen_bp.get('/imagen/pruebas')
def pagina_imagen():
    return render_template('imagen_pruebas.html')

@imagen_bp.post('/api/imagen/pruebas/frame')
def frame_imagen():
    archivo = request.files.get('frame')
    if archivo is None:
        return jsonify({'estado':'error','mensaje':'Falta frame'}),400
    if request.content_length and request.content_length > 7 * 1024 * 1024:
        return jsonify({'estado':'error','mensaje':'Fotografía demasiado grande'}),413
    contenido = archivo.read(7*1024*1024+1)
    if len(contenido)>7*1024*1024:
        return jsonify({'estado':'error','mensaje':'Fotografía demasiado grande'}),413
    frame = cv2.imdecode(np.frombuffer(contenido, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({'estado':'error','mensaje':'Archivo no es una imagen válida'}),400
    if frame.shape[0]*frame.shape[1] > 12_000_000:
        return jsonify({'estado':'error','mensaje':'Resolución demasiado grande'}),413
    modo = request.form.get('modo','adaptativo')
    if modo not in ('original','adaptativo','clahe','gamma'):
        return jsonify({'estado':'error','mensaje':'Modo inválido'}),400
    try:
        from .modulo_imagen.analisis_tiempo_real import analizar_frame
        datos, anotada = analizar_frame(frame,modo=modo)
        ok, jpg = cv2.imencode('.jpg', anotada, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            import base64
            datos['imagen_anotada'] = 'data:image/jpeg;base64,'+base64.b64encode(jpg).decode('ascii')
        return jsonify(datos)
    except Exception as exc:
        # Solo tipo de error para evitar exponer paths internos y stack traces.
        imagen_bp.logger.exception('Fallo en inferencia de Imagen') if hasattr(imagen_bp,'logger') else None
        return jsonify({'estado':'error','mensaje':'Fallo de análisis','tipo':type(exc).__name__}),500
