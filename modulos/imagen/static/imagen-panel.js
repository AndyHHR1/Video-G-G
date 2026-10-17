/* Panel «Imagen» (modulos/imagen). Lógica de la página de pruebas del grupo
 * de imagen (imagen_pruebas.html) llevada a una pestaña del panel principal:
 * fotografía o cámara, un fotograma en vuelo como máximo (no se encolan
 * peticiones) y, además, un resumen legible del JSON. */
(() => {
  const $i = id => document.getElementById(id);
  if (!$i('p-imagen')) return;
  let stream = null, activo = false, ocupado = false;
  const video = $i('img-camara'), estado = $i('img-estado');
  const canvas = document.createElement('canvas');

  const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
  const fila = (k, v) => `<div class="stat"><span>${esc(k)}</span><b>${esc(v)}</b></div>`;

  function resumen(d) {
    if (d.estado !== 'ok') return `<h2>Resumen · módulo Imagen</h2><div class="req">${esc(d.mensaje || 'Error')}</div>`;
    const c = d.calidad_original || {};
    const pre = d.preprocesamiento || {};
    const pos = (d.personas || []).map(p => p.estado_postural.replace(/_/g, ' '));
    let h = '<h2>Resumen · módulo Imagen</h2>';
    h += fila('Dimensiones', `${d.dimensiones.ancho}×${d.dimensiones.alto}`);
    h += fila('Personas (YOLO11n)', (d.personas || []).length);
    h += fila('Poses sin persona YOLO', (d.poses_sin_asociar || []).length);
    h += fila('Celulares', (d.celulares || []).length);
    if (pos.length) h += fila('Estado postural', pos.join(', '));
    h += fila('Luminosidad', c.luminosidad);
    h += fila('Contraste', c.contraste);
    h += fila('Nitidez (Laplaciano)', c.nitidez_laplaciana);
    h += fila('Preprocesamiento', (pre.operaciones || []).join(', ') || 'ninguno');
    const t = d.tiempos_ms || {};
    h += fila('Tiempo total', `${d.tiempo_procesamiento_ms} ms`);
    h += fila('Pre / det / pose', `${t.preprocesamiento} / ${t.deteccion} / ${t.pose} ms`);
    return h;
  }

  async function enviar(blob) {
    if (ocupado || !blob) return;
    ocupado = true;
    try {
      estado.textContent = 'Analizando fotograma…';
      const f = new FormData();
      f.append('frame', blob, 'frame.jpg');
      f.append('modo', $i('img-modo').value);
      const r = await fetch('/api/imagen/pruebas/frame', {method:'POST', body:f});
      const data = await r.json();
      if (data.imagen_anotada) {
        $i('img-stage').innerHTML = `<img src="${data.imagen_anotada}" alt="Resultado con detecciones">`;
        delete data.imagen_anotada;
      }
      $i('img-datos').textContent = JSON.stringify(data, null, 2);
      $i('img-resumen').innerHTML = resumen(data);
      const adv = data.advertencias || [];
      $i('img-advertencias').innerHTML = adv.length
        ? '<ul style="padding-left:16px;line-height:1.6">' + adv.map(a => `<li>${esc(a)}</li>`).join('') + '</ul>'
        : 'Ninguna.';
      estado.textContent = r.ok ? `Análisis completado · ${data.tiempo_procesamiento_ms} ms`
                                : 'Error: ' + (data.mensaje || r.status);
    } catch (e) {
      estado.textContent = 'Error de comunicación: ' + e.message;
    } finally {
      ocupado = false;
    }
  }

  $i('img-evaluar').onclick = () => {
    const f = $i('img-archivo').files[0];
    if (!f) { estado.textContent = 'Selecciona una imagen'; return; }
    enviar(f);
  };

  function capturar() {
    if (!activo) return;
    if (!ocupado && video.videoWidth) {
      canvas.width = Math.min(video.videoWidth, 960);
      canvas.height = Math.round(video.videoHeight * canvas.width / video.videoWidth);
      canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);
      canvas.toBlob(blob => enviar(blob), 'image/jpeg', 0.8);
    }
    setTimeout(capturar, 150);
  }

  function detener(msg = 'Cámara detenida') {
    activo = false;
    if (stream) { stream.getTracks().forEach(t => t.stop()); stream = null; }
    video.srcObject = null; video.style.display = 'none';
    $i('img-iniciar').disabled = false; $i('img-detener').disabled = true;
    estado.textContent = msg;
  }

  $i('img-iniciar').onclick = async () => {
    if (activo) return;
    try {
      stream = await navigator.mediaDevices.getUserMedia({video:{width:960, height:540}, audio:false});
      video.srcObject = stream; video.style.display = 'block';
      await video.play();
      activo = true;
      $i('img-iniciar').disabled = true; $i('img-detener').disabled = false;
      capturar();
    } catch (e) {
      estado.textContent = 'No se pudo abrir cámara: ' + e.message;
    }
  };
  $i('img-detener').onclick = () => detener();

  $i('img-features').onclick = async () => {
    try {
      const r = await fetch('/api/imagen/caracteristicas');
      const d = await r.json();
      $i('img-datos').textContent = JSON.stringify(d, null, 2);
      estado.textContent = r.ok ? 'Vector de características del último fotograma en vivo'
                                : (d.mensaje || 'Sin datos');
    } catch (e) { estado.textContent = 'Error: ' + e.message; }
  };

  window.addEventListener('beforeunload', () => detener());
  // El panel principal llama a esto al cambiar de pestaña o detener el sistema.
  window.imagenPanel = { stop: () => { if (activo) detener(); } };
})();
