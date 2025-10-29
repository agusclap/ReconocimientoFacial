if (!window.API) {
  window.API = (base = window.location.origin) => ({
    planes: () => fetch(`${base}/planes`).then(r => r.json()),
    logs: () => fetch(`${base}/logs`).then(r => r.json()),
    socios: (q = "") => fetch(`${base}/socios${q ? `?q=${encodeURIComponent(q)}` : ``}`).then(r => r.json()),
    altaSocio: (payload) => fetch(`${base}/socios`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    }).then(r => r.json()),
    eliminarSocio: (dni) => fetch(`${base}/socios/${dni}`, { method: 'DELETE' }).then(r => r.json()),
    rostroDesdeVideo: async (dni, file) => {
      const allowedTypes = ['video/mp4', 'video/webm'];
      if (file && file.type && !allowedTypes.includes(file.type)) {
        throw new Error('Formato de video no soportado. Subí un archivo MP4 o WebM.');
      }
      const form = new FormData();
      form.append('video', file);
      const res = await fetch(`${base}/socios/${dni}/rostro-video`, {
        method: 'POST',
        body: form
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
        const message = data && data.detail ? data.detail : 'No se pudo generar el embedding desde el video.';
        throw new Error(message);
      }
      return data;
    },
  });
}
