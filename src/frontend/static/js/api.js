if (typeof window !== 'undefined' && (typeof window.API !== 'function')) {
  const resolveBase = (explicitBase) => {
    if (explicitBase) {
      return explicitBase;
    }
    const origin = window.location && window.location.origin;
    if (!origin || origin === 'null' || origin.startsWith('file:')) {
      return 'http://localhost:8000';
    }
    return origin;
  };

  const safeJson = async (response) => {
    const text = await response.text();
    try {
      return text ? JSON.parse(text) : {};
    } catch (err) {
      console.error('Respuesta JSON inválida', err, text);
      throw new Error('El servidor devolvió una respuesta inválida.');
    }
  };

  const fetchJson = async (input, init, friendlyError) => {
    let response;
    try {
      response = await fetch(input, init);
    } catch (err) {
      console.error('Fallo al conectar con el backend', err);
      throw new Error(friendlyError || 'No se pudo contactar al backend. Verificá que esté iniciado.');
    }

    const data = await safeJson(response);
    if (!response.ok) {
      const message = (data && data.detail) || friendlyError || 'Operación rechazada por el backend.';
      throw new Error(message);
    }
    return data;
  };

  const buildApi = (base = undefined) => {
    const resolvedBase = resolveBase(base);

    const api = {
      planes: () => fetchJson(`${resolvedBase}/planes`, undefined, 'No se pudieron obtener los planes.'),
      logs: () => fetchJson(`${resolvedBase}/logs`, undefined, 'No se pudieron obtener los logs.'),
      socios: (q = '') => {
        const url = `${resolvedBase}/socios${q ? `?q=${encodeURIComponent(q)}` : ''}`;
        return fetchJson(url, undefined, 'No se pudieron obtener los socios.');
      },
      altaSocio: (payload) => fetchJson(
        `${resolvedBase}/socios`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload),
        },
        'No se pudo dar de alta al socio.'
      ),
      renovarMembresia: (dni, id_plan, dias = 30) => fetchJson(
        `${resolvedBase}/membresias/renovar?dni=${dni}&id_plan=${id_plan}&dias=${dias}`,
        { method: 'POST' },
        'No se pudo renovar la membresía.'
),

    };
    

    api.rostroDesdeVideo = async (dni, file) => {
      if (typeof FormData === 'undefined') {
        throw new Error('El navegador no soporta carga de videos para embeddings.');
      }

      const allowedTypes = ['video/mp4', 'video/webm'];
      if (file && file.type && !allowedTypes.includes(file.type)) {
        throw new Error('Formato de video no soportado. Subí un archivo MP4 o WebM.');
      }

      const form = new FormData();
      form.append('video', file);

      return fetchJson(
        `${resolvedBase}/socios/${dni}/rostro-video`,
        {
          method: 'POST',
          body: form,
        },
        'No se pudo generar el embedding desde el video.'
      );
    };

    return api;
  };

  window.API = buildApi;
}

if (typeof window === 'object' && typeof window.API === 'function' && typeof window.API.rostroDesdeVideo !== 'function') {
  window.API.rostroDesdeVideo = (dni, file, base) => window.API(base).rostroDesdeVideo(dni, file);
}
