const API = (base = window.location.origin) => ({
  planes: () => fetch(`${base}/planes`).then(r => r.json()),
  logs: () => fetch(`${base}/logs`).then(r => r.json()),
  socios: (q = "") => fetch(`${base}/socios${q ? `?q=${encodeURIComponent(q)}` : ``}`).then(r => r.json()),
  altaSocio: (payload) => {
    const options = { method: 'POST' };
    if (payload instanceof FormData) {
      options.body = payload;
    } else {
      options.headers = { 'Content-Type': 'application/json' };
      options.body = JSON.stringify(payload);
    }
    return fetch(`${base}/socios`, options).then(r => {
      if (!r.ok) {
        throw new Error('Error creando socio');
      }
      return r.json();
    });
  },
  eliminarSocio: (dni) => fetch(`${base}/socios/${dni}`, { method: 'DELETE' }).then(r => r.json()),
});
