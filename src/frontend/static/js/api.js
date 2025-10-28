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
  });
}
