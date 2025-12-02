// script.js - global helpers (used by templates)
async function fetchWhoami(){
    try {
      const r = await fetch('/api/whoami');
      if (!r.ok) return null;
      return await r.json();
    } catch(e){ return null; }
  }
  
  async function populateTopRight(){
    const info = await fetchWhoami();
    const el = document.getElementById('topRight');
    if (!el) return;
    if (info && info.username){
      el.innerHTML = `<div style="text-align:right"><div style="font-weight:600">${info.username}</div><div style="font-size:0.9rem;color:#6b7280">${info.physio_id ? 'Fisioterapeuta' : ''}</div></div>`;
    } else {
      el.innerHTML = `<a href="/login" class="btn">Entrar</a>`;
    }
  }
  populateTopRight();  