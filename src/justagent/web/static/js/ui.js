/* JustAgent Legal Console — UI primitives (escaping, chat, toasts). */
'use strict';

const UI = (() => {
  function esc(s) {
    const d = document.createElement('div');
    d.textContent = s == null ? '' : String(s);
    return d.innerHTML;
  }

  let toastTimer = null;
  function toast(msg) {
    let el = document.getElementById('toast');
    if (!el) {
      el = document.createElement('div');
      el.id = 'toast';
      el.className = 'toast';
      document.body.appendChild(el);
    }
    el.textContent = msg;
    el.classList.add('show');
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => el.classList.remove('show'), 3200);
  }

  function addMsg(role, text) {
    const chat = document.getElementById('chat');
    const m = document.createElement('div');
    m.className = 'msg ' + role;
    m.innerHTML = '<div class="av">' + (role === 'user' ? '我' : 'J') + '</div>'
      + '<div class="b">' + esc(text) + '</div>';
    chat.appendChild(m);
    chat.scrollTop = chat.scrollHeight;
    return m.querySelector('.b');
  }

  function val(id) {
    const el = document.getElementById(id);
    return el ? el.value.trim() : '';
  }

  function fmtTs(ts) {
    if (!ts || ts <= 0) return 'never';
    return new Date(ts * 1000).toLocaleString();
  }

  function switchTab(name) {
    document.querySelectorAll('.tab').forEach(t => t.classList.toggle('active', t.dataset.panel === name));
    ['chat', 'cases', 'evidence', 'laws', 'system'].forEach(
      p => (document.getElementById('panel-' + p).style.display = p === name ? '' : 'none')
    );
    if (name === 'system' && typeof window.loadSystem === 'function') window.loadSystem();
  }

  return { esc, toast, addMsg, val, fmtTs, switchTab };
})();
window.UI = UI;
window.esc = UI.esc; window.addMsg = UI.addMsg; window.val = UI.val;
window.fmtTs = UI.fmtTs; window.switchTab = UI.switchTab;
