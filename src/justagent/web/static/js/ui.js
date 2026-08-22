/* JustAgent Legal Console — UI primitives. */
'use strict';

const UI = (() => {
  function esc(s) {
    const d = document.createElement('div');
    d.textContent = s == null ? '' : String(s);
    return d.innerHTML;
  }

  /* Markdown → sanitized HTML (stream-safe: marked tolerates partial input). */
  function md(text) {
    if (!text) return '';
    try {
      if (window.marked && window.DOMPurify) {
        return DOMPurify.sanitize(marked.parse(text));
      }
    } catch (e) { /* fall through to plain text */ }
    return '<p>' + esc(text) + '</p>';
  }

  let toastTimer = null;
  function toast(msg) {
    let el = document.getElementById('toast');
    if (!el) {
      el = document.createElement('div');
      el.id = 'toast'; el.className = 'toast';
      document.body.appendChild(el);
    }
    el.textContent = msg;
    el.classList.add('show');
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => el.classList.remove('show'), 3200);
  }

  /**
   * Append a message to the stream.
   * role: 'user' | 'assistant'; opts.markdown renders as rich HTML.
   * Returns the body element for in-place streaming updates.
   */
  function addMsg(role, text, opts) {
    const stream = document.getElementById('stream');
    const o = opts || {};
    const wrap = document.createElement('div');
    wrap.className = 'msg ' + role;
    const who = role === 'user' ? '你' : 'JustAgent';
    let bodyHtml = o.markdown ? md(text) : esc(text);
    if (o.caret) bodyHtml += '<span class="caret"></span>';
    wrap.innerHTML = `<div class="who">${who}</div><div class="body ${o.caret ? 'caret' : ''}">${bodyHtml}</div>`;
    const empty = document.getElementById('empty-state');
    if (empty) empty.remove();
    stream.appendChild(wrap);
    const scroll = document.getElementById('stream-scroll');
    scroll.scrollTop = scroll.scrollHeight;
    return wrap.querySelector('.body');
  }

  /** Update a streaming message body; keeps the caret while active. */
  function updateMsg(bodyEl, html, done) {
    bodyEl.innerHTML = html + (done ? '' : '<span class="caret"></span>');
    const scroll = document.getElementById('stream-scroll');
    const nearBottom = scroll.scrollHeight - scroll.scrollTop - scroll.clientHeight < 120;
    if (nearBottom) scroll.scrollTop = scroll.scrollHeight;
  }

  function val(id) {
    const el = document.getElementById(id);
    return el ? el.value.trim() : '';
  }

  function fmtTs(ts) {
    if (!ts || ts <= 0) return 'never';
    return new Date(ts * 1000).toLocaleString();
  }

  function switchView(name) {
    document.querySelectorAll('#main-nav button').forEach(
      b => b.classList.toggle('active', b.dataset.view === name)
    );
    document.querySelectorAll('.view').forEach(v => v.classList.toggle('active', v.id === 'view-' + name));
    if (name === 'system' && typeof window.loadSystem === 'function') window.loadSystem();
    if (name !== 'chat' && typeof window.loadState === 'function') window.loadState();
  }

  return { esc, md, toast, addMsg, updateMsg, val, fmtTs, switchView };
})();
window.UI = UI;
window.esc = UI.esc; window.addMsg = UI.addMsg; window.val = UI.val;
window.fmtTs = UI.fmtTs; window.switchView = UI.switchView;
