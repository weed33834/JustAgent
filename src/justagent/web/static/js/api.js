/* JustAgent Legal Console — unified API layer.
 * Replaces the old window.fetch monkey-patch: every call goes through
 * api.get/apiPost so auth headers, project routing, JSON parsing and
 * 401 re-login are handled in exactly one place. */
'use strict';

const API = (() => {
  let token = localStorage.getItem('justagent_token') || '';
  let project = '';

  function headers(extra) {
    const h = Object.assign({}, extra || {});
    if (token) h['Authorization'] = 'Bearer ' + token;
    if (project) h['X-JustAgent-Project'] = project;
    return h;
  }

  async function request(method, url, body, opts) {
    const o = opts || {};
    const init = { method, headers: headers(o.headers) };
    if (o.signal) init.signal = o.signal;
    if (body !== undefined) {
      init.body = o.raw ? body : JSON.stringify(body);
      if (!o.raw) init.headers['Content-Type'] = 'application/json';
    }
    const res = await fetch(url, init);
    if (res.status === 401 && !o.noReauth && typeof window.doLogin === 'function') {
      await window.doLogin();
      return request(method, url, body, Object.assign({}, o, { noReauth: true }));
    }
    if (!res.ok) {
      const err = new Error('HTTP ' + res.status);
      err.status = res.status;
      try { err.payload = await res.json(); } catch (e) { /* no body */ }
      throw err;
    }
    return o.raw ? res : res.json();
  }

  return {
    get: (u, o) => request('GET', u, undefined, o),
    post: (u, b, o) => request('POST', u, b, o),
    del: (u, o) => request('DELETE', u, undefined, o),
    rawResponse: (u, o) => request('GET', u, undefined, Object.assign({ raw: true }, o)),
    setToken(t) { token = t || ''; if (t) localStorage.setItem('justagent_token', t); else localStorage.removeItem('justagent_token'); },
    getToken() { return token; },
    setProject(p) { project = p || ''; },
    getProject() { return project; },
  };
})();
window.API = API;
