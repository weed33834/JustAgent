/* JustAgent Legal Console — business logic.
 * Chat-first layout following mainstream AI-console conventions:
 * session sidebar, centered streaming column with stop control,
 * markdown rendering (marked + DOMPurify), starter-card empty state. */
'use strict';

const State = { cases: [], evidence: [], laws: [] };
const Chat = { history: [], attachedImage: null, controller: null, streaming: false };

/* -- sessions ------------------------------------------------------------- */

function getSession() {
  let s = localStorage.getItem('justagent_session');
  if (!s) { s = newSessionId(); localStorage.setItem('justagent_session', s); }
  return s;
}
function newSessionId() {
  return 's' + Date.now().toString(36) + Math.random().toString(36).slice(2, 8);
}

async function loadSessions() {
  const el = document.getElementById('sessions-list');
  try {
    const r = await API.get('/api/sessions');
    const current = getSession();
    el.innerHTML = r.items.length
      ? r.items.map(s => `
        <div class="session-item ${s.id === current ? 'active' : ''}" onclick="switchSession('${s.id}')">
          <span>${esc((s.title || '').slice(0, 26) || s.id.slice(0, 10))}</span>
          <button class="del" title="删除" onclick="event.stopPropagation();delSession('${s.id}')">✕</button>
        </div>`).join('')
      : '<div class="empty" style="padding:8px 4px">暂无历史</div>';
  } catch (e) { /* sidebar history is non-critical */ }
}

async function switchSession(id) {
  localStorage.setItem('justagent_session', id);
  newChat(false);
  loadSessions();
}

function newChat(clearList = true) {
  localStorage.setItem('justagent_session', newSessionId());
  Chat.history = [];
  const stream = document.getElementById('stream');
  stream.innerHTML = document.getElementById('empty-state-template')?.innerHTML
    || stream.innerHTML; // keep original empty-state markup
  if (clearList) loadSessions();
}

async function delSession(id) {
  try {
    await API.del('/api/sessions/' + id);
    if (id === getSession()) newChat(false);
    loadSessions();
  } catch (e) { UI.toast('删除失败: ' + e.message); }
}

/* -- projects -------------------------------------------------------------- */

async function loadProjects() {
  try {
    const r = await API.get('/api/projects');
    const sel = document.getElementById('projectSel');
    sel.innerHTML = '<option value="">默认项目</option>'
      + r.items.map(p => `<option value="${esc(p.name)}">${esc(p.name)}</option>`).join('');
    sel.value = API.getProject();
  } catch (e) { UI.toast('项目列表加载失败: ' + e.message); }
}

function switchProject() {
  const p = document.getElementById('projectSel').value;
  API.setProject(p);
  loadState();
  UI.toast(p ? '已切换到项目 ' + p : '已切换到默认项目');
}

/* -- state / rendering ------------------------------------------------------ */

async function loadState() {
  try { Object.assign(State, await API.get('/api/state')); }
  catch (e) { Object.assign(State, { cases: [], evidence: [], laws: [] }); }
  renderCases(); renderEvidence(); renderLaws();
}

function renderCases() {
  const el = document.getElementById('case-list');
  el.innerHTML = State.cases.length
    ? State.cases.map(c => `
      <div class="item">
        <div class="t">${esc(c.case_number || c.id.slice(0, 8))}<span class="badge">${esc(c.status)}</span></div>
        <div class="d">${esc(c.cause)} · ${esc(c.court)} · ${c.parties} 当事人 · ${c.timeline} 时间线</div>
        <button class="btn" style="margin-top:6px" onclick="generateDoc('${c.id}')">生成文书</button>
      </div>`).join('')
    : '<div class="empty">暂无案件</div>';
}

function renderEvidence() {
  const el = document.getElementById('evidence-list');
  const buttons = '<button class="btn" style="margin-top:8px" onclick="analyzeEvidence()">🔍 证据链分析</button> '
    + '<button class="btn secondary" style="margin-top:8px" onclick="auditEvidence()">⚖️ 证据链审计</button>';
  el.innerHTML = (State.evidence.length
    ? State.evidence.map(e => `
      <div class="item">
        <div class="t">${esc(e.name)}<span class="badge">${esc(e.admissible)}</span></div>
        <div class="d">${esc(e.type)} · 证明力: ${esc(e.strength)}</div>
      </div>`).join('')
    : '<div class="empty">暂无证据</div>') + buttons;
}

function renderLaws() {
  const el = document.getElementById('law-list');
  el.innerHTML = State.laws.length
    ? State.laws.map(l => `
      <div class="item">
        <div class="t">${esc(l.citation)}</div>
        <div class="d">${esc(l.law_name)} · ${esc(l.domain)} · ${esc(l.status)}</div>
      </div>`).join('')
    : '<div class="empty">暂无法条</div>';
}

async function loadDocTypes() {
  try {
    const r = await API.get('/api/judicial/doc/types');
    document.getElementById('docType').innerHTML =
      r.items.map(t => `<option value="${esc(t.id)}">${esc(t.name)}</option>`).join('');
  } catch (e) { /* optional decoration */ }
}

/* -- chat: send / stop / stream ---------------------------------------------- */

function autoGrow(el) {
  el.style.height = 'auto';
  el.style.height = Math.min(el.scrollHeight, 180) + 'px';
}

function useStarter(text) {
  const input = document.getElementById('input');
  input.value = text;
  autoGrow(input);
  send();
}

function setStreamingUI(on) {
  document.getElementById('stop-row').style.display = on ? '' : 'none';
  document.getElementById('send-btn').disabled = on;
}

function stopStreaming() {
  if (Chat.controller) Chat.controller.abort();
}

async function send() {
  if (Chat.streaming) return;
  const input = document.getElementById('input');
  const text = input.value.trim();
  if (!text) return;
  input.value = ''; autoGrow(input);
  addMsg('user', text);
  Chat.history.push({ role: 'user', content: text });

  if (Chat.attachedImage) { await sendVision(text); Chat.attachedImage = null; return; }

  Chat.streaming = true;
  setStreamingUI(true);
  Chat.controller = new AbortController();

  const bubble = UI.addMsg('assistant', '', { caret: true });
  let reply = '';
  let toolBlocksHtml = '';
  const toolsSeen = [];

  try {
    const res = await fetch('/api/chat/stream', {
      method: 'POST',
      headers: Object.assign({ 'Content-Type': 'application/json' }, (() => {
        const t = API.getToken();
        const h = {};
        if (t) h['Authorization'] = 'Bearer ' + t;
        if (API.getProject()) h['X-JustAgent-Project'] = API.getProject();
        return h;
      })()),
      body: JSON.stringify({ message: text, history: Chat.history, session_id: getSession() }),
      signal: Chat.controller.signal,
    });
    if (res.status === 401) { await doLogin(); bubble.remove(); setStreamingUI(false); Chat.streaming = false; return; }
    if (!res.ok || !res.body) {
      const d = await res.json().catch(() => ({}));
      UI.updateMsg(bubble, UI.esc(d.reply || ('请求失败: ' + res.status)), true);
      setStreamingUI(false); Chat.streaming = false;
      return;
    }
    const reader = res.body.getReader();
    const dec = new TextDecoder();
    let buf = '';
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      let idx;
      while ((idx = buf.indexOf('\n\n')) >= 0) {
        const line = buf.slice(0, idx);
        buf = buf.slice(idx + 2);
        if (!line.startsWith('data: ')) continue;
        let ev;
        try { ev = JSON.parse(line.slice(6)); } catch (e) { continue; }
        if (ev.type === 'delta') {
          reply += ev.content || '';
          UI.updateMsg(bubble, UI.md(reply) + toolBlocksHtml);
        } else if (ev.type === 'tool_start' && !toolsSeen.includes(ev.tool)) {
          toolsSeen.push(ev.tool);
          toolBlocksHtml += `<details class="tool-block"><summary>🔧 调用工具：${esc(ev.tool)}</summary><div style="padding:6px 12px">执行中…</div></details>`;
          UI.updateMsg(bubble, UI.md(reply) + toolBlocksHtml);
        } else if (ev.type === 'done') {
          reply = ev.content || reply;
          break;
        }
      }
      if (reply && buf.indexOf('"type": "done"') !== -1) break;
    }
    UI.updateMsg(bubble, UI.md(reply) + toolBlocksHtml, true);
    Chat.history.push({ role: 'assistant', content: reply });
    await loadState();
  } catch (e) {
    if (e.name === 'AbortError') {
      UI.updateMsg(bubble, UI.md(reply || '（已停止）'), true);
    } else {
      UI.updateMsg(bubble, UI.esc('请求失败: ' + e.message), true);
    }
  } finally {
    setStreamingUI(false); Chat.streaming = false; Chat.controller = null;
  }
}

async function sendVision(text) {
  const bubble = UI.addMsg('assistant', '', { caret: true });
  try {
    const d = await API.post('/api/vision', {
      prompt: text || '请描述这张图片并提取关键信息', image: Chat.attachedImage,
    });
    UI.updateMsg(bubble, UI.md(d.reply || '(无输出)'), true);
    Chat.history.push({ role: 'assistant', content: d.reply || '' });
  } catch (e) { UI.updateMsg(bubble, UI.esc('识别失败: ' + e.message), true); }
}

/* -- auth ---------------------------------------------------------------------- */

async function doLogin() {
  const cred = prompt('需要登录。输入 用户名:密码（或直接粘贴访问令牌）：');
  if (!cred) return;
  if (cred.indexOf(':') >= 0) {
    const [u, p] = cred.split(':');
    const r = await fetch('/api/auth/login', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username: u.trim(), password: p.trim() }),
    });
    if (r.ok) {
      const d = await r.json();
      API.setToken(d.token);
      UI.toast('已登录为 ' + d.username + '（' + d.role + '）');
    } else UI.toast('登录失败：' + r.status);
  } else {
    API.setToken(cred);
    UI.toast('已设置访问令牌');
  }
}
window.doLogin = doLogin;

function logout() { API.setToken(''); UI.toast('已登出'); }

/* -- system panel ---------------------------------------------------------------- */

async function loadSystem() {
  const set = (id, html) => (document.getElementById(id).innerHTML = html);

  try {
    const doctor = await API.post('/api/doctor');
    set('sys-doctor', doctor.map(c =>
      `<div class="item"><div class="t">${esc(c.name)} <span class="badge">${esc(c.status)}</span></div><div class="d">${esc(c.detail || '')}</div></div>`).join(''));
  } catch (e) { set('sys-doctor', '<div class="empty">诊断不可用</div>'); }

  try {
    const models = await API.get('/api/models');
    set('sys-config',
      '<div class="item"><div class="t">模型后端</div><div class="d">' + models.items.length + ' 个</div></div>'
      + models.items.map(m => `<div class="item"><div class="t">${esc(m.provider)} <span class="badge">${esc(m.model)}</span></div><div class="d">${esc(m.base_url)} · key ${esc(m.key)}</div></div>`).join(''));
  } catch (e) { UI.toast('配置加载失败: ' + e.message); }

  try {
    const pl = await API.get('/api/plugins');
    set('sys-plugins', pl.items.length
      ? pl.items.map(p => `<div class="item"><div class="t">${esc(p.name)}</div><div class="d">${esc(p.version)} · ${esc(p.trust)}</div></div>`).join('')
      : '<div class="empty">暂无插件</div>');
  } catch (e) { set('sys-plugins', '<div class="empty">插件不可用</div>'); }

  try {
    const tk = await API.get('/api/schedule');
    set('sys-tasks', tk.items.length
      ? tk.items.map(t => `<div class="item"><div class="t">${esc(t.name)} <span class="badge">${t.enabled ? '启用' : '停用'}</span></div><div class="d">${esc(t.schedule)} · 上次 ${fmtTs(t.last_run)}</div></div>`).join('')
      : '<div class="empty">暂无定时任务</div>');
  } catch (e) { set('sys-tasks', '<div class="empty">定时任务不可用</div>'); }

  try {
    const m = await API.get('/api/metrics');
    const evs = Object.entries(m.events || {}).sort((a, b) => b[1] - a[1]).slice(0, 8);
    if (window.echarts && evs.length) {
      echarts.init(document.getElementById('metrics-chart')).setOption({
        tooltip: { trigger: 'axis' },
        xAxis: { type: 'category', data: evs.map(e => e[0]), axisLabel: { rotate: 30, fontSize: 9 } },
        yAxis: { type: 'value' },
        series: [{ type: 'bar', data: evs.map(e => e[1]), itemStyle: { color: '#4f46e5' } }],
      });
    }
  } catch (e) { /* decorative */ }

  try {
    const n = await API.get('/api/notifications');
    set('sys-notify', n.items.length
      ? n.items.slice(0, 20).map(x => `<div class="item"><div class="t">${esc(x.kind)} <span class="badge">${new Date(x.ts * 1000).toLocaleTimeString()}</span></div><div class="d">${esc(x.message)}</div></div>`).join('')
      : '<div class="empty">暂无通知</div>');
  } catch (e) { set('sys-notify', '<div class="empty">暂无通知</div>'); }

  try {
    const a = await API.get('/api/audit');
    set('sys-audit', a.items.length
      ? a.items.slice(0, 20).map(x => `<div class="item"><div class="t">${esc(x.event || x.action || '')}</div><div class="d">${esc((x.trace_id || '').slice(0, 8))} · ${esc(String(x.ts || '').slice(0, 19))}</div></div>`).join('')
      : '<div class="empty">暂无审计</div>');
  } catch (e) { set('sys-audit', '<div class="empty">审计不可用</div>'); }

  try {
    const st = await API.get('/api/state');
    if (window.echarts) {
      const byStatus = {};
      st.cases.forEach(cc => (byStatus[cc.status] = (byStatus[cc.status] || 0) + 1));
      const sd = Object.entries(byStatus);
      if (sd.length) echarts.init(document.getElementById('cases-chart')).setOption({
        series: [{ type: 'pie', radius: '60%', data: sd.map(([n2, v]) => ({ name: n2, value: v })) }],
      });
      const byDomain = {};
      st.laws.forEach(l => (byDomain[l.domain] = (byDomain[l.domain] || 0) + 1));
      const dom = Object.entries(byDomain);
      if (dom.length) echarts.init(document.getElementById('laws-chart')).setOption({
        tooltip: {}, xAxis: { type: 'category', data: dom.map(d => d[0]) }, yAxis: { type: 'value' },
        series: [{ type: 'bar', data: dom.map(d => d[1]), itemStyle: { color: '#7c3aed' } }],
      });
    }
  } catch (e) { /* decorative */ }

  try {
    const pr = await API.get('/api/projects');
    set('sys-projects', '<div class="item"><div class="t">当前项目</div><div class="d">' + esc(pr.current) + '</div></div>'
      + pr.items.map(p => `<div class="item"><div class="t">${esc(p.name)}</div><div class="d">${esc(p.path)}</div></div>`).join(''));
  } catch (e) { set('sys-projects', '<div class="empty">项目不可用</div>'); }
}

/* -- legal features --------------------------------------------------------------- */

async function searchLaw() {
  const q = val('ks-q');
  if (!q) return;
  try {
    const d = await API.post('/api/knowledge/search', { query: q, top_k: 6 });
    document.getElementById('law-search').innerHTML = d.hits.length
      ? d.hits.map(h => `<div class="item"><div class="t">${esc(h.citation)} <span class="badge">${h.score.toFixed(2)}</span></div><div class="d">${esc(h.content || '')}</div></div>`).join('')
      : '<div class="empty">无结果</div>';
  } catch (e) { UI.toast('检索失败: ' + e.message); }
}

async function analyzeEvidence(caseId) {
  try {
    const d = await API.post('/api/judicial/evidence/analyze', { case_id: caseId || '' });
    UI.toast('完整度 ' + d.completeness_score + '，矛盾 ' + d.contradiction_count + ' 处');
    switchView('chat');
    UI.addMsg('assistant', '证据链分析：完整度 ' + d.completeness_score + '，矛盾 ' + d.contradiction_count + ' 处\n' + (d.summary || ''), { markdown: false });
  } catch (e) { UI.toast('分析失败: ' + e.message); }
}

async function auditEvidence(caseId) {
  try {
    const d = await API.post('/api/judicial/evidence/audit', { case_id: caseId || '' });
    const icon = { '通过': '✅', '有瑕疵': '⚠️' }[d.verdict] || '❌';
    const cov = (d.claim_coverage || []).map(c => (c.covered ? '✅ ' : '❌ ') + c.claim_description).join('\n');
    const lines = [
      icon + ' **证据链审计结论：' + d.verdict + '**（完整度 ' + (d.chain ? d.chain.completeness_score : '-') + '）',
      '- 保管链条问题：' + (d.custody_issues || []).length,
      ...(d.custody_issues || []).map(i => '  - ' + i),
      '- 时间线问题：' + (d.timeline_issues || []).length,
      ...(d.timeline_issues || []).map(i => '  - ' + i),
      '- 同源佐证警告：' + (d.independence_warnings || []).length,
      ...(d.independence_warnings || []).map(i => '  - ' + i),
    ];
    if (cov) lines.push('\n**诉请覆盖**\n' + cov);
    if (d.summary) lines.push('\n' + d.summary);
    switchView('chat');
    UI.addMsg('assistant', lines.join('\n'), { markdown: true });
  } catch (e) { UI.toast('审计失败: ' + e.message); }
}

async function generateDoc(caseId) {
  const dt = document.getElementById('docType').value || 'evidence_list';
  try {
    const d = await API.post('/api/judicial/doc', { case_id: caseId, doc_type: dt });
    UI.toast(d.ok ? '文书已生成' : '生成失败: ' + (d.error || ''));
    if (d.ok) {
      switchView('chat');
      UI.addMsg('assistant', '《' + dt + '》已生成：\n\n```\n' + (d.content || '').slice(0, 2000) + '\n```', { markdown: true });
    }
  } catch (e) { UI.toast('文书生成失败: ' + e.message); }
}

async function exportState() {
  const blob = new Blob([JSON.stringify(State, null, 2)], { type: 'application/json' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'justagent-state.json';
  a.click();
  URL.revokeObjectURL(a.href);
}

async function createCase() {
  try {
    const d = await API.post('/api/judicial/case', {
      case_number: val('c-no'), cause: val('c-cause'), court: val('c-court'),
    });
    UI.toast(d.ok ? '已创建案件 ' + d.case_number : '创建失败: ' + d.error);
  } catch (e) { UI.toast('创建案件失败: ' + e.message); }
  await loadState();
}

async function addLaw() {
  try {
    const d = await API.post('/api/judicial/law', {
      law_name: val('l-name'), article_number: val('l-article'),
      content: val('l-content'), domain: val('l-domain'),
    });
    UI.toast(d.ok ? '已添加法条 ' + d.citation : '添加失败: ' + d.error);
  } catch (e) { UI.toast('添加法条失败: ' + e.message); }
  await loadState();
}

/* -- attachments / voice ------------------------------------------------------------ */

let recognizing = false;
let recognition = null;

function pickImage() {
  const f = document.createElement('input');
  f.type = 'file'; f.accept = 'image/*';
  f.onchange = () => {
    const file = f.files[0];
    if (!file) return;
    const r = new FileReader();
    r.onload = () => (Chat.attachedImage = r.result);
    r.readAsDataURL(file);
    UI.toast('图片已附上，发送后开始识别');
  };
  f.click();
}

function pickFile() {
  const f = document.createElement('input');
  f.type = 'file';
  f.onchange = () => {
    const file = f.files[0];
    if (!file) return;
    const r = new FileReader();
    r.onload = () => {
      const inp = document.getElementById('input');
      inp.value = (inp.value + ' [附件 ' + file.name + ']').trim();
      autoGrow(inp);
    };
    r.readAsText(file);
  };
  f.click();
}

function toggleVoice() {
  if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
    UI.toast('当前浏览器不支持语音输入'); return;
  }
  const micBtn = document.getElementById('micBtn');
  if (recognizing) {
    recognition && recognition.stop();
    recognizing = false; micBtn.textContent = '🎤'; return;
  }
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  recognition = new SR();
  recognition.lang = 'zh-CN'; recognition.interimResults = false;
  recognition.onresult = e => {
    const t = e.results[0][0].transcript;
    const inp = document.getElementById('input');
    inp.value = (inp.value + ' ' + t).trim();
    autoGrow(inp);
  };
  recognition.onend = () => { recognizing = false; micBtn.textContent = '🎤'; };
  recognition.onerror = () => { recognizing = false; micBtn.textContent = '🎤'; };
  recognition.start(); recognizing = true; micBtn.textContent = '⏺';
}

function speak(text) {
  if ('speechSynthesis' in window) {
    window.speechSynthesis.cancel();
    const u = new SpeechSynthesisUtterance(text);
    u.lang = 'zh-CN';
    window.speechSynthesis.speak(u);
  }
}

function speakLast() {
  const ms = document.querySelectorAll('.msg.assistant .body');
  if (ms.length) speak(ms[ms.length - 1].textContent);
}

Object.assign(window, {
  switchProject, loadState, logout, delSession, addTask, searchLaw,
  analyzeEvidence, auditEvidence, generateDoc, exportState, createCase,
  addLaw, pickImage, pickFile, toggleVoice, speakLast, send, doLogin,
  newChat, useStarter, stopStreaming,
});

window.addEventListener('DOMContentLoaded', () => {
  // preserve the empty-state markup so newChat can restore it
  const tpl = document.createElement('template');
  tpl.id = 'empty-state-template';
  tpl.innerHTML = document.getElementById('empty-state').outerHTML;
  document.body.appendChild(tpl);

  loadState(); loadProjects(); loadDocTypes(); loadSessions();
});
