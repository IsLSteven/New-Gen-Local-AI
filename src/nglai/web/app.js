(() => {
  const gpuVal = document.getElementById("gpuVal");
  const ramVal = document.getElementById("ramVal");
  const kvVal = document.getElementById("kvVal");
  const speedVal = document.getElementById("speedVal");
  const modelSelect = document.getElementById("modelSelect");
  const pullModelBtn = document.getElementById("pullModelBtn");
  const downloadBanner = document.getElementById("downloadBanner");
  const dlTitle = document.getElementById("dlTitle");
  const dlStats = document.getElementById("dlStats");
  const dlBar = document.getElementById("dlBar");
  const engineBadge = document.getElementById("engineBadge");
  const engineToggleBtn = document.getElementById("engineToggleBtn");
  const fileTree = document.getElementById("fileTree");
  const repoMapView = document.getElementById("repoMapView");
  const splitNotes = document.getElementById("splitNotes");
  const otRegexBox = document.getElementById("otRegexBox");
  const hotSsdInput = document.getElementById("hotSsdInput");
  const coldHddInput = document.getElementById("coldHddInput");
  const vaultFeedback = document.getElementById("vaultFeedback");
  const codeViewer = document.getElementById("codeViewer");
  const currentFileTitle = document.getElementById("currentFileTitle");
  const viewModeBadge = document.getElementById("viewModeBadge");
  const chatMessages = document.getElementById("chatMessages");
  const promptInput = document.getElementById("promptInput");
  const sendBtn = document.getElementById("sendBtn");
  const thinkingToggle = document.getElementById("thinkingToggle");

  let engineRunning = false;
  let ws = null;

  const gatewayLabel = document.getElementById("gatewayUrlLabel");
  if (gatewayLabel) {
    gatewayLabel.textContent = `${window.location.origin}/v1`;
  }

  // Sidebar Tabs
  document.querySelectorAll(".tab").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".tab").forEach((b) => b.classList.remove("active"));
      document.querySelectorAll(".tab-content").forEach((c) => c.classList.remove("active"));
      btn.classList.add("active");
      const target = document.getElementById(`tab-${btn.dataset.tab}`);
      if (target) target.classList.add("active");
      if (btn.dataset.tab === "repomap") loadRepoMap();
    });
  });

  function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text || "";
    return div.innerHTML;
  }

  function renderDiffInViewer(title, diffText) {
    currentFileTitle.textContent = `⚡ Diff Aplicado: ${title}`;
    viewModeBadge.textContent = "Unified Diff";
    const lines = (diffText || "").split("\n").map((line) => {
      let cls = "diff-line";
      if (line.startsWith("+") && !line.startsWith("+++")) cls += " diff-add";
      else if (line.startsWith("-") && !line.startsWith("---")) cls += " diff-del";
      else if (line.startsWith("@@")) cls += " diff-hunk";
      return `<div class="${cls}">${escapeHtml(line)}</div>`;
    });
    codeViewer.innerHTML = lines.join("");
  }

  function renderCodeInViewer(path, content) {
    currentFileTitle.textContent = `📄 ${path}`;
    viewModeBadge.textContent = "Código";
    const numbered = (content || "")
      .split("\n")
      .map((l, idx) => `${String(idx + 1).padStart(4, " ")} │ ${escapeHtml(l)}`)
      .join("\n");
    codeViewer.innerHTML = `<pre>${numbered}</pre>`;
  }

  async function loadModels() {
    const res = await fetch("/api/models");
    const data = await res.json();
    modelSelect.innerHTML = "";
    data.models.forEach((m) => {
      const opt = document.createElement("option");
      opt.value = m.id;
      const badge = m.in_hot_ssd ? "🔥" : m.in_cold_hdd ? "❄️" : "⬇";
      opt.textContent = `${badge} ${m.name} (${m.size_gb} GB)`;
      if (m.active) opt.selected = true;
      modelSelect.appendChild(opt);
    });
  }

  async function loadTelemetry() {
    try {
      const res = await fetch("/api/telemetry");
      const data = await res.json();
      const hw = data.hardware;
      const plan = data.split_plan;
      const eng = data.engine;

      const gpu = hw.gpus && hw.gpus.length > 0 ? hw.gpus[0] : null;
      gpuVal.textContent = gpu
        ? `~${plan.estimated_vram_gb} / ${(gpu.vram_total_mb / 1024).toFixed(1)} GB`
        : "CPU Mode";
      ramVal.textContent = `~${plan.estimated_ram_gb} / ${hw.ram_total_gb} GB`;
      kvVal.textContent = `${plan.kv_cache_type} (${Math.round(plan.recommended_ctx / 1024)}K)`;
      speedVal.textContent = plan.estimated_tok_per_sec;

      engineRunning = eng.running;
      engineBadge.textContent = eng.running ? "Engine Online" : "Standby";
      engineBadge.className = `status-badge ${eng.running ? "online" : "offline"}`;
      engineToggleBtn.textContent = eng.running ? "Parar Motor" : "Iniciar Motor";

      splitNotes.innerHTML = (plan.notes_pt || []).map((n) => `<p>• ${escapeHtml(n)}</p>`).join("");
      otRegexBox.textContent = plan.override_tensor_regex || "(100% GPU — sem override necessário)";

      if (!hotSsdInput.value) hotSsdInput.value = data.config.hot_ssd_dir;
      if (!coldHddInput.value) coldHddInput.value = data.config.cold_hdd_dir;
      thinkingToggle.checked = Boolean(data.config.enable_thinking);
    } catch (err) {
      console.error("Telemetry error:", err);
    }
  }

  async function loadFileTree() {
    const res = await fetch("/api/workspace/tree");
    const data = await res.json();
    document.getElementById("workspaceRootLabel").textContent = data.root.split(/[\\/]/).pop() || data.root;
    fileTree.innerHTML = "";
    data.files.forEach((f) => {
      const li = document.createElement("li");
      li.textContent = f.path;
      li.addEventListener("click", async () => {
        document.querySelectorAll(".file-tree li").forEach((el) => el.classList.remove("active"));
        li.classList.add("active");
        const r = await fetch(`/api/workspace/file?path=${encodeURIComponent(f.path)}`);
        if (r.ok) {
          const payload = await r.json();
          renderCodeInViewer(payload.path, payload.content);
        }
      });
      fileTree.appendChild(li);
    });
  }

  async function loadRepoMap() {
    const res = await fetch("/api/workspace/repomap");
    const data = await res.json();
    repoMapView.textContent = data.repomap;
  }

  modelSelect.addEventListener("change", async () => {
    await fetch("/api/models/select", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model_id: modelSelect.value }),
    });
    await loadTelemetry();
  });

  let dlPollTimer = null;

  async function pollDownloadStatus() {
    try {
      const res = await fetch("/api/vault/pull-status");
      const st = await res.json();
      if (st.active) {
        downloadBanner.classList.remove("hidden");
        pullModelBtn.disabled = true;
        const doneGb = (st.downloaded_bytes / (1024 ** 3)).toFixed(2);
        const totGb = st.total_bytes > 0 ? (st.total_bytes / (1024 ** 3)).toFixed(2) : "?";
        const pct = st.total_bytes > 0 ? Math.min(100, (st.downloaded_bytes / st.total_bytes) * 100) : 5;
        dlTitle.textContent = `⬇ Baixando ${st.model_id}...`;
        dlStats.textContent = `${doneGb} / ${totGb} GB (${pct.toFixed(1)}%)`;
        dlBar.style.width = `${pct}%`;
      } else {
        if (dlPollTimer) {
          clearInterval(dlPollTimer);
          dlPollTimer = null;
        }
        pullModelBtn.disabled = false;
        if (st.error) {
          dlTitle.textContent = `❌ Erro no download: ${st.error}`;
        } else if (st.done_path) {
          dlTitle.textContent = `✅ Modelo pronto em ${st.done_path}! Clique em "Iniciar Motor".`;
          dlBar.style.width = "100%";
          await loadModels();
          setTimeout(() => downloadBanner.classList.add("hidden"), 6000);
        } else {
          downloadBanner.classList.add("hidden");
        }
      }
    } catch (e) {
      console.error(e);
    }
  }

  pullModelBtn.addEventListener("click", async () => {
    downloadBanner.classList.remove("hidden");
    dlTitle.textContent = `⬇ Iniciando download de ${modelSelect.value}...`;
    dlStats.textContent = "Conectando ao HuggingFace...";
    dlBar.style.width = "2%";
    await fetch("/api/vault/pull", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model_id: modelSelect.value }),
    });
    if (!dlPollTimer) {
      dlPollTimer = setInterval(pollDownloadStatus, 1200);
    }
  });

  engineToggleBtn.addEventListener("click", async () => {
    engineToggleBtn.disabled = true;
    try {
      const endpoint = engineRunning ? "/api/engine/stop" : "/api/engine/start";
      const res = await fetch(endpoint, { method: "POST" });
      if (!res.ok) {
        const err = await res.json();
        appendChatCard("err", `⚠️ ${err.detail || "Falha ao acionar motor."}`);
      }
      await loadTelemetry();
    } finally {
      engineToggleBtn.disabled = false;
    }
  });

  document.getElementById("saveVaultBtn").addEventListener("click", async () => {
    const res = await fetch("/api/vault/paths", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        hot_ssd_dir: hotSsdInput.value,
        cold_hdd_dir: coldHddInput.value,
      }),
    });
    if (res.ok) {
      vaultFeedback.textContent = "✓ Pastas Hot SSD & Cold HDD atualizadas!";
      loadModels();
    }
  });

  document.getElementById("evictSsdBtn").addEventListener("click", async () => {
    const res = await fetch("/api/vault/evict", { method: "POST" });
    const data = await res.json();
    vaultFeedback.textContent = data.evicted_files.length
      ? `✓ Movidos p/ HDD: ${data.evicted_files.join(", ")}`
      : "✓ SSD já enxuto (apenas modelo ativo).";
    loadModels();
  });

  document.getElementById("refreshTreeBtn").addEventListener("click", loadFileTree);
  document.getElementById("refreshMapBtn").addEventListener("click", loadRepoMap);

  function scrollChat() {
    chatMessages.scrollTop = chatMessages.scrollHeight;
  }

  function appendUserMsg(text) {
    const div = document.createElement("div");
    div.className = "msg user";
    div.innerHTML = `<div class="msg-body">${escapeHtml(text)}</div>`;
    chatMessages.appendChild(div);
    scrollChat();
  }

  function appendThinkingBlock(text) {
    const details = document.createElement("details");
    details.className = "think-card";
    details.innerHTML = `<summary>🧠 Raciocínio Arquitetural (&lt;think&gt;)</summary><pre>${escapeHtml(text)}</pre>`;
    chatMessages.appendChild(details);
    scrollChat();
  }

  function appendChatCard(cls, text) {
    const div = document.createElement("div");
    div.className = `tool-card ${cls}`;
    div.textContent = text;
    chatMessages.appendChild(div);
    scrollChat();
  }

  function appendAssistantMsg(text) {
    const div = document.createElement("div");
    div.className = "msg assistant";
    div.innerHTML = `<div class="msg-header">⚡ New-Gen-Local-AI</div><div class="msg-body">${escapeHtml(text)}</div>`;
    chatMessages.appendChild(div);
    scrollChat();
  }

  function connectWebSocket() {
    const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
    ws = new WebSocket(`${proto}//${window.location.host}/ws/agent`);

    ws.onmessage = (event) => {
      const ev = JSON.parse(event.data);
      if (ev.kind === "reset_ok") {
        chatMessages.innerHTML = "";
        appendAssistantMsg("Sessão limpa! Pronto para uma nova tarefa.");
      } else if (ev.kind === "thinking" && ev.text) {
        appendThinkingBlock(ev.text);
      } else if (ev.kind === "tool_call") {
        appendChatCard("", `🔧 ${ev.tool_name}(${JSON.stringify(ev.tool_args || {}).slice(0, 140)})`);
      } else if (ev.kind === "tool_result") {
        const ok = ev.metadata && ev.metadata.ok;
        const firstLine = (ev.text || "").split("\n")[0];
        appendChatCard(ok ? "ok" : "err", `${ok ? "✅" : "❌"} [${ev.tool_name}] ${firstLine}`);
        if (ev.diff) {
          renderDiffInViewer(ev.tool_name, ev.diff);
          loadFileTree();
        }
      } else if (ev.kind === "compacted") {
        appendChatCard("ok", `🗜️ ${ev.text}`);
      } else if (ev.kind === "content" && ev.text) {
        appendAssistantMsg(ev.text);
      } else if (ev.kind === "error") {
        appendChatCard("err", ev.text);
        sendBtn.disabled = false;
      } else if (ev.kind === "done") {
        sendBtn.disabled = false;
      }
    };

    ws.onclose = () => {
      setTimeout(connectWebSocket, 2500);
    };
  }

  function sendPrompt() {
    const text = promptInput.value.trim();
    if (!text || !ws || ws.readyState !== WebSocket.OPEN) return;
    appendUserMsg(text);
    promptInput.value = "";
    sendBtn.disabled = true;
    ws.send(
      JSON.stringify({
        action: "chat",
        prompt: text,
        enable_thinking: thinkingToggle.checked,
      })
    );
  }

  sendBtn.addEventListener("click", sendPrompt);
  promptInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendPrompt();
    }
  });

  document.getElementById("resetChatBtn").addEventListener("click", () => {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ action: "reset" }));
    }
  });

  loadModels();
  loadTelemetry();
  loadFileTree();
  connectWebSocket();
  pollDownloadStatus().then(() => {
    fetch("/api/vault/pull-status")
      .then((r) => r.json())
      .then((st) => {
        if (st.active && !dlPollTimer) {
          dlPollTimer = setInterval(pollDownloadStatus, 1200);
        }
      });
  });
  setInterval(loadTelemetry, 8000);
})();
