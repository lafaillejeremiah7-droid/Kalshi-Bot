const ROOM_ORDER = ["market", "research", "analysis", "risk", "decision", "performance"];
const ROOM_META = {
  market: { name: "Market Intelligence", color: "blue" },
  research: { name: "Strategy Laboratory", color: "purple" },
  analysis: { name: "Market Analysis", color: "green" },
  risk: { name: "Macro & Risk", color: "gold" },
  decision: { name: "Decision Room", color: "red" },
  performance: { name: "Performance Room", color: "cyan" },
};

const els = {
  floorplan: document.getElementById("floorplan"),
  systemStatus: document.getElementById("systemStatus"),
  marketStatus: document.getElementById("marketStatus"),
  telegramStatus: document.getElementById("telegramStatus"),
  activeCount: document.getElementById("activeCount"),
  headlineText: document.getElementById("headlineText"),
  decisionWord: document.getElementById("decisionWord"),
  decisionSymbol: document.getElementById("decisionSymbol"),
  decisionReason: document.getElementById("decisionReason"),
  decisionPulse: document.getElementById("decisionPulse"),
  entryValue: document.getElementById("entryValue"),
  slValue: document.getElementById("slValue"),
  tpValue: document.getElementById("tpValue"),
  strategyValue: document.getElementById("strategyValue"),
  confidenceBar: document.getElementById("confidenceBar"),
  confidenceValue: document.getElementById("confidenceValue"),
  activityLog: document.getElementById("activityLog"),
  eventCount: document.getElementById("eventCount"),
  signalRows: document.getElementById("signalRows"),
  metricSignals: document.getElementById("metricSignals"),
  metricResolved: document.getElementById("metricResolved"),
  metricWins: document.getElementById("metricWins"),
  metricLosses: document.getElementById("metricLosses"),
  metricWinRate: document.getElementById("metricWinRate"),
  metricBrier: document.getElementById("metricBrier"),
  handoffLayer: document.getElementById("handoffLayer"),
};

let built = false;
let rosterSignature = "";
let firstState = true;
const seenHandoffs = new Set();
let selectedRoom = "all";
let pollTimer = null;

function safeText(value, fallback = "—") {
  if (value === null || value === undefined || value === "") return fallback;
  return String(value);
}

function money(value) {
  const n = Number(value);
  if (!Number.isFinite(n)) return "—";
  return n.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

function percent(value) {
  const n = Number(value);
  if (!Number.isFinite(n)) return "—";
  return `${(n * 100).toFixed(1)}%`;
}

function shortTime(value) {
  if (!value) return "—";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "—";
  return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

function employeeRosterSignature(state) {
  const employees = Array.isArray(state.employees) ? state.employees : [];
  return employees.map(employee => `${employee.id}:${employee.room}`).join("|");
}

function createFigure() {
  const scene = document.createElement("div");
  scene.className = "office-scene";
  scene.innerHTML = `
    <div class="monitor"></div><div class="monitor-stand"></div>
    <div class="mini-person">
      <div class="person-head"></div><div class="person-hair"></div>
      <div class="person-body"></div>
      <div class="arm left"></div><div class="arm right"></div>
      <div class="leg left"></div><div class="leg right"></div>
    </div>
    <div class="paper-stack"></div><div class="desk"></div>`;
  return scene;
}

function buildFloor(state) {
  els.floorplan.replaceChildren();
  const employees = Array.isArray(state.employees) ? state.employees : [];
  const rooms = Array.isArray(state.rooms) ? state.rooms : [];
  const roomNames = new Map(rooms.map(r => [r.id, r.name]));

  for (const roomId of ROOM_ORDER) {
    const room = document.createElement("section");
    room.className = "room";
    room.dataset.room = roomId;

    const roomEmployees = employees.filter(e => e.room === roomId);
    const header = document.createElement("div");
    header.className = "room-header";
    const title = document.createElement("h2");
    title.textContent = roomNames.get(roomId) || ROOM_META[roomId]?.name || roomId;
    const count = document.createElement("span");
    count.className = "room-count";
    count.textContent = `${roomEmployees.length} BOTS`;
    header.append(title, count);

    const workers = document.createElement("div");
    workers.className = "workers";
    for (const employee of roomEmployees) {
      const worker = document.createElement("article");
      worker.className = "worker";
      worker.dataset.worker = employee.id;
      worker.title = employee.name;

      const stateDot = document.createElement("span");
      stateDot.className = "worker-state";
      const veto = document.createElement("span");
      veto.className = "worker-veto";
      veto.textContent = "!";
      const name = document.createElement("div");
      name.className = "worker-name";
      name.textContent = employee.name;
      const task = document.createElement("div");
      task.className = "worker-task";
      task.dataset.task = "";
      task.textContent = employee.task || "Waiting";

      worker.append(stateDot, veto, createFigure(), name, task);
      workers.append(worker);
    }
    room.append(header, workers);
    els.floorplan.append(room);
  }
  rosterSignature = employeeRosterSignature(state);
  built = true;
}

function stateClass(state) {
  const normalized = String(state || "idle").toLowerCase();
  if (["working", "handoff", "receiving", "blocked", "done"].includes(normalized)) return normalized;
  return "idle";
}

function updateWorkers(state) {
  const employees = Array.isArray(state.employees) ? state.employees : [];
  for (const employee of employees) {
    const worker = document.querySelector(`[data-worker="${CSS.escape(employee.id)}"]`);
    if (!worker) continue;
    const normalized = stateClass(employee.state);
    // "Online/active" and "currently doing a task" are intentionally separate.
    // The live dashboard is served only while the company session is running,
    // so every canonical roster member is online even when its current task is idle.
    worker.className = `worker is-${normalized} is-online`;
    worker.dataset.state = normalized;
    worker.dataset.online = "true";
    worker.title = ["ONLINE", employee.name, employee.task, employee.detail].filter(Boolean).join(" — ");
    const task = worker.querySelector("[data-task]");
    if (task) {
      const direction = employee.direction && employee.direction !== "HOLD" ? ` · ${employee.direction}` : "";
      task.textContent = `${employee.task || "Waiting"}${direction}`;
    }
  }
  // Header reports roster availability, not the subset whose task-state is non-idle.
  // This prevents a healthy 28-bot company from being mislabeled 10/28 active.
  els.activeCount.textContent = String(employees.length);
}

function updateSystem(state) {
  const system = state.system || {};
  const status = String(system.status || "WAITING").toUpperCase();
  const tone = status === "ERROR" ? "error" : status === "HOLD" ? "hold" : "online";
  els.systemStatus.dataset.tone = tone;
  els.systemStatus.querySelector("b").textContent = status;
  els.headlineText.textContent = system.headline || "Waiting for runtime";
  els.marketStatus.textContent = system.market_open === true ? "OPEN" : system.market_open === false ? "CLOSED" : "UNKNOWN";
  els.telegramStatus.textContent = safeText(system.telegram, "UNKNOWN").toUpperCase();
}

function updateBoss(state) {
  const boss = state.boss || {};
  const direction = String(boss.decision || "HOLD").toUpperCase();
  els.decisionWord.textContent = direction;
  els.decisionWord.className = `decision-word ${direction === "BUY" ? "buy" : direction === "SELL" ? "sell" : "hold"}`;
  els.decisionSymbol.textContent = boss.symbol || "XAU/USD";
  els.decisionReason.textContent = boss.reason || "Waiting for synchronized bot output";
  els.decisionPulse.textContent = direction === "BUY" || direction === "SELL" ? "AUTHORIZED" : "STANDBY";
  els.entryValue.textContent = money(boss.entry);
  els.slValue.textContent = money(boss.stop_loss);
  els.tpValue.textContent = money(boss.take_profit);
  els.strategyValue.textContent = boss.strategy || "No decision yet";
  els.confidenceValue.textContent = percent(boss.confidence);
  const conf = Number(boss.confidence);
  els.confidenceBar.style.width = Number.isFinite(conf) ? `${Math.max(0, Math.min(100, conf * 100))}%` : "0%";
}

function updateActivity(state) {
  const activity = Array.isArray(state.activity) ? state.activity.slice(-14).reverse() : [];
  const handoffs = Array.isArray(state.handoffs) ? state.handoffs : [];
  els.eventCount.textContent = `${handoffs.length} EVENTS`;
  els.activityLog.replaceChildren();
  if (!activity.length) {
    const empty = document.createElement("div");
    empty.className = "empty-log";
    empty.textContent = "Waiting for the company to move its first file.";
    els.activityLog.append(empty);
    return;
  }
  for (const item of activity) {
    const row = document.createElement("div");
    row.className = "log-row";
    const time = document.createElement("div");
    time.className = "log-time";
    time.textContent = shortTime(item.at);
    const text = document.createElement("div");
    text.className = "log-text";
    text.textContent = item.text || "Activity";
    if (item.detail) {
      const small = document.createElement("small");
      small.textContent = item.detail;
      text.append(small);
    }
    row.append(time, text);
    els.activityLog.append(row);
  }
}

function updatePerformance(state) {
  const performance = state.performance || {};
  els.metricSignals.textContent = safeText(performance.signals, "0");
  els.metricResolved.textContent = safeText(performance.resolved, "0");
  els.metricWins.textContent = safeText(performance.wins, "0");
  els.metricLosses.textContent = safeText(performance.losses, "0");
  els.metricWinRate.textContent = percent(performance.win_rate);
  els.metricBrier.textContent = Number.isFinite(Number(performance.brier)) ? Number(performance.brier).toFixed(4) : "—";

  const rows = Array.isArray(performance.recent_signals) ? performance.recent_signals : [];
  els.signalRows.replaceChildren();
  if (!rows.length) {
    const tr = document.createElement("tr");
    const td = document.createElement("td");
    td.colSpan = 7;
    td.className = "empty-row";
    td.textContent = "No recorded signals yet.";
    tr.append(td);
    els.signalRows.append(tr);
    return;
  }
  for (const signal of rows) {
    const tr = document.createElement("tr");
    const values = [
      shortTime(signal.observed_at),
      signal.direction || "—",
      signal.strategy || "—",
      percent(signal.confidence),
      money(signal.entry),
      signal.status || "OPEN",
      signal.delivery_state || "UNKNOWN",
    ];
    values.forEach((value, index) => {
      const td = document.createElement("td");
      td.textContent = value;
      if (index === 1) td.className = String(signal.direction).toUpperCase() === "BUY" ? "direction-buy" : String(signal.direction).toUpperCase() === "SELL" ? "direction-sell" : "";
      if (index === 5) td.className = String(signal.status).toUpperCase() === "WIN" ? "result-win" : String(signal.status).toUpperCase() === "LOSS" ? "result-loss" : "";
      if (index === 6 && String(signal.delivery_state).toUpperCase() === "SENT") td.className = "delivery-sent";
      tr.append(td);
    });
    els.signalRows.append(tr);
  }
}

function animatePaper(event) {
  const source = document.querySelector(`[data-worker="${CSS.escape(event.from || "")}"]`);
  const target = document.querySelector(`[data-worker="${CSS.escape(event.to || "")}"]`);
  if (!source || !target) return;
  const a = source.getBoundingClientRect();
  const b = target.getBoundingClientRect();
  const startX = a.left + a.width * .56;
  const startY = a.top + a.height * .46;
  const endX = b.left + b.width * .44;
  const endY = b.top + b.height * .48;

  const paper = document.createElement("div");
  paper.className = "flying-paper";
  paper.style.left = `${startX}px`;
  paper.style.top = `${startY}px`;
  paper.style.transform = "translate3d(0,0,0) rotate(-7deg) scale(.92)";

  const label = document.createElement("div");
  label.className = "flying-label";
  label.textContent = event.document || "Report";
  label.style.left = `${startX}px`;
  label.style.top = `${startY - 22}px`;

  els.handoffLayer.append(paper, label);
  source.classList.add("is-handoff");
  target.classList.add("is-receiving");

  requestAnimationFrame(() => {
    paper.style.transform = `translate3d(${endX - startX}px, ${endY - startY}px, 0) rotate(8deg) scale(1.05)`;
    paper.style.opacity = "0";
    label.style.transform = `translate3d(${(endX - startX) * .55}px, ${(endY - startY) * .55 - 8}px, 0)`;
  });

  window.setTimeout(() => {
    paper.remove();
    label.remove();
  }, 1350);
}

function processHandoffs(state) {
  const handoffs = Array.isArray(state.handoffs) ? state.handoffs : [];
  if (firstState) {
    handoffs.forEach(h => seenHandoffs.add(h.id));
    firstState = false;
    return;
  }
  for (const event of handoffs) {
    if (!event.id || seenHandoffs.has(event.id)) continue;
    seenHandoffs.add(event.id);
    animatePaper(event);
  }
  if (seenHandoffs.size > 500) {
    const keep = new Set(handoffs.map(h => h.id).filter(Boolean));
    seenHandoffs.clear();
    keep.forEach(id => seenHandoffs.add(id));
  }
}

function applyRoomFilter() {
  document.querySelectorAll(".room").forEach(room => {
    if (selectedRoom === "all") {
      room.classList.remove("dimmed", "focused");
    } else if (room.dataset.room === selectedRoom) {
      room.classList.remove("dimmed");
      room.classList.add("focused");
    } else {
      room.classList.remove("focused");
      room.classList.add("dimmed");
    }
  });
}

function render(state) {
  const nextRosterSignature = employeeRosterSignature(state);
  if (!built || nextRosterSignature !== rosterSignature) buildFloor(state);
  updateWorkers(state);
  updateSystem(state);
  updateBoss(state);
  updateActivity(state);
  updatePerformance(state);
  processHandoffs(state);
  applyRoomFilter();
}

async function poll() {
  try {
    const response = await fetch(`/api/state?t=${Date.now()}`, { cache: "no-store" });
    if (!response.ok) throw new Error(`state request ${response.status}`);
    const state = await response.json();
    render(state);
  } catch (error) {
    els.systemStatus.dataset.tone = "error";
    els.systemStatus.querySelector("b").textContent = "OFFLINE";
    els.headlineText.textContent = "Dashboard server cannot reach runtime state";
    console.warn("dashboard poll failed", error);
  } finally {
    pollTimer = window.setTimeout(poll, 1200);
  }
}

document.querySelectorAll(".nav-tab").forEach(button => {
  button.addEventListener("click", () => {
    document.querySelectorAll(".nav-tab").forEach(btn => btn.classList.remove("active"));
    button.classList.add("active");
    selectedRoom = button.dataset.roomFilter || "all";
    applyRoomFilter();
  });
});

window.addEventListener("beforeunload", () => {
  if (pollTimer) window.clearTimeout(pollTimer);
});

poll();
