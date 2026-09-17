const API = '';
let metricsChart = null;
let monitoringChart = null;
let feedPollInterval = null;
let startTime = Date.now();

// ── Boot ─────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  loadHealth();
  loadModels();
  loadMonitoring();
  startFeedPolling();

  document.getElementById('predict-form').addEventListener('submit', onPredict);
  document.getElementById('refresh-monitoring').addEventListener('click', loadMonitoring);
  document.getElementById('feed-auto').addEventListener('change', (e) => {
    e.target.checked ? startFeedPolling() : stopFeedPolling();
  });

  // Refresh health + monitoring every 30 s
  setInterval(() => { loadHealth(); loadMonitoring(); }, 30_000);
  // Update uptime counter every second
  setInterval(updateUptime, 1000);
});

function updateUptime() {
  const s = Math.floor((Date.now() - startTime) / 1000);
  const h = Math.floor(s / 3600).toString().padStart(2, '0');
  const m = Math.floor((s % 3600) / 60).toString().padStart(2, '0');
  const sec = (s % 60).toString().padStart(2, '0');
  document.getElementById('uptime-label').textContent = `Session ${h}:${m}:${sec}`;
}

// ── Health ───────────────────────────────────────────────────────────────────
async function loadHealth() {
  try {
    const data = await get('/health');
    const badge = document.getElementById('health-badge');
    badge.textContent = data.status === 'ok' ? 'System OK' : 'Degraded';
    badge.className = `badge ${data.status === 'ok' ? 'badge-ok' : 'badge-warn'}`;

    document.getElementById('active-model').textContent = data.active_model ?? '—';
    document.getElementById('total-preds').textContent = data.total_predictions.toLocaleString();
  } catch {
    const badge = document.getElementById('health-badge');
    badge.textContent = 'API Offline';
    badge.className = 'badge badge-error';
  }
}

// ── Models ───────────────────────────────────────────────────────────────────
async function loadModels() {
  const models = await get('/models').catch(() => []);
  const tbody = document.getElementById('model-tbody');

  if (!models.length) {
    tbody.innerHTML = '<tr><td colspan="8" class="loading-cell">No models registered — run <code>make train</code>.</td></tr>';
    return;
  }

  tbody.innerHTML = '';
  const chartLabels = [], aucData = [], f1Data = [];

  for (const m of models) {
    const tr = document.createElement('tr');
    if (m.is_active) tr.className = 'active-row';

    const cvStr = (m.cv_roc_auc_mean != null && m.cv_roc_auc_std != null)
      ? `${m.cv_roc_auc_mean.toFixed(3)} ±${m.cv_roc_auc_std.toFixed(3)}`
      : '—';

    const statusBadge = m.is_active
      ? '<span class="badge badge-ok">Active</span>'
      : '<span class="badge badge-loading">Standby</span>';

    const activateBtn = m.is_active ? '' :
      `<button class="btn-sm" onclick="activateModel('${m.version}')">Promote</button>`;

    tr.innerHTML = `
      <td class="mono"><strong>${m.version}</strong></td>
      <td>${m.algorithm.replace('Classifier', '')}</td>
      <td>${m.roc_auc != null ? m.roc_auc.toFixed(4) : '—'}</td>
      <td class="mono small">${cvStr}</td>
      <td>${m.optimal_threshold != null ? m.optimal_threshold.toFixed(3) : '0.500'}</td>
      <td>${m.f1_score != null ? m.f1_score.toFixed(4) : '—'}</td>
      <td>${statusBadge}</td>
      <td>${activateBtn}</td>
    `;
    tbody.appendChild(tr);
    chartLabels.push(m.version);
    aucData.push(m.roc_auc ?? 0);
    f1Data.push(m.f1_score ?? 0);

    if (m.is_active) {
      document.getElementById('roc-auc').textContent = m.roc_auc?.toFixed(4) ?? '—';
      document.getElementById('threshold').textContent = m.optimal_threshold?.toFixed(3) ?? '0.500';
      document.getElementById('avg-latency').textContent =
        m.avg_latency_ms ? m.avg_latency_ms.toFixed(1) + ' ms' : '—';
      if (m.cv_roc_auc_mean != null)
        document.getElementById('cv-auc').textContent =
          `CV: ${m.cv_roc_auc_mean.toFixed(3)} ±${m.cv_roc_auc_std?.toFixed(3)}`;
    }
  }

  drawMetricsChart(chartLabels, aucData, f1Data);
}

function drawMetricsChart(labels, aucData, f1Data) {
  const ctx = document.getElementById('metrics-chart').getContext('2d');
  if (metricsChart) metricsChart.destroy();
  metricsChart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [
        { label: 'ROC-AUC', data: aucData, backgroundColor: 'rgba(99,102,241,.75)', borderRadius: 6 },
        { label: 'F1 Score', data: f1Data, backgroundColor: 'rgba(34,197,94,.75)', borderRadius: 6 },
      ],
    },
    options: {
      responsive: true,
      plugins: { legend: { labels: { color: '#94a3b8', font: { size: 12 } } } },
      scales: {
        y: { min: 0, max: 1, grid: { color: '#2e3347' }, ticks: { color: '#94a3b8' } },
        x: { grid: { display: false }, ticks: { color: '#94a3b8' } },
      },
    },
  });
}

async function activateModel(version) {
  try {
    await fetch(`${API}/models/${version}/activate`, { method: 'PATCH' });
    await Promise.all([loadHealth(), loadModels()]);
  } catch (e) {
    alert('Failed to activate model: ' + e.message);
  }
}

// ── Predict ───────────────────────────────────────────────────────────────────
async function onPredict(e) {
  e.preventDefault();
  const btn = document.getElementById('submit-btn');
  btn.disabled = true;
  btn.textContent = 'Analyzing…';

  const version = document.getElementById('api-version').value;
  const payload = {
    amount: parseFloat(document.getElementById('amount').value),
    hour_of_day: parseInt(document.getElementById('hour_of_day').value),
    day_of_week: parseInt(document.getElementById('day_of_week').value),
    merchant_category: parseInt(document.getElementById('merchant_category').value),
    distance_from_home: parseFloat(document.getElementById('distance_from_home').value),
    is_weekend: document.getElementById('is_weekend').checked,
    velocity_1h: parseInt(document.getElementById('velocity_1h').value),
    velocity_24h: parseInt(document.getElementById('velocity_24h').value),
    amount_z_score: parseFloat(document.getElementById('amount_z_score').value),
  };

  try {
    const result = await post(`/${version}/predict`, payload);
    showResult(result, version);
    loadHealth();
    loadFeed(); // immediate feed refresh after prediction
  } catch (err) {
    alert('Prediction failed: ' + err.message);
  } finally {
    btn.disabled = false;
    btn.textContent = 'Analyze Transaction';
  }
}

function showResult(r, apiVersion) {
  const box = document.getElementById('result-box');
  box.classList.remove('hidden');

  const verdict = document.getElementById('result-verdict');
  verdict.textContent = r.predicted_fraud ? '⚠ FRAUD DETECTED' : '✓ Legitimate Transaction';
  verdict.className = `verdict ${r.predicted_fraud ? 'fraud' : 'legit'}`;

  document.getElementById('res-prob').textContent = (r.fraud_probability * 100).toFixed(2) + '%';
  document.getElementById('res-version').textContent = r.model_version;
  document.getElementById('res-latency').textContent = r.latency_ms?.toFixed(2) + ' ms';
  document.getElementById('res-reqid').textContent = r.request_id ?? '—';

  const riskEl = document.getElementById('res-risk');
  if (apiVersion === 'v2' && r.risk_band) {
    riskEl.textContent = r.risk_band.toUpperCase();
    riskEl.className = `badge badge-${r.risk_band}`;
  } else {
    riskEl.textContent = '—';
    riskEl.className = 'badge';
  }

  const factorsSection = document.getElementById('risk-factors-section');
  const factorsList = document.getElementById('risk-factors-list');
  if (apiVersion === 'v2' && r.top_risk_factors?.length) {
    factorsSection.classList.remove('hidden');
    const maxImp = r.top_risk_factors[0].importance;
    factorsList.innerHTML = r.top_risk_factors.map(f => `
      <div class="factor-bar-wrap">
        <span class="factor-name">${formatFeatureName(f.feature)}</span>
        <div class="factor-bar">
          <div class="factor-fill" style="width:${(f.importance / maxImp * 100).toFixed(1)}%"></div>
        </div>
        <span class="factor-pct">${(f.importance * 100).toFixed(1)}%</span>
      </div>
    `).join('');
  } else {
    factorsSection.classList.add('hidden');
  }
}

// ── Live Feed ─────────────────────────────────────────────────────────────────
let lastFeedCount = 0;
let feedTimestamps = [];

function startFeedPolling() {
  if (feedPollInterval) return;
  loadFeed();
  feedPollInterval = setInterval(loadFeed, 4000);
}

function stopFeedPolling() {
  clearInterval(feedPollInterval);
  feedPollInterval = null;
}

async function loadFeed() {
  const rows = await get('/predictions/recent?limit=20').catch(() => null);
  if (!rows) return;

  // Throughput: count new rows per minute using timestamps
  feedTimestamps = feedTimestamps.filter(t => Date.now() - t < 60_000);
  const newRows = rows.length - lastFeedCount;
  if (newRows > 0 && lastFeedCount > 0) {
    for (let i = 0; i < newRows; i++) feedTimestamps.push(Date.now());
  }
  lastFeedCount = rows.length;

  const tpm = feedTimestamps.length;
  document.getElementById('feed-rate').textContent = tpm > 0 ? `~${tpm} req/min` : '';

  const tbody = document.getElementById('feed-tbody');
  if (!rows.length) {
    tbody.innerHTML = '<tr><td colspan="8" class="loading-cell">No predictions yet — run a prediction above.</td></tr>';
    return;
  }

  tbody.innerHTML = rows.map((r, i) => {
    const ts = r.created_at ? new Date(r.created_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }) : '—';
    const prob = r.fraud_probability;
    const probBar = `<div class="prob-bar-wrap"><div class="prob-bar" style="width:${(prob*100).toFixed(1)}%;background:${probColor(prob)}"></div><span>${(prob*100).toFixed(1)}%</span></div>`;
    const verdict = r.predicted_fraud
      ? '<span class="verdict-pill fraud-pill">FRAUD</span>'
      : '<span class="verdict-pill legit-pill">Legit</span>';
    const risk = `<span class="badge badge-${r.risk_band}">${r.risk_band}</span>`;
    const newClass = i === 0 ? ' row-new' : '';
    const shortId = r.transaction_id ? r.transaction_id.slice(0, 8) + '…' : '—';
    return `<tr class="${newClass}">
      <td class="mono small">${ts}</td>
      <td>$${r.amount?.toFixed(2) ?? '—'}</td>
      <td>${probBar}</td>
      <td>${risk}</td>
      <td>${verdict}</td>
      <td class="mono small">${r.model_version}</td>
      <td class="small">${r.latency_ms?.toFixed(1) ?? '—'} ms</td>
      <td class="mono small text-muted">${shortId}</td>
    </tr>`;
  }).join('');
}

function probColor(p) {
  if (p < 0.1) return '#22c55e';
  if (p < 0.4) return '#f59e0b';
  if (p < 0.7) return '#f97316';
  return '#ef4444';
}

// ── Monitoring ────────────────────────────────────────────────────────────────
async function loadMonitoring() {
  try {
    const snap = await get('/monitoring/snapshot');
    renderSnapshot(snap);
  } catch {}
  try {
    const history = await get('/monitoring/history?limit=40');
    renderMonitoringChart(history);
  } catch {}
}

function renderSnapshot(snap) {
  document.getElementById('mon-fraud-rate').textContent =
    (snap.fraud_rate_observed * 100).toFixed(2) + '%';
  document.getElementById('mon-baseline').textContent =
    `Baseline: ${(snap.fraud_rate_baseline * 100).toFixed(2)}%`;
  document.getElementById('mon-window').textContent = snap.window_size.toLocaleString();
  document.getElementById('mon-accuracy').textContent =
    snap.accuracy != null ? (snap.accuracy * 100).toFixed(1) + '%' : 'Awaiting labels';

  const psiEl = document.getElementById('mon-psi');
  psiEl.textContent = snap.psi_score.toFixed(4);
  psiEl.className = 'monitor-value ' + (
    snap.psi_score > 0.2 ? 'drift-alert' :
    snap.psi_score > 0.1 ? 'drift-warn' : 'drift-ok'
  );

  const driftEl = document.getElementById('drift-status');
  const driftCard = document.getElementById('drift-card');
  driftEl.textContent = snap.drift_detected ? 'DRIFT' : 'Stable';
  driftCard.style.borderColor = snap.drift_detected ? 'rgba(239,68,68,.5)' : '';

  const alertBanner = document.getElementById('alert-banner');
  if (snap.alert_message) {
    alertBanner.textContent = '⚠ ' + snap.alert_message;
    alertBanner.classList.remove('hidden');
  } else {
    alertBanner.classList.add('hidden');
  }
}

function renderMonitoringChart(history) {
  const fraudRows = history.filter(r => r.metric_name === 'fraud_rate').reverse();
  if (!fraudRows.length) return;

  const labels = fraudRows.map(r =>
    new Date(r.created_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
  );

  const ctx = document.getElementById('monitoring-chart').getContext('2d');
  if (monitoringChart) monitoringChart.destroy();
  monitoringChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels,
      datasets: [
        {
          label: 'Observed Fraud Rate (%)',
          data: fraudRows.map(r => (r.metric_value * 100).toFixed(3)),
          borderColor: '#ef4444',
          backgroundColor: 'rgba(239,68,68,.1)',
          tension: 0.3,
          fill: true,
          yAxisID: 'y',
        },
        {
          label: 'PSI Score',
          data: fraudRows.map(r => r.psi_score ?? 0),
          borderColor: '#f59e0b',
          borderDash: [4, 4],
          tension: 0.3,
          yAxisID: 'y2',
        },
      ],
    },
    options: {
      responsive: true,
      interaction: { mode: 'index', intersect: false },
      plugins: { legend: { labels: { color: '#94a3b8', font: { size: 12 } } } },
      scales: {
        y: {
          type: 'linear', position: 'left',
          grid: { color: '#2e3347' },
          ticks: { color: '#94a3b8', callback: v => v + '%' },
        },
        y2: {
          type: 'linear', position: 'right',
          grid: { display: false },
          ticks: { color: '#f59e0b' },
        },
        x: {
          grid: { display: false },
          ticks: { color: '#94a3b8', maxTicksLimit: 10 },
        },
      },
    },
  });
}

// ── Helpers ───────────────────────────────────────────────────────────────────
function formatFeatureName(name) {
  const map = {
    amount: 'Transaction Amount', amount_log: 'Amount (log)', amount_z_score: 'Amount Z-Score',
    hour_of_day: 'Hour of Day', day_of_week: 'Day of Week', merchant_category: 'Merchant Category',
    distance_from_home: 'Distance from Home', is_weekend: 'Is Weekend',
    velocity_1h: 'Velocity (1h)', velocity_24h: 'Velocity (24h)',
    velocity_ratio: 'Velocity Ratio', night_flag: 'Night Transaction',
    high_distance_flag: 'High Distance',
  };
  return map[name] ?? name;
}

async function get(path) {
  const r = await fetch(API + path);
  if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
  return r.json();
}

async function post(path, body) {
  const r = await fetch(API + path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!r.ok) {
    const err = await r.json().catch(() => ({}));
    throw new Error(err.detail ?? `${r.status}`);
  }
  return r.json();
}
