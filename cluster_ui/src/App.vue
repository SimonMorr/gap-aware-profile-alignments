<script setup>
import { ref, computed, watch, onMounted } from 'vue'

const MODEL_LABELS = { classic: 'Linear', affine: 'Affine', log: 'Logarithmic', power: 'Power' }
const THR_LABELS = { '_pn30': 'pn30', '_pn50': 'pn50', '_pn70': 'pn70', '': 'model' }
const M = { t: 26, r: 64, b: 8, l: 16 }, ROW = 13, INNERW = 500

const logsData = ref([]), models = ref([])
const log = ref(''), model = ref('affine'), maxVariants = ref(150), threshold = ref('')
const theta = ref(0.8)
const prep = ref(null), tau = ref(0), loading = ref(false), error = ref('')
const tip = ref({ show: false, x: 0, y: 0, html: '' })

// selected cluster -> profile
const selected = ref(null)          // { members: [ids], key }
const profile = ref(null)           // { reference, profile, medoid, n_cases, n_variants }
const profileLoading = ref(false), profileError = ref('')

const thresholds = computed(() => {
  const l = logsData.value.find(x => x.name === log.value)
  return l ? l.thresholds : []
})

function prepare(raw) {
  let yi = 0; const leaves = [], merges = []
  ;(function walk(n) {
    if (n.leaf) { n._y = yi++; leaves.push(n); return }
    n.children.forEach(walk)
    n._y = (n.children[0]._y + n.children[n.children.length - 1]._y) / 2
    merges.push(n.height)
  })(raw.tree)
  merges.sort((a, b) => b - a)
  return { root: raw.tree, leaves, leafCount: leaves.length, maxH: raw.tree.height, merges, acts: raw.activities, total: raw.total_freq }
}
function tauForK(p, k) {
  if (k <= 1) return p.maxH + 1e-9
  if (k >= p.leafCount) return -1e-9
  return (p.merges[k - 2] + p.merges[k - 1]) / 2
}
function clustersAt(p, t) {
  const r = []
  ;(function rec(n) { if (n.leaf || n.height <= t) r.push(n); else n.children.forEach(rec) })(p.root)
  return r
}
function leavesUnder(n, acc = []) { if (n.leaf) { acc.push(n); return acc } n.children.forEach(c => leavesUnder(c, acc)); return acc }
function hue(i, n) { return `hsl(${(i * 360 / Math.max(n, 1) + 8) % 360} 58% 52%)` }
function clearProfile() { selected.value = null; profile.value = null; profileError.value = '' }

async function loadTree(keepK) {
  if (!log.value) return
  loading.value = true; error.value = ''; clearProfile()
  try {
    const r = await fetch(`/api/tree?log=${encodeURIComponent(log.value)}&model=${model.value}&max_variants=${maxVariants.value}`)
    if (!r.ok) throw new Error('API ' + r.status)
    const p = prepare(await r.json()); prep.value = p
    tau.value = tauForK(p, Math.min(keepK || 12, p.leafCount))
  } catch (e) { error.value = String(e.message || e); prep.value = null }
  finally { loading.value = false }
}

onMounted(async () => {
  try {
    const d = await (await fetch('/api/logs')).json()
    logsData.value = d.logs; models.value = d.models
    log.value = d.logs[0].name
    threshold.value = d.logs[0].thresholds[0] ?? ''
    await loadTree(12)
  } catch (e) { error.value = 'Cannot reach API on :8000 — start the backend (see README).' }
})
watch(log, () => { threshold.value = thresholds.value[0] ?? ''; loadTree(12) })
watch([model, maxVariants], () => loadTree(clusters.value.length || 12))
watch(tau, () => { if (selected.value) clearProfile() })
watch(threshold, () => { if (selected.value) fetchProfile() })

async function selectCluster(node, key) {
  selected.value = { members: leavesUnder(node).map(l => l.id), key }
  await fetchProfile()
}
async function fetchProfile() {
  if (!selected.value) return
  profileLoading.value = true; profileError.value = ''; profile.value = null
  try {
    const r = await fetch('/api/profile', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ log: log.value, model: model.value, threshold: threshold.value, max_variants: maxVariants.value, members: selected.value.members }),
    })
    if (!r.ok) throw new Error('API ' + r.status)
    profile.value = await r.json()
  } catch (e) { profileError.value = String(e.message || e) }
  finally { profileLoading.value = false }
}

const clusters = computed(() => prep.value ? clustersAt(prep.value, tau.value) : [])
const ordered = computed(() => [...clusters.value].sort((a, b) => a._y - b._y))
const colorByCluster = computed(() => { const m = new Map(); ordered.value.forEach((c, i) => m.set(c, hue(i, ordered.value.length))); return m })
const leafColor = computed(() => {
  const m = new Map()
  clusters.value.forEach(c => leavesUnder(c).forEach(l => m.set(l, colorByCluster.value.get(c))))
  return m
})
const X = h => M.l + INNERW * (1 - h / prep.value.maxH)
const Y = n => M.t + n._y * ROW + ROW / 2
const dims = computed(() => prep.value ? { w: M.l + INNERW + M.r, h: M.t + M.b + prep.value.leafCount * ROW } : { w: 0, h: 0 })
const cutX = computed(() => prep.value ? X(tau.value) : 0)
const ticks = computed(() => prep.value ? [0, 1, 2, 3, 4].map(i => { const hv = prep.value.maxH * i / 4; return { x: X(hv), label: hv.toFixed(1) } }) : [])
const links = computed(() => {
  const p = prep.value; if (!p) return []
  const out = [], lc = leafColor.value, t = tau.value
  ;(function draw(n) {
    if (n.leaf) return
    const px = X(n.height), ys = n.children.map(Y)
    const col = n.height <= t ? lc.get(leavesUnder(n)[0]) : 'var(--link)'
    out.push({ d: `M${px},${Math.min(...ys)} L${px},${Math.max(...ys)}`, stroke: col })
    n.children.forEach(c => {
      const cx = X(c.height), cy = Y(c)
      const cc = (n.height <= t || c.height <= t) ? lc.get(leavesUnder(c)[0]) : 'var(--link)'
      out.push({ d: `M${px},${cy} L${cx},${cy}`, stroke: cc }); draw(c)
    })
  })(p.root)
  return out
})
const leafDots = computed(() => {
  const p = prep.value; if (!p) return []
  const lc = leafColor.value, x = X(0)
  return p.leaves.map(l => ({ cx: x, cy: Y(l), fill: lc.get(l) || 'var(--link)', freq: l.freq, l }))
})
const clusterList = computed(() => {
  const p = prep.value; if (!p) return []
  return clusters.value.map((c, i) => {
    const ls = leavesUnder(c), freq = ls.reduce((a, l) => a + l.freq, 0)
    return { node: c, key: c.id, color: colorByCluster.value.get(c), count: ls.length, freq, pct: 100 * freq / p.total, members: ls }
  }).sort((a, b) => b.freq - a.freq)
})
const coverage = computed(() => {
  const p = prep.value; if (!p) return 0
  const f = clusters.value.reduce((s, c) => s + leavesUnder(c).reduce((a, l) => a + l.freq, 0), 0)
  return 100 * f / p.total
})

const kModel = computed({
  get: () => clusters.value.length,
  set: v => { const p = prep.value; if (!p) return; tau.value = tauForK(p, Math.max(1, Math.min(p.leafCount, parseInt(v) || 1))) },
})
const sliderValue = computed({
  get: () => prep.value ? Math.round((1 - tau.value / prep.value.maxH) * 1000) : 0,
  set: v => { const p = prep.value; if (!p) return; tau.value = p.maxH * (1 - v / 1000) },
})

// profile viz layout
const pv = computed(() => {
  if (!profile.value) return null
  const P = profile.value.profile, ref = profile.value.reference
  const colW = 54, padL = 14, padR = 14, top = 16, barH = 100, labelH = 60
  const positions = ref.map((act, idx) => ({
    i: idx + 1, act, abbr: abbr(act), p: P[idx], stable: P[idx] >= theta.value, x: padL + idx * colW,
  }))
  return { positions, colW, top, barH, labelH,
    width: padL + ref.length * colW + padR, height: top + barH + labelH, thetaY: top + barH * (1 - theta.value) }
})
const nStable = computed(() => pv.value ? pv.value.positions.filter(p => p.stable).length : 0)

function abbr(name) { return name.split(/\s+/).map(w => w[0]).join('').toUpperCase() }
function decode(seq) { return seq.map(i => abbr(prep.value.acts[i])).join(' › ') }
function full(seq) { return seq.map(i => prep.value.acts[i]).join(' › ') }
function showTip(e, l) { tip.value = { show: true, x: e.clientX + 14, y: e.clientY + 14, html: `freq ${l.freq} · len ${l.seq.length}<br>${full(l.seq)}` } }
function hideTip() { tip.value.show = false }
function fmt(n) { return n.toLocaleString() }
</script>

<template>
  <div class="wrap">
    <header class="top">
      <h1>Trace Variant Clustering</h1>
      <p>Agglomerative hierarchical clustering of trace variants under the gap-aware dissimilarity
        <span class="m">δ<sub>g</sub></span>. Drag the cut to a linkage threshold <span class="m">τ</span>;
        pick a cluster to see its model-conforming <b>reference trace</b> and position <b>profile</b>
        <span class="m">P<sub>ℓ</sub>(p)</span>, with the stability threshold <span class="m">θ</span>
        marking the stable positions.</p>
    </header>

    <div class="controls">
      <div class="ctl"><label>Event log</label>
        <select v-model="log">
          <option v-for="l in logsData" :key="l.name" :value="l.name">{{ l.name.replace(/_/g, ' ') }}</option>
        </select>
      </div>
      <div class="ctl"><label>Gap cost model (g)</label>
        <div class="seg">
          <button v-for="m in models" :key="m" :aria-pressed="m === model" @click="model = m">{{ MODEL_LABELS[m] || m }}</button>
        </div>
      </div>
      <div class="ctl"><label>Model (noise τ)</label>
        <select v-model="threshold">
          <option v-for="t in thresholds" :key="t" :value="t">{{ THR_LABELS[t] ?? t }}</option>
        </select>
      </div>
      <div class="ctl"><label>Variants</label>
        <select v-model.number="maxVariants">
          <option :value="100">top 100</option><option :value="150">top 150</option><option :value="250">top 250</option>
        </select>
      </div>
      <div class="ctl"><label>Clusters (k)</label>
        <input type="number" min="1" :max="prep ? prep.leafCount : 150" v-model.number="kModel" />
      </div>
      <div class="ctl thetactl"><label>Stability θ = <span class="m">{{ theta.toFixed(2) }}</span></label>
        <input type="range" min="0" max="1" step="0.01" v-model.number="theta" />
      </div>
      <div class="readouts">
        <div class="ro"><span class="k">τ (cut)</span><span class="v">{{ prep ? tau.toFixed(2) : '–' }}</span></div>
        <div class="ro"><span class="k">Clusters</span><span class="v">{{ prep ? clusters.length : '–' }}</span></div>
        <div class="ro"><span class="k">Cases</span><span class="v">{{ prep ? coverage.toFixed(1) : '–' }}<small>%</small></span></div>
      </div>
    </div>

    <div class="stage">
      <div class="chart">
        <div class="slider-row">
          <span class="cap">1 cluster</span>
          <input type="range" min="0" max="1000" v-model.number="sliderValue" :disabled="!prep" />
          <span class="cap">all singletons</span>
        </div>
        <div class="scroll">
          <div v-if="loading" class="state">Computing δ<sub>g</sub> and clustering… (first time per log/model is slow)</div>
          <div v-else-if="error" class="state err">{{ error }}</div>
          <svg v-else-if="prep" :width="dims.w" :height="dims.h" :viewBox="`0 0 ${dims.w} ${dims.h}`">
            <g>
              <line v-for="(t, i) in ticks" :key="'t' + i" :x1="t.x" :y1="M.t - 6" :x2="t.x" :y2="dims.h - M.b" stroke="var(--line)" stroke-width="1" />
              <text v-for="(t, i) in ticks" :key="'tl' + i" :x="t.x" :y="M.t - 10" text-anchor="middle" class="axis">{{ t.label }}</text>
            </g>
            <path v-for="(lk, i) in links" :key="'l' + i" :d="lk.d" :stroke="lk.stroke" fill="none" stroke-width="1.4" />
            <g v-for="(d, i) in leafDots" :key="'d' + i">
              <circle :cx="d.cx" :cy="d.cy" r="2.6" :fill="d.fill" />
              <text :x="d.cx + 7" :y="d.cy + 3" class="freqtxt">{{ d.freq }}</text>
              <rect :x="M.l" :y="d.cy - ROW / 2" :width="INNERW + M.r" :height="ROW" fill="transparent"
                @mousemove="showTip($event, d.l)" @mouseleave="hideTip" />
            </g>
            <line :x1="cutX" :y1="M.t - 6" :x2="cutX" :y2="dims.h - M.b" stroke="var(--accent)" stroke-width="2" stroke-dasharray="4 3" />
          </svg>
        </div>
      </div>

      <aside class="panel">
        <h2>Clusters at this cut</h2>
        <div class="clist">
          <div v-for="(c, i) in clusterList" :key="i" class="cl" :class="{ sel: selected && selected.key === c.key }">
            <div class="clhead">
              <span class="sw" :style="{ background: c.color }"></span>
              <span class="meta">{{ c.count }} variant{{ c.count > 1 ? 's' : '' }}</span>
              <span class="sub">{{ fmt(c.freq) }} · {{ c.pct.toFixed(1) }}%</span>
              <button class="pbtn" @click="selectCluster(c.node, c.key)">Profile</button>
            </div>
            <details>
              <summary>members</summary>
              <div class="members">
                <div v-for="(m, j) in c.members.slice(0, 40)" :key="j" class="mem"><b>{{ m.freq }}</b>&nbsp;&nbsp;{{ decode(m.seq) }}</div>
                <div v-if="c.members.length > 40" class="mem">… +{{ c.members.length - 40 }} more</div>
              </div>
            </details>
          </div>
        </div>
      </aside>
    </div>

    <!-- Profile of the selected cluster -->
    <section class="profile">
      <h2>Cluster profile <span v-if="threshold" class="thr">· model {{ THR_LABELS[threshold] }}</span></h2>
      <div v-if="!selected" class="hint">Select a cluster (▸ Profile) to see its reference trace and stability profile.</div>
      <div v-else-if="profileLoading" class="hint">Computing reference and profile…</div>
      <div v-else-if="profileError" class="hint err">{{ profileError }}</div>
      <div v-else-if="profile && pv">
        <div class="pmeta">
          Reference of length <b>{{ profile.reference.length }}</b> ·
          <b>{{ nStable }}</b> stable position{{ nStable === 1 ? '' : 's' }} at θ={{ theta.toFixed(2) }} ·
          {{ profile.n_variants }} variants / {{ fmt(profile.n_cases) }} cases.
        </div>
        <div class="pscroll">
          <svg :width="pv.width" :height="pv.height" :viewBox="`0 0 ${pv.width} ${pv.height}`">
            <line :x1="6" :y1="pv.thetaY" :x2="pv.width - 6" :y2="pv.thetaY" stroke="var(--accent)" stroke-width="1.4" stroke-dasharray="4 3" />
            <text :x="pv.width - 8" :y="pv.thetaY - 4" text-anchor="end" class="thetalab">θ</text>
            <g v-for="pos in pv.positions" :key="pos.i">
              <rect :x="pos.x + 8" :y="pv.top + pv.barH * (1 - pos.p)" :width="pv.colW - 16" :height="pv.barH * pos.p"
                :fill="pos.stable ? 'var(--accent)' : 'var(--bar-mut)'" rx="2" />
              <text :x="pos.x + pv.colW / 2" :y="pv.top + pv.barH * (1 - pos.p) - 4" text-anchor="middle" class="pval">{{ pos.p.toFixed(2) }}</text>
              <text :x="pos.x + pv.colW / 2" :y="pv.top + pv.barH + 16" text-anchor="middle"
                :class="['plabel', { stableL: pos.stable }]">{{ pos.abbr }}</text>
              <text :x="pos.x + pv.colW / 2" :y="pv.top + pv.barH + 30" text-anchor="middle" class="ppos">p{{ pos.i }}</text>
            </g>
          </svg>
        </div>
        <div class="reftxt"><b>Reference:</b> {{ profile.reference.join('  ›  ') }}</div>
        <div class="legend"><span class="lg"><i class="sw" style="background:var(--accent)"></i>stable (P ≥ θ)</span>
          <span class="lg"><i class="sw" style="background:var(--bar-mut)"></i>variable</span></div>
      </div>
    </section>

    <p class="note">Top {{ maxVariants }} most frequent variants of the log. δ<sub>g</sub> and clustering are model-independent;
      the reference and profile depend on the chosen discovered net (noise τ). θ is applied on the client — drag it to
      re-mark stable positions instantly.</p>
  </div>

  <div class="tip" v-show="tip.show" :style="{ left: tip.x + 'px', top: tip.y + 'px' }" v-html="tip.html"></div>
</template>

<style scoped>
.wrap { max-width: 1180px; margin: 0 auto; padding-inline: 20px; padding-block: 22px 40px }
.top h1 { font-size: 22px; font-weight: 600; letter-spacing: -.01em; margin: 0; text-wrap: balance }
.top p { margin: 4px 0 0; color: var(--ink-soft); font-size: 13.5px; max-width: 78ch }
.m { font-family: var(--mono) }
.controls { display: flex; flex-wrap: wrap; gap: 14px 22px; align-items: flex-end; margin-top: 18px;
  padding: 14px 16px; background: var(--panel); border: 1px solid var(--line); border-radius: 12px }
.ctl { display: flex; flex-direction: column; gap: 5px }
.ctl label { font-size: 10.5px; text-transform: uppercase; letter-spacing: .09em; color: var(--ink-faint); font-weight: 500 }
.thetactl { min-width: 150px }
.thetactl input[type=range] { accent-color: var(--accent) }
select, input[type=number] { font-family: var(--sans); font-size: 13.5px; color: var(--ink);
  background: var(--panel-2); border: 1px solid var(--line-strong); border-radius: 8px; padding: 7px 10px }
select { min-width: 130px }
input[type=number] { min-width: 78px; font-family: var(--mono) }
.seg { display: inline-flex; border: 1px solid var(--line-strong); border-radius: 8px; overflow: hidden; background: var(--panel-2) }
.seg button { font-family: var(--sans); font-size: 13px; color: var(--ink-soft); background: transparent; border: 0;
  padding: 7px 13px; cursor: pointer; border-right: 1px solid var(--line) }
.seg button:last-child { border-right: 0 }
.seg button[aria-pressed=true] { background: var(--ink); color: var(--bg); font-weight: 500 }
.readouts { display: flex; gap: 22px; margin-left: auto; align-items: flex-end }
.ro { display: flex; flex-direction: column; gap: 2px }
.ro .k { font-size: 10.5px; text-transform: uppercase; letter-spacing: .09em; color: var(--ink-faint); font-weight: 500 }
.ro .v { font-family: var(--mono); font-size: 19px; font-weight: 500; font-variant-numeric: tabular-nums }
.ro .v small { font-size: 12px; color: var(--ink-soft) }
.stage { display: grid; grid-template-columns: 1fr 340px; gap: 18px; margin-top: 18px; align-items: start }
.chart { background: var(--panel); border: 1px solid var(--line); border-radius: 12px; overflow: hidden }
.slider-row { display: flex; align-items: center; gap: 12px; padding: 12px 16px 4px }
.slider-row .cap { font-size: 11px; color: var(--ink-faint); font-family: var(--mono); white-space: nowrap }
input[type=range] { flex: 1; accent-color: var(--accent); height: 22px }
.scroll { max-height: 60vh; overflow: auto; padding: 0 4px 10px }
.state { padding: 40px 20px; color: var(--ink-soft); font-size: 14px }
.state.err { color: #d14343 }
svg { display: block }
.axis { font-family: var(--mono); font-size: 10.5px; fill: var(--ink-faint) }
.freqtxt { font-family: var(--mono); font-size: 9px; fill: var(--ink-faint) }
.panel { background: var(--panel); border: 1px solid var(--line); border-radius: 12px; padding: 4px 4px 8px;
  position: sticky; top: 12px; max-height: calc(60vh + 56px); display: flex; flex-direction: column }
.panel h2 { font-size: 12px; text-transform: uppercase; letter-spacing: .08em; color: var(--ink-faint); font-weight: 600; margin: 12px 14px 8px }
.clist { overflow: auto; padding: 0 8px 4px }
.cl { border: 1px solid var(--line); border-radius: 9px; margin-bottom: 7px; overflow: hidden; background: var(--panel-2) }
.cl.sel { border-color: var(--accent); box-shadow: 0 0 0 1px var(--accent) }
.clhead { display: flex; align-items: center; gap: 9px; padding: 8px 11px }
.sw { width: 11px; height: 11px; border-radius: 3px; flex: none; display: inline-block }
.cl .meta { font-size: 13px; font-weight: 500 }
.cl .sub { font-family: var(--mono); font-size: 11px; color: var(--ink-soft); margin-left: auto; font-variant-numeric: tabular-nums }
.pbtn { font-family: var(--sans); font-size: 11px; color: var(--accent); background: transparent;
  border: 1px solid var(--line-strong); border-radius: 6px; padding: 3px 8px; cursor: pointer }
.pbtn:hover { border-color: var(--accent) }
.cl details summary { list-style: none; cursor: pointer; font-size: 11px; color: var(--ink-faint);
  font-family: var(--mono); padding: 0 11px 8px 31px }
.cl details summary::-webkit-details-marker { display: none }
.members { padding: 0 12px 10px 31px }
.mem { font-family: var(--mono); font-size: 11px; color: var(--ink-soft); padding: 3px 0; white-space: nowrap; overflow: hidden; text-overflow: ellipsis }
.mem b { color: var(--ink); font-weight: 500 }

.profile { margin-top: 18px; background: var(--panel); border: 1px solid var(--line); border-radius: 12px; padding: 6px 16px 16px }
.profile h2 { font-size: 12px; text-transform: uppercase; letter-spacing: .08em; color: var(--ink-faint); font-weight: 600 }
.profile h2 .thr { color: var(--ink-soft); font-family: var(--mono); text-transform: none; letter-spacing: 0 }
.hint { color: var(--ink-faint); font-size: 13px; padding: 12px 0 }
.hint.err { color: #d14343 }
.pmeta { font-size: 13px; color: var(--ink-soft); margin: 2px 0 8px }
.pmeta b { color: var(--ink); font-weight: 600 }
.pscroll { overflow-x: auto; padding-bottom: 4px }
.pval { font-family: var(--mono); font-size: 9.5px; fill: var(--ink-faint); font-variant-numeric: tabular-nums }
.plabel { font-family: var(--sans); font-size: 11px; font-weight: 500; fill: var(--ink-soft) }
.plabel.stableL { fill: var(--accent) }
.ppos { font-family: var(--mono); font-size: 9px; fill: var(--ink-faint) }
.thetalab { font-family: var(--mono); font-size: 11px; fill: var(--accent) }
.reftxt { font-family: var(--mono); font-size: 11px; color: var(--ink-soft); margin-top: 8px; line-height: 1.7 }
.reftxt b { color: var(--ink) }
.legend { display: flex; gap: 18px; margin-top: 10px; font-size: 11.5px; color: var(--ink-soft) }
.legend .lg { display: inline-flex; align-items: center; gap: 6px }
.legend i.sw { width: 10px; height: 10px; border-radius: 2px }

.note { margin-top: 14px; color: var(--ink-faint); font-size: 12px; max-width: 82ch }
@media (max-width: 820px) { .stage { grid-template-columns: 1fr } .panel { position: static; max-height: none } .readouts { margin-left: 0; width: 100% } }
</style>

<style>
:root { --bar-mut: #C3CAD3 }
@media (prefers-color-scheme: dark) { :root { --bar-mut: #39424D } }
.tip { position: fixed; pointer-events: none; z-index: 9; background: var(--ink); color: var(--bg);
  font-family: var(--mono); font-size: 11px; line-height: 1.45; padding: 7px 10px; border-radius: 7px;
  max-width: 340px; white-space: normal; box-shadow: 0 6px 22px rgba(0, 0, 0, .22) }
</style>
