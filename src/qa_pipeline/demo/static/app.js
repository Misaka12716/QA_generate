const NAMES = {
  E1_baseline: "基线",
  E1_direct: "直接生成",
  E1_anchor: "传统锚点",
  E1_ku: "知识单元",
  E1_ours_full: "推荐 S+A",
  E1_ours_s_only: "推荐仅 S",
  E2_fixed: "固定比例",
  E2_gap: "缺口回填",
  E3_diagnostic: "抽样诊断",
  E4_multi: "多次验证",
  E6_behavior: "不足证据",
  E7_rationale: "解释蒸馏",
  E8_scale: "规模子集",
  E2_a1: "Self-Instruct",
  E2_a2: "锚点提问",
  E2_a3: "Tag-Evol",
  E2_a4: "无约束进化",
  E2_a5: "难度采样",
  E3_b1: "廉价教师",
  E3_b2: "强教师",
  E3_b3: "分级路由",
  E3_b4: "难题多教师",
  E4_c1: "无锚点",
  E4_c2: "TF-IDF",
  E4_c3: "NER+TF-IDF",
  E4_c4: "NER+关键句",
  E4_c5: "LLM 锚点",
  E2_recommended: "推荐栈",
  E3_evol: "深度进化",
  E4_full: "全过滤",
  E4_no_nli: "去掉 NLI",
  E4_no_ablation: "去掉消融",
  E4_no_judge: "去掉 Judge",
  E5_cost_min: "成本下界",
  E5_quality_max: "质量上界",
};

const COMPARE_GROUPS = [
  ["e1", "方案"],
  ["e2", "提问"],
  ["e3", "蒸馏"],
  ["e4", "锚点"],
];

const METRICS = [
  { key: "retention", label: "保留率", scale: "unit" },
  { key: "nli_mean", label: "NLI 均值", scale: "unit" },
  { key: "knowledge_gain_rate", label: "知识增益", scale: "unit" },
  { key: "judge_mean", label: "Judge", scale: "five" },
  { key: "type_entropy", label: "题型熵", scale: "max" },
  { key: "s_ratio", label: "S 级占比", scale: "unit" },
  { key: "evidence_grounded", label: "证据可定位", scale: "unit" },
  { key: "estimated_cost_usd", label: "估算成本", scale: "max", better: "low", digits: 5 },
  { key: "kept", label: "保留条数", scale: "max", digits: 0 },
];

const STAGES = [
  ["chunking", "切分"],
  ["anchor", "锚点"],
  ["question_gen", "提问"],
  ["evolution", "进化"],
  ["distillation", "蒸馏"],
  ["teacher_router", "路由"],
  ["filters", "过滤"],
  ["grading", "分层"],
];

const FILTER_ORDER = [
  "rule_clean",
  "exact_hash_dedup",
  "minhash_dedup",
  "semdedup",
  "evidence_substring",
  "llm_supported",
  "nli_fact",
  "llm_judge",
  "knowledge_ablation",
  "diversity_sample",
  "evolution",
];

const state = {
  experiments: [],
  left: { id: "", status: "kept", offset: 0 },
  right: { id: "", status: "kept", offset: 0 },
  compare: { group: "e1", status: "kept", offset: 0 },
};

let refreshTimer = 0;

function shortName(exp) {
  return NAMES[exp.id] || exp.recipe || exp.id;
}

function family(exp) {
  return String(exp.id || "").startsWith("E4") ? "ablation" : "pipeline";
}

function ordered(experiments) {
  const pipe = experiments.filter((exp) => family(exp) === "pipeline");
  const ablation = experiments.filter((exp) => family(exp) === "ablation");
  return { pipe, ablation, all: pipe.concat(ablation) };
}

function asNumber(raw) {
  if (raw === null || raw === undefined || raw === "") return null;
  const n = Number(raw);
  return Number.isFinite(n) ? n : null;
}

function fmt(value, digits) {
  const n = asNumber(value);
  if (n === null) return "—";
  if (digits === 0) return String(Math.round(n));
  const d = digits === undefined ? 3 : digits;
  return n.toFixed(d);
}

function specText(value) {
  if (Array.isArray(value)) return value.map(specText).join(" → ");
  if (value && typeof value === "object") {
    const params = Object.entries(value).filter(([key]) => key !== "name");
    if (!params.length) return String(value.name || "—");
    const bits = params.map(([key, item]) => {
      const shown = typeof item === "object" ? JSON.stringify(item) : String(item);
      return `${key}=${shown}`;
    });
    return `${value.name} · ${bits.join(", ")}`;
  }
  return value === null || value === undefined || value === "" ? "—" : String(value);
}

function specName(value) {
  if (Array.isArray(value)) return value.map((item) => (item && item.name) || specText(item)).join("→");
  if (value && typeof value === "object") return String(value.name || "—");
  return String(value ?? "—");
}

function el(tag, attrs, children) {
  const node = document.createElement(tag);
  Object.entries(attrs || {}).forEach(([key, value]) => {
    if (key === "class") node.className = value;
    else if (key === "html") node.innerHTML = value;
    else if (key.startsWith("on") && typeof value === "function") node.addEventListener(key.slice(2), value);
    else if (value !== null && value !== undefined) node.setAttribute(key, value);
  });
  (children || []).forEach((child) => {
    if (child === null || child === undefined) return;
    node.append(child.nodeType ? child : document.createTextNode(String(child)));
  });
  return node;
}

function showError(message) {
  document.getElementById("lede").textContent = message;
  document.getElementById("lede").classList.add("error");
}

function renderMetrics(groups) {
  const section = document.getElementById("metrics");
  section.replaceChildren();
  section.append(
    el("h2", { id: "metrics-title" }, ["质量与成本"]),
    el("p", { class: "hint" }, ["同一行里，青绿条是该指标更好的一端，红条是更差的一端。成本以更低为好，其余指标以更高为好。过滤消融复用推荐栈已经蒸馏好的问答，那几列成本只计再次过滤。"]),
  );
  const columns = groups.all;
  const headGroups = el("tr", {}, [
    el("th", { class: "metric-name", rowspan: "2" }, ["指标"]),
    groups.pipe.length ? el("th", { class: "group", colspan: String(groups.pipe.length) }, ["全管线"]) : null,
    groups.ablation.length ? el("th", { class: "group", colspan: String(groups.ablation.length) }, ["过滤消融"]) : null,
  ]);
  const headIds = el("tr", {}, columns.map((exp) => {
    const title = `${exp.id} ${exp.purpose || ""}`.trim();
    return el("th", { class: "eid", title }, [shortName(exp)]);
  }));
  const body = el("tbody", {}, METRICS.map((metric) => {
    const values = columns.map((exp) => asNumber((exp.metrics || {})[metric.key]));
    const present = values.filter((n) => n !== null);
    const max = present.length ? Math.max(...present) : 0;
    const min = present.length ? Math.min(...present) : 0;
    const cells = values.map((n) => {
      let width = 0;
      if (n !== null) {
        if (metric.scale === "unit") width = Math.max(0, Math.min(1, n)) * 100;
        else if (metric.scale === "five") width = Math.max(0, Math.min(5, n)) / 5 * 100;
        else width = max > 0 ? (n / max) * 100 : 0;
      }
      const betterLow = metric.better === "low";
      const comparable = present.length > 1 && min !== max && n !== null;
      let mark = "";
      if (comparable && ((betterLow && n === min) || (!betterLow && n === max))) mark = "best";
      if (comparable && ((betterLow && n === max) || (!betterLow && n === min))) mark = "worst";
      return el("td", { class: "cell" }, [
        el("span", { class: "num" }, [fmt(n, metric.digits)]),
        n === null ? null : el("span", { class: "track" }, [
          el("span", { class: mark ? `fill ${mark}` : "fill", style: `width:${width}%` }, []),
        ]),
      ]);
    });
    return el("tr", {}, [el("th", { class: "metric-name" }, [metric.label]), ...cells]);
  }));
  section.append(el("div", { class: "scroller" }, [
    el("table", {}, [el("thead", {}, [headGroups, headIds]), body]),
  ]));
}

function renderRecipes(groups) {
  const section = document.getElementById("recipes");
  section.replaceChildren();
  section.append(
    el("h2", { id: "recipes-title" }, ["配方差异"]),
    el("p", { class: "hint" }, ["浅底色表示该阶段策略与基线不同。悬停单元格可看参数。"]),
  );
  const baseline = groups.all.find((exp) => exp.id === "E1_baseline") || groups.all[0];
  const baseSnap = (baseline && baseline.recipe_snapshot) || {};
  const head = el("tr", {}, [
    el("th", { class: "stick" }, ["方案"]),
    ...STAGES.map(([, label]) => el("th", {}, [label])),
  ]);
  const body = el("tbody", {}, groups.all.map((exp) => {
    const snap = exp.recipe_snapshot || {};
    const cells = STAGES.map(([key]) => {
      const value = snap[key];
      const same = specName(value) === specName(baseSnap[key]);
      return el("td", { class: same ? "" : "diff", title: specText(value) }, [specName(value)]);
    });
    const title = exp.purpose || "";
    return el("tr", {}, [el("th", { class: "eid stick", title }, [shortName(exp)]), ...cells]);
  }));
  section.append(el("div", { class: "scroller" }, [
    el("table", {}, [el("thead", {}, [head]), body]),
  ]));
}

function filterKeys(experiments) {
  const seen = new Set();
  experiments.forEach((exp) => {
    Object.keys((exp.metrics || {}).rejected_by_filter || {}).forEach((key) => seen.add(key));
    ((exp.recipe_snapshot || {}).filters || []).forEach((item) => {
      if (item && item.name) seen.add(item.name);
    });
  });
  const orderedKeys = FILTER_ORDER.filter((key) => seen.has(key));
  seen.forEach((key) => {
    if (!orderedKeys.includes(key)) orderedKeys.push(key);
  });
  return orderedKeys;
}

function renderFilters(groups) {
  const section = document.getElementById("filters");
  section.replaceChildren();
  const keys = filterKeys(groups.all);
  section.append(
    el("h2", { id: "filters-title" }, ["过滤淘汰"]),
    el("p", { class: "hint" }, ["数字是该过滤器丢掉的条数。横线表示这组配方没有这一关。"]),
  );
  if (!keys.length) {
    section.append(el("p", { class: "empty" }, ["这批实验没有记录淘汰。"]));
    return;
  }
  const head = el("tr", {}, [el("th", { class: "stick" }, ["方案"]), ...keys.map((key) => el("th", { class: "eid" }, [key]))]);
  const body = el("tbody", {}, groups.all.map((exp) => {
    const rejected = (exp.metrics || {}).rejected_by_filter || {};
    const used = new Set(((exp.recipe_snapshot || {}).filters || []).map((item) => item && item.name));
    const cells = keys.map((key) => {
      if (!(key in rejected) && !used.has(key)) return el("td", {}, ["—"]);
      const n = Number(rejected[key]) || 0;
      return el("td", { class: n ? "count has" : "count" }, [String(n)]);
    });
    return el("tr", {}, [el("th", { class: "eid stick" }, [shortName(exp)]), ...cells]);
  }));
  section.append(el("div", { class: "scroller" }, [
    el("table", {}, [el("thead", {}, [head]), body]),
  ]));
}

function fillSelect(select, experiments, current) {
  select.replaceChildren(...experiments.map((exp) => {
    const option = el("option", { value: exp.id }, [`${shortName(exp)}（${exp.id}）`]);
    if (exp.id === current) option.selected = true;
    return option;
  }));
}

function sampleMeta(sample) {
  const bits = [sample.intent_primary || sample.q_type || "未分类", sample.grade || "未分层"];
  if (sample.evidence_state) bits.push(sample.evidence_state);
  if (sample.expected_action) bits.push(sample.expected_action);
  if (sample.nli_score !== null && sample.nli_score !== undefined) bits.push(`NLI ${fmt(sample.nli_score, 2)}`);
  if (sample.judge_overall !== null && sample.judge_overall !== undefined) bits.push(`Judge ${fmt(sample.judge_overall, 2)}`);
  if (sample.kb_gain !== null && sample.kb_gain !== undefined) bits.push(`增益 ${fmt(sample.kb_gain, 2)}`);
  if (sample.evolution_type) bits.push(sample.evolution_type);
  return bits.join("，");
}

async function loadPane(which) {
  const paneState = state[which];
  const list = document.getElementById(`${which}-list`);
  const page = document.getElementById(`${which}-page`);
  if (!paneState.id) {
    list.replaceChildren(el("li", { class: "empty" }, ["还没有可选方案。"]));
    return;
  }
  const params = new URLSearchParams({
    status: paneState.status,
    offset: String(paneState.offset),
    limit: "8",
  });
  const response = await fetch(`/api/experiments/${encodeURIComponent(paneState.id)}?${params}`);
  if (!response.ok) {
    list.replaceChildren(el("li", { class: "error" }, ["样本读取失败。"]));
    return;
  }
  const data = await response.json();
  const start = data.total ? data.offset + 1 : 0;
  const end = Math.min(data.total, data.offset + data.samples.length);
  page.textContent = data.total ? `${start}–${end} / ${data.total}` : "0";
  document.getElementById(`${which}-prev`).disabled = data.offset <= 0;
  document.getElementById(`${which}-next`).disabled = data.offset + data.limit >= data.total;
  if (!data.samples.length) {
    list.replaceChildren(el("li", { class: "empty" }, [paneState.status === "kept" ? "这组没有保留样本。" : "这组没有淘汰样本。"]));
    return;
  }
  list.replaceChildren(...data.samples.map((sample) => el("li", {}, [
    el("div", { class: "meta" }, [sampleMeta(sample)]),
    el("p", { class: "q" }, [sample.question || "（无问题）"]),
    el("p", { class: "a" }, [sample.answer || "（无答案）"]),
    el("p", { class: "ev" }, [sample.evidence_span || "（无证据片段）"]),
  ])));
}

function renderSamples(groups) {
  const section = document.getElementById("samples");
  section.replaceChildren();
  section.append(
    el("h2", { id: "samples-title" }, ["样本并排"]),
    el("p", { class: "hint" }, ["左右各选一个方案，对照保留或淘汰的问答和证据。"]),
  );
  if (!state.left.id && groups.all[0]) state.left.id = groups.all[0].id;
  if (!state.right.id) {
    const second = groups.all.find((exp) => exp.id !== state.left.id) || groups.all[0];
    state.right.id = second ? second.id : "";
  }
  const pair = el("div", { class: "pair" }, ["left", "right"].map((which) => {
    const select = el("select", {
      id: `${which}-select`,
      "aria-label": which === "left" ? "左栏方案" : "右栏方案",
      onchange: (event) => {
        state[which].id = event.target.value;
        state[which].offset = 0;
        loadPane(which);
      },
    });
    fillSelect(select, groups.all, state[which].id);
    const kept = el("button", {
      type: "button",
      "aria-pressed": state[which].status === "kept" ? "true" : "false",
      onclick: () => setStatus(which, "kept"),
    }, ["保留"]);
    const rejected = el("button", {
      type: "button",
      "aria-pressed": state[which].status === "rejected" ? "true" : "false",
      onclick: () => setStatus(which, "rejected"),
    }, ["淘汰"]);
    kept.id = `${which}-kept`;
    rejected.id = `${which}-rejected`;
    const prev = el("button", {
      type: "button",
      id: `${which}-prev`,
      onclick: () => {
        state[which].offset = Math.max(0, state[which].offset - 8);
        loadPane(which);
      },
    }, ["上一页"]);
    const next = el("button", {
      type: "button",
      id: `${which}-next`,
      onclick: () => {
        state[which].offset += 8;
        loadPane(which);
      },
    }, ["下一页"]);
    return el("div", { class: "pane" }, [
      el("div", { class: "controls" }, [
        el("label", { class: "field" }, [which === "left" ? "左栏" : "右栏", select]),
        kept,
        rejected,
        el("div", { class: "pager" }, [prev, el("span", { id: `${which}-page` }, ["0"]), next]),
      ]),
      el("ol", { class: "samples", id: `${which}-list` }, []),
    ]);
  }));
  section.append(pair);
  loadPane("left");
  loadPane("right");
}

function traceText(trace) {
  if (!trace || typeof trace !== "object") return "";
  return Object.entries(trace).map(([name, info]) => {
    const action = info && typeof info === "object" ? info.action : "";
    return action ? `${name}:${action}` : name;
  }).join("，");
}

function sampleBlock(sample) {
  const trace = traceText(sample.filter_trace);
  return el("div", { class: "cell-qa" }, [
    el("div", { class: "meta" }, [sampleMeta(sample)]),
    el("p", { class: "q" }, [sample.question || "（无问题）"]),
    el("p", { class: "a" }, [sample.answer || "（无答案）"]),
    el("p", { class: "ev" }, [sample.evidence_span || "（无证据片段）"]),
    trace ? el("p", { class: "trace" }, [trace]) : null,
  ]);
}

async function loadCompare() {
  const section = document.getElementById("compare");
  const params = new URLSearchParams({
    group: state.compare.group,
    status: state.compare.status,
    offset: String(state.compare.offset),
    limit: "4",
  });
  const response = await fetch(`/api/compare?${params}`);
  const board = document.getElementById("compare-board");
  if (!response.ok || !board) {
    if (board) board.replaceChildren(el("p", { class: "error" }, ["对照样本读取失败。"]));
    return;
  }
  const data = await response.json();
  const present = data.columns.some((col) => (data.rows || []).some((row) => (row.cells[col.id] || []).length));
  const start = data.total ? data.offset + 1 : 0;
  const end = Math.min(data.total, data.offset + data.rows.length);
  const page = document.getElementById("compare-page");
  if (page) page.textContent = data.total ? `${start}–${end} / ${data.total}` : "0";
  const prev = document.getElementById("compare-prev");
  const next = document.getElementById("compare-next");
  if (prev) prev.disabled = data.offset <= 0;
  if (next) next.disabled = data.offset + data.limit >= data.total;
  if (!data.rows.length || !present) {
    board.replaceChildren(el("p", { class: "empty" }, ["这组还没有可对照的样本。实验跑完一批后会自动出现。"]));
    return;
  }
  const head = el("tr", {}, [
    el("th", { class: "stick" }, ["材料"]),
    ...data.columns.map((col) => el("th", { class: "eid", title: col.purpose || col.id }, [NAMES[col.id] || col.id])),
  ]);
  const body = el("tbody", {}, data.rows.map((row) => el("tr", {}, [
    el("td", { class: "source" }, [row.source || "（无材料片段）"]),
    ...data.columns.map((col) => {
      const samples = row.cells[col.id] || [];
      if (!samples.length) return el("td", { class: "empty" }, ["—"]);
      return el("td", {}, samples.map(sampleBlock));
    }),
  ])));
  board.replaceChildren(el("div", { class: "scroller" }, [
    el("table", { class: "compare-table" }, [el("thead", {}, [head]), body]),
  ]));
  section.hidden = false;
}

function renderCompare() {
  const section = document.getElementById("compare");
  section.replaceChildren();
  section.hidden = false;
  const buttons = COMPARE_GROUPS.map(([group, label]) => {
    const button = el("button", {
      type: "button",
      "aria-pressed": state.compare.group === group ? "true" : "false",
      onclick: () => {
        state.compare.group = group;
        state.compare.offset = 0;
        renderCompare();
      },
    }, [label]);
    button.dataset.group = group;
    return button;
  });
  const kept = el("button", {
    type: "button",
    "aria-pressed": state.compare.status === "kept" ? "true" : "false",
    onclick: () => {
      state.compare.status = "kept";
      state.compare.offset = 0;
      renderCompare();
    },
  }, ["保留"]);
  const rejected = el("button", {
    type: "button",
    "aria-pressed": state.compare.status === "rejected" ? "true" : "false",
    onclick: () => {
      state.compare.status = "rejected";
      state.compare.offset = 0;
      renderCompare();
    },
  }, ["淘汰"]);
  const prev = el("button", {
    type: "button",
    id: "compare-prev",
    onclick: () => {
      state.compare.offset = Math.max(0, state.compare.offset - 4);
      loadCompare();
    },
  }, ["上一页"]);
  const next = el("button", {
    type: "button",
    id: "compare-next",
    onclick: () => {
      state.compare.offset += 4;
      loadCompare();
    },
  }, ["下一页"]);
  section.append(
    el("h2", { id: "compare-title" }, ["同源文本对照"]),
    el("p", { class: "hint" }, ["同一行是同一段材料或同一道题。列是不同方法生成的问题、答案、证据和分层。"]),
    el("div", { class: "controls" }, [
      ...buttons,
      kept,
      rejected,
      el("div", { class: "pager" }, [prev, el("span", { id: "compare-page" }, ["0"]), next]),
    ]),
    el("div", { id: "compare-board" }, []),
  );
  loadCompare();
}

function setStatus(which, status) {
  state[which].status = status;
  state[which].offset = 0;
  document.getElementById(`${which}-kept`).setAttribute("aria-pressed", status === "kept" ? "true" : "false");
  document.getElementById(`${which}-rejected`).setAttribute("aria-pressed", status === "rejected" ? "true" : "false");
  loadPane(which);
}

function teacherMode(payload) {
  const name = payload.teacher_model;
  if (payload.llm === "live") return name ? `教师 ${name}` : "真实教师";
  if (payload.llm === "local") return "本地教师";
  return "离线对照";
}

async function main() {
  const response = await fetch("/api/suite");
  if (!response.ok) {
    const detail = await response.json().catch(() => ({}));
    showError(detail.detail || "读不到实验结果。请先运行 qa-pipeline experiment。");
    return;
  }
  const payload = await response.json();
  state.experiments = payload.experiments || [];
  const groups = ordered(state.experiments);
  const file = (payload.input || "").split("/").filter(Boolean).pop() || "未记录文档";
  const progress = payload.progress;
  const progressText = progress ? `，已完成 ${progress.done}/${progress.total}` : "";
  const lede = document.getElementById("lede");
  lede.classList.remove("error");
  lede.textContent =
    `${payload.suite || "实验"}，${teacherMode(payload)}，材料是${file}，共 ${groups.all.length} 组${progressText}。`;
  if (!groups.all.length) {
    showError("实验目录里还没有完成的方案。");
    if (refreshTimer) clearTimeout(refreshTimer);
    if (progress && progress.done < progress.total) refreshTimer = setTimeout(main, 15000);
    return;
  }
  renderMetrics(groups);
  renderRecipes(groups);
  renderFilters(groups);
  renderCompare();
  renderSamples(groups);
  if (refreshTimer) clearTimeout(refreshTimer);
  if (progress && progress.done < progress.total) refreshTimer = setTimeout(main, 15000);
}

main().catch(() => showError("对照台没有加载完成。"));
