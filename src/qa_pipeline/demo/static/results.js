const state = {
  mode: "rag_grounded",
  batch: "historical",
  relation: "",
  subset: "CB-paraphrase",
  serial: 0,
  view: "compare",
};

const headline = document.querySelector("#headline");
const tags = document.querySelector("#tags");
const cards = document.querySelector("#cards");
const bars = document.querySelector("#bars");
const cases = document.querySelector("#cases");
const status = document.querySelector("#status");
const scaleNote = document.querySelector("#scale-note");
const batchSelect = document.querySelector("#batch");

function clearCompare() {
  tags.replaceChildren();
  cards.replaceChildren();
  bars.replaceChildren();
  cases.replaceChildren();
  scaleNote.textContent = "";
  document.querySelectorAll(".qa-retry").forEach((node) => node.remove());
}

function setStatus(text, kind) {
  status.textContent = text;
  status.dataset.state = kind || "";
}

function button(label, className, onClick) {
  const node = document.createElement("button");
  node.type = "button";
  node.className = className;
  node.textContent = label;
  node.addEventListener("click", onClick);
  return node;
}

function formatCount(numerator, denominator) {
  if (numerator === null || numerator === undefined) return "—";
  if (denominator === null || denominator === undefined) return String(numerator);
  return `${numerator} / ${denominator}`;
}

async function loadCompare() {
  const serial = ++state.serial;
  clearCompare();
  setStatus("正在读取这一批次…", "");
  const query = new URLSearchParams({ task_mode: state.mode, batch: state.batch });
  let response;
  let body;
  try {
    response = await fetch("/api/v1/comparisons?" + query.toString());
    body = await response.json();
  } catch (error) {
    if (serial !== state.serial) return;
    setStatus("网络请求失败。可以重试。", "error");
    status.after(button("重试", "qa-inline qa-retry", () => loadCompare()));
    return;
  }
  if (serial !== state.serial) return;
  if (!response.ok) {
    const message = body.error ? body.error.message : "结果不可用";
    setStatus(message, "error");
    status.after(button("重试", "qa-inline qa-retry", () => loadCompare()));
    return;
  }
  const data = body.data || {};
  fillBatches(data.batches || []);
  headline.textContent = data.headline || "没有可显示的结论";
  (data.tags || []).forEach((tag) => {
    const span = document.createElement("span");
    span.className = "qa-tag" + (String(tag).includes("未") || String(tag).includes("受限") ? " warn" : "");
    span.textContent = tag;
    tags.append(span);
  });
  if (data.status === "not_executed" || data.availability === "empty") {
    setStatus(data.headline || "尚无闭卷训练对照", "empty");
    const note = document.createElement("p");
    note.className = "qa-empty";
    note.textContent = (data.limitations || []).join(" ");
    cases.append(note);
    if (data.protocol) {
      const extra = document.createElement("p");
      extra.className = "qa-note";
      const blockers = (data.protocol.blockers || []).join("；");
      extra.textContent = `协议状态：${data.protocol.status || "未记录"}。${blockers}`;
      cases.append(extra);
    }
    return;
  }
  (data.cards || []).forEach((card) => renderCard(card));
  const denominator = data.scale_denominator;
  scaleNote.textContent = denominator
    ? `条形图使用共同分母 ${denominator}。单位写在每行末尾。覆盖不足不会画成错误率。`
    : "当前没有可比较的共同分母。";
  (data.bars || []).forEach((bar) => renderBar(bar, denominator));
  const limits = document.createElement("p");
  limits.className = "qa-note";
  limits.textContent = (data.limitations || []).join(" ");
  bars.append(limits);
  setStatus(data.availability === "summary_only" ? "指标来自已保存汇总。" : "已读取当前批次。", "");
  await loadCases(serial);
}

function fillBatches(batches) {
  const current = state.batch;
  batchSelect.replaceChildren();
  batches.forEach((item) => {
    const option = document.createElement("option");
    option.value = item.batch_id;
    option.textContent = item.label || item.batch_id;
    batchSelect.append(option);
  });
  if ([...batchSelect.options].some((item) => item.value === current)) {
    batchSelect.value = current;
  } else if (batchSelect.options.length) {
    state.batch = batchSelect.value;
  }
}

function renderCard(card) {
  const node = document.createElement("button");
  node.type = "button";
  node.className = "qa-card";
  node.setAttribute("aria-pressed", state.relation && card.name && card.name.includes(state.relation) ? "true" : "false");
  const label = document.createElement("span");
  label.textContent = card.name || "";
  const value = document.createElement("b");
  value.textContent = formatCount(card.numerator, card.denominator);
  node.append(label, value);
  const filter = card.name === "未决" ? "unresolved" : card.name === "旧口径持平" ? "tie" : card.name === "改善" ? "improve" : "";
  if (filter) {
    node.addEventListener("click", () => {
      state.relation = state.relation === filter ? "" : filter;
      loadCompare();
    });
  }
  cards.append(node);
}

function renderBar(bar, scale) {
  const row = document.createElement("div");
  row.className = "qa-bar-row";
  const name = document.createElement("button");
  name.type = "button";
  name.textContent = bar.label || "";
  if (bar.filter) {
    name.addEventListener("click", () => {
      if (String(bar.filter).startsWith("CB-")) {
        state.subset = bar.filter;
        state.relation = "";
      } else {
        state.relation = bar.filter;
      }
      loadCompare();
    });
  }
  const track = document.createElement("div");
  track.className = "qa-track";
  const fill = document.createElement("i");
  const denom = Number(bar.denominator || scale || 0);
  const numer = Number(bar.numerator || 0);
  const width = denom > 0 ? Math.max(0, Math.min(100, (numer / denom) * 100)) : 0;
  fill.style.width = width + "%";
  track.append(fill);
  track.setAttribute("aria-hidden", "true");
  const count = document.createElement("span");
  count.textContent = `${formatCount(bar.numerator, bar.denominator)} ${bar.unit || ""}`;
  row.append(name, track, count);
  bars.append(row);
}

async function loadCases(serial) {
  const query = new URLSearchParams({
    task_mode: state.mode,
    relation: state.relation,
    subset: state.mode === "closed_book_domain" ? state.subset : "",
  });
  const response = await fetch(`/api/v1/comparisons/${encodeURIComponent(state.batch)}/cases?` + query.toString());
  const body = await response.json();
  if (serial !== state.serial) return;
  cases.replaceChildren();
  if (!response.ok) {
    const error = document.createElement("p");
    error.className = "qa-error";
    error.textContent = body.error ? body.error.message : "案例读取失败";
    cases.append(error, button("重试", "qa-inline", () => loadCompare()));
    return;
  }
  const meta = body.meta || {};
  if (!body.data || !body.data.length) {
    const empty = document.createElement("p");
    empty.className = "qa-empty";
    empty.textContent = meta.missing_reason || "当前筛选没有案例。";
    cases.append(empty);
    if (state.relation) {
      const count = document.createElement("p");
      count.className = "qa-note";
      count.textContent = meta.matched_filter_count === null || meta.matched_filter_count === undefined
        ? "汇总里没有这条筛选的原始案例。"
        : `汇总计数为 ${meta.matched_filter_count}，原始案例未随运行目录提供。`;
      cases.append(count);
      return;
    }
    if ((meta.identifiers_only || []).length) {
      const note = document.createElement("p");
      note.className = "qa-note";
      note.textContent = "下面只是规则分歧汇总里的标识，不是未决 4 题的原文。";
      cases.append(note);
    }
    (meta.identifiers_only || []).forEach((item) => {
      const line = document.createElement("p");
      line.className = "qa-id";
      line.textContent = `${item} · ${meta.identifier_note || "仅有标识"}`;
      cases.append(line);
    });
    return;
  }
  body.data.forEach((item) => cases.append(renderCase(item)));
}

function renderCase(item) {
  const wrap = document.createElement("article");
  wrap.className = "qa-case";
  const title = document.createElement("h3");
  title.textContent = item.question || "（问题缺失）";
  const id = document.createElement("p");
  id.className = "qa-id";
  id.textContent = `${item.case_id || "未记录"} · ${item.eval_subset || item.relation || ""}`;
  wrap.append(title, id);
  const grid = document.createElement("div");
  grid.className = "qa-answers";
  grid.append(renderAnswer("微调前", item.base || {}), renderAnswer("微调后", item.adapter || {}));
  wrap.append(grid);
  const details = document.createElement("details");
  details.className = "qa-evidence";
  const summary = document.createElement("summary");
  summary.textContent = item.task_mode === "closed_book_domain" ? "教师评分依据（学生未看到这些资料）" : "学生可见资料与审核理由";
  const body = document.createElement("div");
  const context = document.createElement("p");
  context.textContent = item.task_mode === "closed_book_domain"
    ? (item.judge_source_context || item.review_reason || "教师审核未执行。")
    : (item.student_context || "学生可见资料未记录。");
  const reason = document.createElement("p");
  reason.textContent = item.review_reason || "审核理由：未记录";
  body.append(context, reason);
  details.append(summary, body);
  wrap.append(details);
  return wrap;
}

function renderAnswer(label, slot) {
  const box = document.createElement("div");
  box.className = "qa-answer";
  const header = document.createElement("header");
  const name = document.createElement("strong");
  name.textContent = label;
  const copy = button("复制", "qa-inline", async () => {
    const text = slot.text || "";
    try {
      if (navigator.clipboard && window.isSecureContext) {
        await navigator.clipboard.writeText(text);
      } else {
        throw new Error("clipboard unavailable");
      }
      copy.textContent = "已复制";
    } catch (_error) {
      const area = document.createElement("textarea");
      area.value = text;
      area.setAttribute("readonly", "");
      area.style.position = "fixed";
      area.style.left = "-9999px";
      document.body.append(area);
      area.select();
      const ok = document.execCommand("copy");
      area.remove();
      copy.textContent = ok ? "已复制" : "复制失败";
    }
  });
  header.append(name, copy);
  const text = document.createElement("p");
  text.className = "collapsed";
  text.textContent = slot.text || slot.status || "（没有回答）";
  const toggle = button("展开全文", "qa-inline", () => {
    const closed = text.classList.toggle("collapsed");
    toggle.textContent = closed ? "展开全文" : "收起";
  });
  const meta = document.createElement("p");
  meta.className = "qa-note";
  meta.textContent = slot.label || "";
  box.append(header, text, toggle, meta);
  return box;
}

function switchMode(mode) {
  state.mode = mode;
  state.relation = "";
  state.batch = mode === "closed_book_domain" ? "cb1" : "historical";
  document.querySelector("#mode-rag").setAttribute("aria-pressed", mode === "rag_grounded" ? "true" : "false");
  document.querySelector("#mode-closed").setAttribute("aria-pressed", mode === "closed_book_domain" ? "true" : "false");
  clearCompare();
  headline.textContent = "正在切换模式…";
  loadCompare();
}

const coverageState = { stage: "consumed", reading: "" };

function coverageText(value) {
  if (value === null || value === undefined || value === "") return "未记录";
  return String(value);
}

function renderCoverage(data) {
  const statusNode = document.querySelector("#coverage-status");
  const summary = document.querySelector("#coverage-summary");
  const samples = document.querySelector("#coverage-samples");
  summary.replaceChildren();
  samples.replaceChildren();
  if (!data || data.status === "not_executed") {
    statusNode.textContent = data && data.reason ? "未执行：" + data.reason : "未执行";
    return;
  }
  const stageSummary = data.summary || {};
  if (stageSummary.available === false) {
    statusNode.textContent = "阶段 " + data.stage + " 未执行：" + (stageSummary.reason || "未记录");
    return;
  }
  statusNode.textContent = "阶段 " + data.stage + "。缺字段不是 0 分。";
  const chars = stageSummary.chars || {};
  const line = document.createElement("p");
  line.textContent = "行数 " + coverageText(stageSummary.rows)
    + "；字符 P50 " + coverageText(chars.p50)
    + "；来源族 " + coverageText(stageSummary.source_family_count)
    + "；样本族 " + coverageText(stageSummary.sample_family_count);
  summary.append(line);
  const types = stageSummary.q_type || {};
  const typeLine = document.createElement("p");
  typeLine.textContent = "题型 " + JSON.stringify(types);
  summary.append(typeLine);
  const predictions = data.predictions || {};
  if (predictions.adapter) {
    const pred = document.createElement("p");
    pred.className = "qa-note";
    pred.textContent = "adapter 停止证据：" + coverageText(predictions.adapter.finish_reason_evidence)
      + "；末尾 token " + coverageText(predictions.adapter.last_token);
    summary.append(pred);
  }
  (data.samples || []).filter((row) => !coverageState.reading || row.length_reading === coverageState.reading).forEach((row) => {
    const card = document.createElement("article");
    card.className = "qa-case";
    const title = document.createElement("h3");
    title.textContent = (row.document_title || "未记录标题") + " / " + (row.canonical_subject || "对象未记录");
    const body = document.createElement("p");
    body.textContent = (row.q_type || "unknown") + " · " + (row.question || "");
    const meta = document.createElement("p");
    meta.className = "qa-note";
    meta.textContent = "要点 " + coverageText(row.point_count)
      + "；证据 " + coverageText(row.evidence_count)
      + "；状态 " + (row.selection_status || row.action || "未记录")
      + "；字符 " + coverageText(row.char_len);
    card.append(title, body, meta);
    samples.append(card);
  });
  if (!samples.childElementCount) {
    const empty = document.createElement("p");
    empty.className = "qa-note";
    empty.textContent = coverageState.reading ? "当前筛选没有样本。" : "这一阶段没有样本行。";
    samples.append(empty);
  }
}

async function loadCoverage() {
  const statusNode = document.querySelector("#coverage-status");
  statusNode.textContent = "正在读取覆盖审计…";
  let response;
  let body;
  try {
    response = await fetch("/api/v1/coverage?stage=" + encodeURIComponent(coverageState.stage));
    body = await response.json();
  } catch (_error) {
    statusNode.textContent = "网络请求失败。";
    return;
  }
  if (!response.ok) {
    statusNode.textContent = body.error ? body.error.message : "覆盖数据不可用";
    return;
  }
  renderCoverage(body.data || {});
}

function ensureCoverageControls() {
  const stages = document.querySelector("#coverage-stages");
  if (!stages || stages.childElementCount) return;
  ["generated", "qualified", "selected", "exported", "consumed"].forEach((stage) => {
    stages.append(button(stage, "qa-inline", () => {
      coverageState.stage = stage;
      loadCoverage();
    }));
  });
  const filters = document.querySelector("#coverage-filters");
  const labels = {"": "全部", short_complete: "简短但完整", short_incomplete: "简短且不完整", long_unsupported: "较长但有编造"};
  Object.entries(labels).forEach(([reading, label]) => {
    filters.append(button(label, "qa-inline", () => {
      coverageState.reading = reading;
      loadCoverage();
    }));
  });
}

document.querySelectorAll(".qa-nav-list button").forEach((node) => {
  node.addEventListener("click", () => {
    state.view = node.dataset.view;
    document.querySelectorAll(".qa-nav-list button").forEach((item) => item.removeAttribute("aria-current"));
    node.setAttribute("aria-current", "page");
    document.querySelectorAll(".qa-workspace").forEach((section) => {
      section.hidden = section.id !== "view-" + state.view;
    });
    document.querySelector("#page-title").textContent = node.textContent.trim();
    document.querySelector("#nav").classList.remove("open");
    document.querySelector("#menu").setAttribute("aria-expanded", "false");
    if (state.view === "coverage") {
      ensureCoverageControls();
      loadCoverage();
    }
  });
});

document.querySelector("#menu").addEventListener("click", () => {
  const nav = document.querySelector("#nav");
  const open = nav.classList.toggle("open");
  document.querySelector("#menu").setAttribute("aria-expanded", open ? "true" : "false");
});

document.querySelector("#mode-rag").addEventListener("click", () => switchMode("rag_grounded"));
document.querySelector("#mode-closed").addEventListener("click", () => switchMode("closed_book_domain"));
batchSelect.addEventListener("change", () => {
  state.batch = batchSelect.value;
  state.relation = "";
  clearCompare();
  loadCompare();
});
document.querySelector("#clear-filter").addEventListener("click", () => {
  state.relation = "";
  loadCompare();
});

loadCompare();
