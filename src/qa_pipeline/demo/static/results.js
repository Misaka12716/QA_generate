const rows = document.querySelector("#rows");
const count = document.querySelector("#count");
const detail = document.querySelector("#detail");
let current = "";

function fillSelect(id, values) {
  const select = document.querySelector(id);
  const existing = new Set([...select.options].map((item) => item.value));
  values.filter(Boolean).forEach((value) => {
    if (existing.has(value)) return;
    const option = document.createElement("option");
    option.value = value;
    option.textContent = value;
    select.append(option);
  });
}

function params() {
  const query = new URLSearchParams();
  ["pattern", "rule", "review", "block"].forEach((name) => {
    const value = document.querySelector("#" + name).value;
    if (value) query.set(name, value);
  });
  return query;
}

function teacherText(label) {
  return label || "待自动评估";
}

async function loadList() {
  const response = await fetch("/api/v1/result-cases?" + params().toString());
  const body = await response.json();
  if (!response.ok) {
    count.textContent = body.error ? body.error.message : "结果不可用";
    rows.replaceChildren();
    return;
  }
  const view = body.meta.view || {};
  document.querySelector("#banner").textContent = view.banner || "历史真实预测";
  document.querySelector("#limitation").textContent = view.limitation || "";
  fillSelect("#pattern", body.data.map((item) => item.pattern));
  fillSelect("#review", body.data.map((item) => item.human_review_status));
  const pilot = view.b_pilot_selected_n ? `；实验 B 小试 ${view.b_pilot_selected_n} / 父协议 ${view.b_parent_planned_n}` : "";
  count.textContent = `显示 ${body.data.length} 题。历史计划 ${view.protocol_n ?? "—"} 题${pilot}。教师分未返回时显示待自动评估。`;
  rows.replaceChildren();
  body.data.forEach((item) => {
    const tr = document.createElement("tr");
    if (item.case_id === current) tr.setAttribute("aria-current", "true");
    tr.innerHTML = `<td></td><td></td><td></td><td></td><td></td>`;
    const cells = [...tr.children];
    cells[0].textContent = `${item.block === "b_pilot" ? "新推理 · " : "历史 · "}${item.case_id}`;
    cells[1].textContent = `${item.pattern || ""} / ${item.expected_action || ""}`;
    cells[2].textContent = item.rule_status === "pass" ? "通过" : item.rule_status === "fail" ? "失败" : "未评分";
    cells[3].textContent = item.human_review_status || "";
    cells[4].textContent = teacherText(item.teacher_label);
    tr.addEventListener("click", () => openCase(item.case_id));
    rows.append(tr);
  });
}

function slotText(slot) {
  if (!slot || slot.match_status !== "matched") {
    const reasons = (slot && slot.unmatch_reasons) || ["未匹配"];
    return "未匹配：" + reasons.join("，");
  }
  return slot.text || "";
}

function slotMeta(slot) {
  if (!slot || slot.match_status !== "matched") return "未并入对比";
  const teacher = slot.teacher;
  const teacherLabel = teacher ? teacher.label : "待自动评估";
  const passed = slot.rule_passed === true ? "规则通过" : slot.rule_passed === false ? "规则失败" : "规则未判定";
  return `${passed}（辅助） · 自动评估 ${teacherLabel} · ${slot.model_id || ""} · ${slot.signature || ""}`;
}

async function openCase(caseId) {
  current = caseId;
  const response = await fetch("/api/v1/result-cases/" + encodeURIComponent(caseId));
  const body = await response.json();
  if (!response.ok) return;
  const item = body.data;
  detail.hidden = false;
  document.querySelector("#detail-title").textContent = item.case_id;
  document.querySelector("#detail-meta").textContent =
    `来源族 ${item.source_family_id || ""} · 人工 ${item.human_review_status || ""} · 金标自动评估 ${teacherText(item.teacher_label)}`;
  document.querySelector("#question").textContent = item.question || "";
  document.querySelector("#context").textContent = item.student_context || "";
  document.querySelector("#gold").textContent =
    `期望行为 ${item.expected_action || ""}\n候选答案 ${item.candidate_answer || ""}`;
  const base = (item.answers || {}).base || {};
  const adapter = (item.answers || {}).adapter || {};
  document.querySelector("#base-meta").textContent = slotMeta(base);
  document.querySelector("#adapter-meta").textContent = slotMeta(adapter);
  document.querySelector("#base-text").textContent = slotText(base);
  document.querySelector("#adapter-text").textContent = slotText(adapter);
  await loadList();
  detail.scrollIntoView({ block: "nearest" });
}

["pattern", "rule", "review", "block"].forEach((name) => {
  document.querySelector("#" + name).addEventListener("change", loadList);
});
loadList();
