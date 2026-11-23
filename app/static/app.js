// Small helpers for the teacher pages. All actions go through the /api endpoints.

const POLL_MS = 2500;
const STATUS_TEXT = { pending: "Waiting", processing: "Grading", evaluated: "Ready for review",
                      finalized: "Finalised", error: "Error" };
const BUSY = new Set(["pending", "processing"]);

async function api(method, url, body) {
  const opts = { method };
  if (body instanceof FormData) {
    opts.body = body;
  } else if (body !== undefined) {
    opts.headers = { "Content-Type": "application/json" };
    opts.body = JSON.stringify(body);
  }
  let res;
  try {
    res = await fetch(url, opts);
  } catch (_) {
    throw new Error("Could not reach the server. Check that it is still running and try again.");
  }
  let data = null;
  try { data = await res.json(); } catch (_) { /* not JSON */ }
  if (!res.ok) {
    let msg = data && data.detail ? data.detail : `Request failed (${res.status})`;
    if (Array.isArray(msg)) msg = msg.map((d) => d.msg).join("; ");   // FastAPI validation errors
    throw new Error(msg);
  }
  return data;
}

// ---------- small UI helpers ----------
function toast(text, isError = false) {
  const el = document.getElementById("toast");
  el.textContent = text;
  el.classList.toggle("error", isError);
  el.classList.add("show");
  clearTimeout(el._timer);
  el._timer = setTimeout(() => el.classList.remove("show"), 3200);
}

function showError(container, message) {
  const el = container.querySelector("[data-error]");
  if (!el) return toast(message, true);
  el.textContent = message;
  el.hidden = !message;
}

/** Show a spinner + message next to the buttons while `fn` runs; disable the buttons. */
async function withBusy(container, buttons, message, fn) {
  const msg = container.querySelector("[data-busy-msg]");
  buttons.forEach((b) => { b.disabled = true; b.setAttribute("aria-busy", "true"); });
  if (msg) msg.innerHTML = message ? `<span class="spinner" aria-hidden="true"></span>${escapeHtml(message)}` : "";
  let keepBusy = false;
  try {
    keepBusy = (await fn()) === "keep-busy";    // e.g. the page is about to reload
  } finally {
    if (!keepBusy) {
      buttons.forEach((b) => { b.disabled = false; b.removeAttribute("aria-busy"); });
      if (msg) msg.textContent = "";
    }
  }
}

function escapeHtml(text) {
  const d = document.createElement("div");
  d.textContent = text == null ? "" : String(text);
  return d.innerHTML;
}

function fmtMarks(v) {
  if (v === null || v === undefined) return "–";
  return Number.isInteger(v) ? String(v) : String(Math.round(v * 10) / 10);
}

function plural(n, word) { return `${n} ${word}${n === 1 ? "" : "s"}`; }

// Reload but come back to the same place on the page
function reloadKeepingScroll() {
  try { sessionStorage.setItem("scroll:" + location.pathname, String(window.scrollY)); } catch (_) { /* ignore */ }
  window.location.reload();
}
(function restoreScroll() {
  try {
    const key = "scroll:" + location.pathname;
    const y = sessionStorage.getItem(key);
    if (y !== null && !location.hash) {
      sessionStorage.removeItem(key);
      window.scrollTo(0, Number(y));
    }
  } catch (_) { /* ignore */ }
})();

// ---------- exams page: create exam ----------
const newExamForm = document.getElementById("new-exam-form");
if (newExamForm) {
  newExamForm.addEventListener("submit", (e) => {
    e.preventDefault();
    const btn = newExamForm.querySelector("button[type=submit]");
    withBusy(newExamForm, [btn], "Reading the schema…", async () => {
      showError(newExamForm, "");
      try {
        const exam = await api("POST", "/api/exams", new FormData(newExamForm));
        window.location.href = `/exams/${exam.id}`;
        return "keep-busy";
      } catch (err) {
        showError(newExamForm, err.message);
      }
    });
  });
}

// ---------- exam page ----------
const uploadForm = document.getElementById("upload-form");
const register = document.querySelector("[data-register]");

if (uploadForm) {
  const section = uploadForm.closest("section");
  const input = uploadForm.querySelector("input[type=file]");
  const zone = uploadForm.querySelector(".dropzone");
  const listBox = uploadForm.querySelector("[data-file-list]");
  const btn = uploadForm.querySelector("button[type=submit]");

  function showSelectedFiles() {
    const files = Array.from(input.files);
    listBox.hidden = files.length === 0;
    listBox.querySelector("[data-file-count]").textContent = `${plural(files.length, "file")} selected`;
    listBox.querySelector("ul").innerHTML = files.map((f) => `<li>${escapeHtml(f.name)}</li>`).join("");
  }
  input.addEventListener("change", showSelectedFiles);

  let depth = 0;  // dragenter/dragleave fire for child elements too
  zone.addEventListener("dragenter", () => { depth += 1; zone.classList.add("drag"); });
  zone.addEventListener("dragleave", () => { depth = Math.max(0, depth - 1); if (!depth) zone.classList.remove("drag"); });
  zone.addEventListener("drop", () => { depth = 0; zone.classList.remove("drag"); });

  uploadForm.addEventListener("submit", (e) => {
    e.preventDefault();
    const n = input.files.length;
    withBusy(section, [btn], `Uploading ${plural(n, "file")}…`, async () => {
      showError(section, "");
      try {
        const res = await api("POST", `/api/exams/${uploadForm.dataset.exam}/submissions`, new FormData(uploadForm));
        if (res.rejected.length) showError(section, "Skipped: " + res.rejected.join(" "));
        res.queued.forEach(addRow);
        updateProgress({ submissions: res.queued });   // show "Grading 1 of N" straight away
        uploadForm.reset();
        showSelectedFiles();
        startPolling();
      } catch (err) {
        showError(section, err.message);
      }
    });
  });
}

function rowCells(s) {
  const busy = BUSY.has(s.status);
  const total = document.querySelector("[data-stats]").dataset.total;
  const spinner = busy ? '<span class="spinner" aria-hidden="true"></span>' : "";
  let attention = "";
  if (s.status === "error") attention = `<span class="err">${escapeHtml(s.error || "Could not grade this script.")}</span>`;
  else if (s.n_flagged) attention = `<span class="chip chip-flag">${plural(s.n_flagged, "answer")} flagged</span>`;
  if (s.manual_marks) attention += `<span class="chip chip-manual">${fmtMarks(s.manual_marks)} marks to mark by hand</span>`;
  const final = s.total_final === null ? '<span class="muted">–</span>' : `<span class="ink">${fmtMarks(s.total_final)}</span>`;
  return {
    status: `<span class="chip chip-${s.status}">${spinner}${STATUS_TEXT[s.status] || s.status}</span>`,
    suggested: `<span class="pencil">${fmtMarks(s.total_suggested)}</span>`,
    final: `${final}<span class="of"> / ${fmtMarks(Number(s.total_max || total))}</span>`,
    attention: `<div class="attention">${attention}</div>`,
  };
}

function addRow(s) {
  if (!register || register.querySelector(`[data-row="${s.id}"]`)) return;
  const tr = document.createElement("tr");
  tr.dataset.row = s.id;
  tr.className = `status-${s.status}`;
  tr.innerHTML = `
    <td class="roll" data-label="Roll no.">${escapeHtml(s.roll_no || "–")}</td>
    <td class="student" data-label="Student"><span data-name>${escapeHtml(s.student_name || s.file_name)}</span>
      <span class="file">${escapeHtml(s.file_name)}</span></td>
    <td data-label="Status" data-cell="status"></td>
    <td class="num" data-label="Suggested" data-cell="suggested"></td>
    <td class="num" data-label="Final" data-cell="final"></td>
    <td data-label="Needs attention" data-cell="attention"></td>`;
  register.querySelector("tbody").appendChild(tr);
  updateRow(s);
  register.hidden = false;
  document.querySelector("[data-register-note]").hidden = false;
  document.querySelector("[data-empty]").hidden = true;
}

function updateRow(s) {
  const tr = register && register.querySelector(`[data-row="${s.id}"]`);
  if (!tr) return addRow(s);
  tr.className = `status-${s.status}`;
  const cells = rowCells(s);
  Object.entries(cells).forEach(([name, html]) => {
    const td = tr.querySelector(`[data-cell="${name}"]`);
    if (td && td.innerHTML !== html) td.innerHTML = html;
  });
  const roll = tr.querySelector(".roll");
  if (roll) roll.textContent = s.roll_no || "–";
  const name = tr.querySelector("[data-name]");
  if (name) name.textContent = s.student_name || s.file_name;
}

function updateStats(exam) {
  const subs = exam.submissions;
  const graded = subs.filter((s) => s.status === "evaluated" || s.status === "finalized");
  const sugg = graded.map((s) => s.total_suggested).filter((v) => v !== null);
  const values = {
    uploaded: subs.length,
    graded: graded.length,
    attention: subs.filter((s) => s.status === "error" || (s.status === "evaluated" && s.n_flagged > 0)).length,
    finalised: subs.filter((s) => s.status === "finalized").length,
    average: sugg.length ? fmtMarks(Math.round((sugg.reduce((a, b) => a + b, 0) / sugg.length) * 10) / 10) : "–",
  };
  Object.entries(values).forEach(([k, v]) => {
    const el = document.querySelector(`[data-stat="${k}"]`);
    if (el) el.textContent = v;
  });
  const att = document.querySelector(".stat-attention");
  if (att) att.classList.toggle("has-value", values.attention > 0);
}

// Progress of the current grading batch: every script we have seen waiting or grading
const batch = new Set();
function updateProgress(exam) {
  const box = document.querySelector("[data-progress]");
  if (!box) return 0;
  exam.submissions.forEach((s) => { if (BUSY.has(s.status)) batch.add(s.id); });
  const byId = new Map(exam.submissions.map((s) => [s.id, s]));
  const ids = [...batch].filter((id) => byId.has(id));
  const done = ids.filter((id) => !BUSY.has(byId.get(id).status)).length;
  const remaining = ids.length - done;
  box.hidden = ids.length === 0;
  if (remaining > 0) {
    box.querySelector("[data-progress-text]").textContent =
      `Grading ${Math.min(done + 1, ids.length)} of ${plural(ids.length, "script")}…`;
  } else if (ids.length) {
    box.querySelector("[data-progress-text]").textContent = `All ${plural(ids.length, "script")} graded.`;
  }
  box.querySelector("[data-progress-bar]").style.width = `${ids.length ? (100 * done) / ids.length : 0}%`;
  return remaining;
}

let pollTimer = null;
function startPolling() {
  const examId = uploadForm && uploadForm.dataset.exam;
  if (!examId || pollTimer) return;
  const tick = async () => {
    try {
      const exam = await api("GET", `/api/exams/${examId}`);
      exam.submissions.forEach(updateRow);
      updateStats(exam);
      if (updateProgress(exam) === 0) {
        pollTimer = null;
        setTimeout(reloadKeepingScroll, 600);   // reload once so names become links
        return;
      }
    } catch (err) {
      toast("Lost contact with the server, retrying…", true);
    }
    pollTimer = setTimeout(tick, POLL_MS);
  };
  pollTimer = setTimeout(tick, 400);
}

if (document.body.dataset.busy === "1") startPolling();

// Delete exam
document.querySelectorAll("[data-delete-exam]").forEach((btn) => {
  btn.addEventListener("click", async () => {
    if (!confirm("Delete this exam and all its scripts and marks? This cannot be undone.")) return;
    try {
      await api("DELETE", `/api/exams/${btn.dataset.deleteExam}`);
      window.location.href = "/";
    } catch (err) { toast(err.message, true); }
  });
});

// ---------- review page ----------
const head = document.querySelector("[data-sub]");
if (head) {
  const subId = head.dataset.sub;
  const bar = document.querySelector(".review-bar");
  const actionButtons = Array.from(document.querySelectorAll("[data-action]"));

  // Keep anchor jumps clear of the sticky bar + question navigator
  const sticky = document.querySelector("[data-sticky]");
  if (sticky && "ResizeObserver" in window) {
    new ResizeObserver(() => {
      const isSticky = getComputedStyle(sticky).position === "sticky";
      document.documentElement.style.setProperty("--sticky-h", isSticky ? `${sticky.offsetHeight + 12}px` : "1rem");
    }).observe(sticky);
  }

  function actionMessage(btn) {
    if (btn.dataset.action === "feedback") {
      const n = Number(btn.dataset.count || 0);
      return n ? `Writing feedback for ${plural(n, "answer")}… this takes a few seconds per answer.`
               : "Checking which answers need feedback…";
    }
    return btn.dataset.msg || "Working…";
  }

  async function waitUntilGraded() {
    for (;;) {
      await new Promise((r) => setTimeout(r, 2000));
      try {
        const sub = await api("GET", `/api/submissions/${subId}`);
        if (!BUSY.has(sub.status)) return;
      } catch (_) { /* keep waiting */ }
    }
  }

  actionButtons.forEach((btn) => {
    btn.addEventListener("click", () => withBusy(bar, actionButtons, actionMessage(btn), async () => {
      const action = btn.dataset.action;
      showError(bar, "");
      try {
        const res = await api("POST", `/api/submissions/${subId}/${action}`);
        if (action === "feedback" && res.failed.length) {
          showError(bar, `Feedback written for ${plural(res.written, "answer")}. Could not write: ${res.failed.join(", ")}.`);
          return;
        }
        if (action === "reevaluate") await waitUntilGraded();
        reloadKeepingScroll();
        return "keep-busy";
      } catch (err) {
        showError(bar, err.message);
      }
    }));
  });

  // Per-answer mark forms
  document.querySelectorAll(".mark-form").forEach((form) => {
    const input = form.querySelector("input[name=final_marks]");
    const comment = form.querySelector("[name=teacher_comment]");
    const saved = form.querySelector("[data-saved]");
    const saveBtn = form.querySelector("button[type=submit]");
    const useBtn = form.querySelector("[data-use-suggestion]");

    function showSaved() {
      if (!saved) return;
      saved.textContent = "Saved";
      saved.classList.remove("fade"); void saved.offsetWidth; saved.classList.add("show", "fade");
      clearTimeout(saved._timer);
      saved._timer = setTimeout(() => { saved.classList.remove("show", "fade"); saved.textContent = ""; }, 2200);
    }

    async function save() {
      const value = input.value.trim();
      if (saveBtn) saveBtn.disabled = true;
      try {
        const res = await api("PATCH", `/api/answers/${form.dataset.answer}`, {
          final_marks: value === "" ? null : Number(value),
          teacher_comment: comment.value,
        });
        const has = res.final_marks !== null;
        const text = has ? fmtMarks(res.final_marks) : "";
        const mark = document.querySelector(`[data-ink="${form.dataset.answer}"]`);
        mark.textContent = text;
        mark.classList.toggle("empty", !has);
        mark.classList.remove("pop"); void mark.offsetWidth; mark.classList.add("pop");
        const chip = document.querySelector(`[data-nav="${form.dataset.answer}"]`);
        if (chip) {
          chip.classList.toggle("is-saved", has);
          chip.querySelector("[data-nav-mark]").textContent = text;
        }
        showSaved();
        updateTotals();
      } catch (err) {
        if (saved) { saved.textContent = ""; saved.classList.remove("show", "fade"); }
        toast(err.message, true);
      } finally {
        if (saveBtn) saveBtn.disabled = false;
      }
    }

    form.addEventListener("submit", (e) => { e.preventDefault(); save(); });
    if (useBtn) useBtn.addEventListener("click", () => { input.value = fmtMarks(Number(form.dataset.suggested)); save(); });
  });

  function updateTotals() {
    let total = 0, reviewed = 0;
    document.querySelectorAll(".answer:not(.not-counted) [data-ink]").forEach((el) => {
      if (el.textContent.trim() !== "") { total += Number(el.textContent); reviewed += 1; }
    });
    const t = document.querySelector("[data-final-total]");
    if (t) t.textContent = fmtMarks(total);
    const r = document.querySelector("[data-reviewed]");
    if (r) r.textContent = reviewed;
  }
}
