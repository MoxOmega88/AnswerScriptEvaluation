// Small helpers for the teacher pages. All actions go through the /api endpoints.

async function api(method, url, body) {
  const opts = { method };
  if (body instanceof FormData) {
    opts.body = body;
  } else if (body !== undefined) {
    opts.headers = { "Content-Type": "application/json" };
    opts.body = JSON.stringify(body);
  }
  const res = await fetch(url, opts);
  let data = null;
  try { data = await res.json(); } catch (_) { /* not JSON */ }
  if (!res.ok) {
    let msg = data && data.detail ? data.detail : `Request failed (${res.status})`;
    if (Array.isArray(msg)) msg = msg.map((d) => d.msg).join("; ");   // FastAPI validation errors
    throw new Error(msg);
  }
  return data;
}

function toast(text, isError = false) {
  const el = document.getElementById("toast");
  el.textContent = text;
  el.classList.toggle("error", isError);
  el.classList.add("show");
  clearTimeout(el._timer);
  el._timer = setTimeout(() => el.classList.remove("show"), 2600);
}

function showError(container, message) {
  const el = container.querySelector("[data-error]");
  if (!el) return toast(message, true);
  el.textContent = message;
  el.hidden = !message;
}

async function withBusy(button, fn) {
  if (button) button.disabled = true;
  try { await fn(); } finally { if (button) button.disabled = false; }
}

// ---------- exams page: create exam ----------
const newExamForm = document.getElementById("new-exam-form");
if (newExamForm) {
  newExamForm.addEventListener("submit", (e) => {
    e.preventDefault();
    const btn = newExamForm.querySelector("button[type=submit]");
    withBusy(btn, async () => {
      showError(newExamForm, "");
      try {
        const exam = await api("POST", "/api/exams", new FormData(newExamForm));
        window.location.href = `/exams/${exam.id}`;
      } catch (err) {
        showError(newExamForm, err.message);
      }
    });
  });
}

// ---------- exam page: upload scripts ----------
const uploadForm = document.getElementById("upload-form");
if (uploadForm) {
  const input = uploadForm.querySelector("input[type=file]");
  const list = uploadForm.querySelector("[data-file-list]");
  const zone = uploadForm.querySelector(".dropzone");

  input.addEventListener("change", () => {
    const n = input.files.length;
    list.textContent = n ? `${n} file${n === 1 ? "" : "s"} selected` : "";
  });
  ["dragenter", "dragover"].forEach((ev) => zone.addEventListener(ev, () => zone.classList.add("drag")));
  ["dragleave", "drop"].forEach((ev) => zone.addEventListener(ev, () => zone.classList.remove("drag")));

  uploadForm.addEventListener("submit", (e) => {
    e.preventDefault();
    const btn = uploadForm.querySelector("button[type=submit]");
    const section = uploadForm.closest("section");
    withBusy(btn, async () => {
      showError(section, "");
      try {
        const res = await api("POST", `/api/exams/${uploadForm.dataset.exam}/submissions`, new FormData(uploadForm));
        if (res.rejected.length) {
          showError(section, "Skipped: " + res.rejected.join(" "));
          setTimeout(() => window.location.reload(), 2500);
        } else {
          window.location.reload();
        }
      } catch (err) {
        showError(section, err.message);
      }
    });
  });
}

// Refresh the exam page while scripts are being graded
if (document.body.dataset.busy === "1") {
  setTimeout(() => window.location.reload(), 3000);
}

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

  // Script-level actions
  document.querySelectorAll("[data-action]").forEach((btn) => {
    btn.addEventListener("click", () => withBusy(btn, async () => {
      const action = btn.dataset.action;
      showError(bar, "");
      try {
        await api("POST", `/api/submissions/${subId}/${action}`);
        if (action === "reevaluate") {
          toast("Grading again...");
          setTimeout(() => (window.location.href = document.querySelector(".crumbs a").href), 800);
        } else {
          window.location.reload();
        }
      } catch (err) {
        showError(bar, err.message);
      }
    }));
  });

  // Per-answer mark forms
  document.querySelectorAll(".mark-form").forEach((form) => {
    const input = form.querySelector("input[name=final_marks]");
    const comment = form.querySelector("input[name=teacher_comment]");
    const saved = form.querySelector("[data-saved]");
    const useBtn = form.querySelector("[data-use-suggestion]");

    async function save() {
      const value = input.value.trim();
      try {
        const res = await api("PATCH", `/api/answers/${form.dataset.answer}`, {
          final_marks: value === "" ? null : Number(value),
          teacher_comment: comment.value,
        });
        const mark = document.querySelector(`[data-ink="${form.dataset.answer}"]`);
        mark.textContent = res.final_marks === null ? "" : String(res.final_marks);
        mark.classList.toggle("empty", res.final_marks === null);
        mark.classList.remove("pop"); void mark.offsetWidth; mark.classList.add("pop");
        if (saved) saved.textContent = "Saved";
        updateTotals();
      } catch (err) {
        if (saved) saved.textContent = "";
        toast(err.message, true);
      }
    }

    form.addEventListener("submit", (e) => { e.preventDefault(); save(); });
    if (useBtn) useBtn.addEventListener("click", () => { input.value = form.dataset.suggested; save(); });
    input.addEventListener("input", () => { if (saved) saved.textContent = ""; });
  });

  function updateTotals() {
    let total = 0, reviewed = 0;
    document.querySelectorAll(".answer:not(.not-counted) [data-ink]").forEach((el) => {
      if (el.textContent.trim() !== "") { total += Number(el.textContent); reviewed += 1; }
    });
    const t = document.querySelector("[data-final-total]");
    if (t) t.textContent = Number.isInteger(total) ? total : total.toFixed(1);
    const r = document.querySelector("[data-reviewed]");
    if (r) r.textContent = reviewed;
  }
}