(function () {
  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("file");
  const fileName = document.getElementById("file-name");
  const submitBtn = document.getElementById("submit");
  const lengthEl = document.getElementById("length");
  const focusEl = document.getElementById("focus");
  const statusEl = document.getElementById("status");
  const errorEl = document.getElementById("error");

  const apiCustom = document.getElementById("api-custom");
  const apiFields = document.getElementById("api-fields");
  const openaiBaseUrl = document.getElementById("openai-base-url");
  const openaiApiKey = document.getElementById("openai-api-key");
  const openaiTextModel = document.getElementById("openai-text-model");
  const openaiVisionModel = document.getElementById("openai-vision-model");

  /** @type {File | null} */
  let selected = null;

  function showError(msg) {
    errorEl.textContent = msg || "";
    errorEl.hidden = !msg;
  }

  function setStatus(msg) {
    statusEl.textContent = msg || "";
  }

  const lengthField = lengthEl && lengthEl.closest(".field");
  const focusField = focusEl && focusEl.closest(".field");

  function setLengthFocusLocked(locked) {
    if (lengthEl && focusEl) {
      lengthEl.disabled = locked;
      focusEl.disabled = locked;
    }
    if (lengthField) lengthField.classList.toggle("field--locked", locked);
    if (focusField) focusField.classList.toggle("field--locked", locked);
  }

  function applyDefaultLengthFocus() {
    if (lengthEl) lengthEl.value = "medium";
    if (focusEl) focusEl.value = "method";
  }

  function syncCustomApiUi() {
    const on = apiCustom && apiCustom.checked;
    if (apiFields) apiFields.hidden = !on;
    if (on) {
      setLengthFocusLocked(false);
    } else {
      applyDefaultLengthFocus();
      setLengthFocusLocked(true);
    }
  }

  if (apiCustom) {
    apiCustom.addEventListener("change", syncCustomApiUi);
    syncCustomApiUi();
  }

  function pickOnePdf(fileList) {
    const pdf = Array.from(fileList).find(
      (f) => f.type === "application/pdf" || /\.pdf$/i.test(f.name)
    );
    return pdf || null;
  }

  function setFile(file) {
    selected = file;
    fileName.textContent = file ? file.name : "";
    submitBtn.disabled = !file;
    showError("");
  }

  dropzone.addEventListener("click", () => fileInput.click());

  dropzone.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      fileInput.click();
    }
  });

  fileInput.addEventListener("change", () => {
    const f = fileInput.files && fileInput.files[0];
    if (f) setFile(f);
  });

  ["dragenter", "dragover", "dragleave", "drop"].forEach((ev) => {
    dropzone.addEventListener(ev, (e) => {
      e.preventDefault();
      e.stopPropagation();
    });
  });

  dropzone.addEventListener("dragover", () => {
    dropzone.classList.add("dragover");
  });

  dropzone.addEventListener("dragleave", () => {
    dropzone.classList.remove("dragover");
  });

  dropzone.addEventListener("drop", (e) => {
    dropzone.classList.remove("dragover");
    const file = pickOnePdf(e.dataTransfer.files);
    if (!file) {
      showError("请提供 PDF 格式的单篇文献。");
      return;
    }
    setFile(file);
    try {
      const dt = new DataTransfer();
      dt.items.add(file);
      fileInput.files = dt.files;
    } catch (_) {
      /* 部分浏览器不支持 DataTransfer 赋值，仅内存态仍可上传 */
    }
  });

  submitBtn.addEventListener("click", async () => {
    if (!selected) return;
    showError("");

    if (apiCustom && apiCustom.checked) {
      const bu = openaiBaseUrl.value.trim();
      const key = openaiApiKey.value.trim();
      const tm = openaiTextModel.value.trim();
      const vm = openaiVisionModel.value.trim();
      if (!bu || !key || !tm || !vm) {
        showError("已开启自定义 API：请填写 Base URL、API Key、文本模型与视觉模型。");
        return;
      }
    }

    setStatus("正在解析版面并撰写解读稿，耗时取决于页数与模型负载，请稍候…");
    submitBtn.disabled = true;

    const fd = new FormData();
    fd.append("file", selected, selected.name);
    fd.append("length", lengthEl.value);
    fd.append("focus", focusEl.value);

    if (apiCustom && apiCustom.checked) {
      fd.append("api_custom", "true");
      fd.append("openai_base_url", openaiBaseUrl.value.trim());
      fd.append("openai_api_key", openaiApiKey.value.trim());
      fd.append("openai_text_model", openaiTextModel.value.trim());
      fd.append("openai_vision_model", openaiVisionModel.value.trim());
    }

    try {
      const res = await fetch("/api/build", {
        method: "POST",
        body: fd,
      });

      if (!res.ok) {
        let detail = "请求未能完成";
        try {
          const j = await res.json();
          if (j && j.detail) detail = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail);
        } catch (_) {
          detail = res.statusText || detail;
        }
        throw new Error(detail);
      }

      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "paper_readable.pdf";
      a.rel = "noopener";
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
      setStatus("解读 PDF 已生成；若未自动保存，请查看浏览器下载提示或下载文件夹。");
    } catch (err) {
      showError(err instanceof Error ? err.message : String(err));
      setStatus("");
    } finally {
      submitBtn.disabled = !selected;
    }
  });
})();
