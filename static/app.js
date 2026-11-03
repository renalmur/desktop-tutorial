// Облачный контроль — работа страницы.
// Здесь нет нейросети: страница только отправляет снимки на сервер (app.py) и показывает ответ.
"use strict";

const $ = (id) => document.getElementById(id);

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;   // textContent: имя файла не станет кодом страницы
  return node;
}

async function api(path, options) {
  let response;
  try {
    response = await fetch(path, options);
  } catch (e) {
    throw new Error("Сервер не отвечает. Проверьте, что программа app.py запущена.");
  }
  if (!response.ok) {
    let detail = "Ошибка сервера (" + response.status + ")";
    try {
      const data = await response.json();
      if (typeof data.detail === "string") detail = data.detail;
    } catch (e) { /* ответ не в формате JSON */ }
    throw new Error(detail);
  }
  return response.json();
}

function percentText(value) {
  return value.toFixed(1).replace(".", ",") + " %";
}

// ============================================================ вкладки
const tabs = { process: $("tab-process"), label: $("tab-label") };
const panels = { process: $("panel-process"), label: $("panel-label") };

function showTab(name) {
  Object.keys(tabs).forEach((key) => {
    const active = key === name;
    tabs[key].classList.toggle("is-active", active);
    tabs[key].setAttribute("aria-selected", String(active));
    panels[key].hidden = !active;
  });
  if (name === "label") loadImageList();
}
tabs.process.addEventListener("click", () => showTab("process"));
tabs.label.addEventListener("click", () => showTab("label"));

// ============================================================ состояние модели и папки
async function refreshStatus() {
  try {
    const s = await api("/api/status");
    const note = $("model-note");
    if (!s.model_ready) {
      note.textContent = "Модель не найдена. Обучите сеть: python train.py";
      note.classList.add("is-bad");
    } else {
      const m = s.model;
      const when = m.trained_at ? m.trained_at.replace("T", " ").slice(0, 16) : "—";
      const real = m.real_pixels ? "дообучена на настоящих снимках (" + m.real_images.length + " шт.)" : "базовая (только учебные снимки)";
      note.textContent = "Нейросеть: " + real + ", обучена " + when;
      note.classList.remove("is-bad");
    }
    const waiting = s.incoming_waiting.length;
    $("folder-box").hidden = waiting === 0;
    if (waiting) {
      $("folder-text").textContent = "В папке data/incoming уже лежат снимки: " + waiting + " шт. Их можно обработать без загрузки.";
    }
  } catch (e) {
    $("model-note").textContent = e.message;
    $("model-note").classList.add("is-bad");
  }
}

// ============================================================ обработка снимков
let picked = [];

function setPicked(fileList) {
  picked = Array.from(fileList);
  const box = $("picked");
  if (!picked.length) {
    box.textContent = "Файлы не выбраны";
  } else {
    const names = picked.slice(0, 3).map((f) => f.name).join(", ");
    box.textContent = "Выбрано: " + picked.length + " — " + names + (picked.length > 3 ? " и ещё " + (picked.length - 3) : "");
  }
  $("run").disabled = picked.length === 0;
}

$("files").addEventListener("change", (e) => setPicked(e.target.files));
const drop = $("drop");
["dragenter", "dragover"].forEach((type) => drop.addEventListener(type, (e) => { e.preventDefault(); drop.classList.add("is-over"); }));
["dragleave", "drop"].forEach((type) => drop.addEventListener(type, (e) => { e.preventDefault(); drop.classList.remove("is-over"); }));
drop.addEventListener("drop", (e) => setPicked(e.dataTransfer.files));

$("threshold").addEventListener("input", (e) => { $("threshold-value").textContent = e.target.value + " %"; });

function deleteMode() {
  return document.querySelector('input[name="delete-mode"]:checked').value;
}

function setStatus(text, isError) {
  const box = $("status");
  box.textContent = text;
  box.classList.toggle("is-error", Boolean(isError));
}

function setBusy(busy) {
  $("run").disabled = busy || picked.length === 0;
  $("run-folder").disabled = busy;
  $("run").textContent = busy ? "Обрабатываю…" : "Обработать снимки";
}

function summaryCell(value, caption, className) {
  const cell = el("div", className);
  cell.appendChild(el("b", "", String(value)));
  cell.appendChild(el("span", "", caption));
  return cell;
}

function renderReport(data) {
  $("empty").hidden = true;
  $("report").hidden = false;

  const summary = $("summary");
  summary.replaceChildren(
    summaryCell(data.counts.total, "обработано"),
    summaryCell(data.counts.saved, "сохранено", "is-saved"),
    summaryCell(data.counts.deleted, "удалено", "is-deleted"),
    summaryCell(data.counts.errors, "не прочитано")
  );
  $("message").textContent = data.message;
  $("download-txt").href = "/api/report/" + data.run_id + "?format=txt";
  $("download-csv").href = "/api/report/" + data.run_id + "?format=csv";

  const rows = $("rows");
  rows.replaceChildren();
  data.results.forEach((r) => {
    const row = el("li", "row");

    const thumbs = el("div", "thumbs");
    if (r.preview) {
      const photo = el("img"); photo.src = "/previews/" + data.run_id + "/" + r.preview; photo.alt = "Снимок " + r.name; photo.loading = "lazy";
      const clouds = el("img"); clouds.src = "/previews/" + data.run_id + "/" + r.overlay; clouds.alt = "Облака на снимке закрашены"; clouds.loading = "lazy";
      // Клик по миниатюре открывает снимок в полном размере.
      photo.addEventListener("click", () => openViewer(photo.src, r.name));
      clouds.addEventListener("click", () => openViewer(clouds.src, r.name + " — облака закрашены"));
      thumbs.append(photo, clouds);
    }
    row.appendChild(thumbs);

    const middle = el("div");
    middle.appendChild(el("div", "row-name", r.name));
    if (r.status === "error") {
      middle.appendChild(el("div", "row-note", r.error));
    } else {
      const meter = el("div", "meter");
      meter.setAttribute("role", "img");
      meter.setAttribute("aria-label", "Облачность " + percentText(r.cloud_percent) + ", порог " + data.threshold + " %");
      const fill = el("div", "meter-fill"); fill.style.width = Math.max(0, Math.min(100, r.cloud_percent)) + "%";
      const mark = el("div", "meter-mark"); mark.style.left = data.threshold + "%";
      meter.append(fill, mark);
      middle.appendChild(meter);
      const text = el("div", "meter-text");
      const left = el("span"); left.append("облачность ", el("b", "", percentText(r.cloud_percent)));
      text.append(left, el("span", "", "порог " + data.threshold + " %"));
      middle.appendChild(text);
      if (r.saved_as && r.saved_as !== r.name) middle.appendChild(el("div", "row-note", "Сохранён под именем " + r.saved_as));
    }
    row.appendChild(middle);

    const labels = { saved: "Сохранён", deleted: r.moved_to_trash ? "В корзине" : "Удалён", error: "Не прочитан" };
    row.appendChild(el("div", "verdict " + r.status, labels[r.status]));
    rows.appendChild(row);
  });
}

async function runProcessing(request) {
  setBusy(true);
  setStatus("Нейросеть смотрит снимки…");
  try {
    const data = await request();
    renderReport(data);
    setStatus(data.counts.total ? "Готово за " + data.seconds.toFixed(1).replace(".", ",") + " с." : "В папке не нашлось снимков.");
    $("files").value = "";
    setPicked([]);
  } catch (e) {
    setStatus(e.message, true);
  } finally {
    setBusy(false);
    refreshStatus();
  }
}

$("run").addEventListener("click", () => runProcessing(() => {
  const form = new FormData();
  picked.forEach((file) => form.append("files", file, file.name));
  form.append("threshold", $("threshold").value);
  form.append("delete_mode", deleteMode());
  return api("/api/process", { method: "POST", body: form });
}));

$("run-folder").addEventListener("click", () => runProcessing(() => api("/api/process-folder", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ threshold: Number($("threshold").value), delete_mode: deleteMode() }),
})));

$("copy-message").addEventListener("click", async () => {
  const button = $("copy-message");
  try {
    await navigator.clipboard.writeText($("message").textContent);
    button.textContent = "Скопировано";
  } catch (e) {
    // Буфер обмена недоступен — выделяем текст, чтобы его можно было скопировать клавишами Ctrl+C.
    const range = document.createRange();
    range.selectNodeContents($("message"));
    const selection = window.getSelection();
    selection.removeAllRanges();
    selection.addRange(range);
    button.textContent = "Текст выделен — Ctrl+C";
  }
  setTimeout(() => { button.textContent = "Скопировать"; }, 2500);
});

// ============================================================ разметка
const label = { name: null, tool: "cloud", drawing: false, last: null, dirty: false, labeled: new Set() };
const imageCanvas = $("canvas-image");
const labelCanvas = $("canvas-label");
const labelCtx = labelCanvas.getContext("2d");
// Цвета мазков на холсте. Сервер различает их именно так: красный — облако, синий — не облако.
const TOOL_COLORS = { cloud: "rgb(255,0,0)", clear: "rgb(0,0,255)" };

function setLabelStatus(text, isError) {
  const box = $("label-status");
  box.textContent = text;
  box.classList.toggle("is-error", Boolean(isError));
}

function loadPicture(src) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error("Не удалось показать картинку"));
    img.src = src;
  });
}

async function loadImageList() {
  const list = $("image-list");
  try {
    const data = await api("/api/label/images");
    list.replaceChildren();
    label.labeled = new Set(data.images.filter((item) => item.labeled).map((item) => item.name));
    if (!data.images.length) {
      list.appendChild(el("li", "none", "Снимков пока нет. Добавьте их кнопкой выше."));
      return;
    }
    data.images.forEach((item) => {
      const button = el("button", item.name === label.name ? "is-active" : "");
      button.appendChild(el("span", "", item.name));
      if (item.labeled) button.appendChild(el("span", "tag", "размечен"));
      button.addEventListener("click", () => openImage(item.name));
      const li = el("li"); li.appendChild(button); list.appendChild(li);
    });
  } catch (e) {
    list.replaceChildren(el("li", "none", e.message));
  }
}

async function openImage(name) {
  if (label.dirty && !window.confirm("Мазки на текущем снимке не сохранены. Перейти к другому снимку?")) return;
  setLabelStatus("Открываю снимок…");
  try {
    const picture = await loadPicture("/api/label/image/" + encodeURIComponent(name) + "?t=" + Date.now());
    imageCanvas.width = labelCanvas.width = picture.naturalWidth;
    imageCanvas.height = labelCanvas.height = picture.naturalHeight;
    imageCanvas.getContext("2d").drawImage(picture, 0, 0);
    labelCtx.clearRect(0, 0, labelCanvas.width, labelCanvas.height);
    labelCtx.globalCompositeOperation = "source-over";
    if (label.labeled.has(name)) {                      // если разметка уже была — показываем её
      const saved = await loadPicture("/api/label/mask/" + encodeURIComponent(name) + "?t=" + Date.now());
      labelCtx.drawImage(saved, 0, 0);
    }
    // Снимок целиком помещается на экран: высота холста — не больше 70 % высоты окна.
    const fitWidth = 0.7 * window.innerHeight * picture.naturalWidth / picture.naturalHeight;
    $("stage").style.maxWidth = Math.round(Math.max(260, Math.min(900, fitWidth))) + "px";
    label.name = name;
    label.dirty = false;
    $("label-empty").hidden = true;
    $("label-work").hidden = false;
    $("show-prediction").checked = false;
    $("prediction").hidden = true;
    setLabelStatus("Проведите кистью по облакам и по местам без облаков, потом сохраните разметку.");
    loadImageList();
  } catch (e) {
    setLabelStatus(e.message, true);
  }
}

document.querySelectorAll(".tool").forEach((button) => {
  button.addEventListener("click", () => {
    label.tool = button.dataset.tool;
    document.querySelectorAll(".tool").forEach((b) => {
      b.classList.toggle("is-active", b === button);
      b.setAttribute("aria-checked", String(b === button));
    });
  });
});

function canvasPoint(event) {
  const rect = labelCanvas.getBoundingClientRect();
  return {
    x: (event.clientX - rect.left) * labelCanvas.width / rect.width,
    y: (event.clientY - rect.top) * labelCanvas.height / rect.height,
  };
}

function stroke(from, to) {
  // Размер кисти задан в пикселях экрана — переводим в пиксели холста.
  const scale = labelCanvas.width / labelCanvas.getBoundingClientRect().width;
  labelCtx.lineWidth = Math.max(1.5, Number($("brush").value) * scale);
  labelCtx.lineCap = "round";
  labelCtx.lineJoin = "round";
  if (label.tool === "erase") {
    labelCtx.globalCompositeOperation = "destination-out";
    labelCtx.strokeStyle = "rgb(0,0,0)";
  } else {
    labelCtx.globalCompositeOperation = "source-over";
    labelCtx.strokeStyle = TOOL_COLORS[label.tool];
  }
  labelCtx.beginPath();
  labelCtx.moveTo(from.x, from.y);
  labelCtx.lineTo(to.x + 0.01, to.y + 0.01);            // крошечный сдвиг: одиночный щелчок тоже оставляет точку
  labelCtx.stroke();
  label.dirty = true;
}

labelCanvas.addEventListener("pointerdown", (e) => {
  if (!label.name) return;
  labelCanvas.setPointerCapture(e.pointerId);
  label.drawing = true;
  label.last = canvasPoint(e);
  stroke(label.last, label.last);
});
labelCanvas.addEventListener("pointermove", (e) => {
  if (!label.drawing) return;
  const point = canvasPoint(e);
  stroke(label.last, point);
  label.last = point;
});
["pointerup", "pointercancel"].forEach((type) => labelCanvas.addEventListener(type, () => { label.drawing = false; }));

$("show-prediction").addEventListener("change", async (e) => {
  const picture = $("prediction");
  if (!e.target.checked) { picture.hidden = true; return; }
  setLabelStatus("Спрашиваю сеть…");
  try {
    const data = await api("/api/label/prediction/" + encodeURIComponent(label.name));
    picture.src = data.overlay;
    picture.hidden = false;
    setLabelStatus("Сеть считает, что облачность здесь " + percentText(data.cloud_percent) + ". Малиновым закрашено то, что она приняла за облака. Поправьте её мазками там, где она ошиблась.");
  } catch (err) {
    e.target.checked = false;
    setLabelStatus(err.message, true);
  }
});

$("save-label").addEventListener("click", async () => {
  if (!label.name) return;
  try {
    const data = await api("/api/label/save", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name: label.name, image: labelCanvas.toDataURL("image/png") }),
    });
    label.dirty = false;
    const total = data.cloud_pixels + data.clear_pixels;
    setLabelStatus(total
      ? "Разметка сохранена: облако — " + data.cloud_pixels + " пикселей, не облако — " + data.clear_pixels + ". Теперь можно дообучить сеть."
      : "На холсте нет мазков — разметка для этого снимка удалена.");
    loadImageList();
  } catch (e) {
    setLabelStatus(e.message, true);
  }
});

$("real-files").addEventListener("change", async (e) => {
  if (!e.target.files.length) return;
  const form = new FormData();
  Array.from(e.target.files).forEach((file) => form.append("files", file, file.name));
  try {
    await api("/api/label/upload", { method: "POST", body: form });
    e.target.value = "";
    loadImageList();
  } catch (err) {
    window.alert(err.message);
  }
});

async function trainRequest(path, busyText, button, idleText) {
  const log = $("train-log");
  log.hidden = false;
  log.textContent = busyText;
  $("train").disabled = true;
  $("reset-model").disabled = true;
  button.textContent = busyText;
  try {
    const data = await api(path, { method: "POST" });
    const info = data.info;
    const lines = (data.log || []).concat([
      "",
      "Модель: " + (info.real_pixels ? "с вашей разметкой (снимков: " + info.real_images.length + ", пикселей: " + info.real_pixels + ")" : "базовая"),
      "Точность на отложенных пикселях: " + (info.pixel_accuracy_holdout * 100).toFixed(1).replace(".", ",") + " %",
      "Проверьте результат: включите «Показать, что видит сеть» или запустите python evaluate.py",
    ]);
    log.textContent = lines.join("\n");
    if ($("show-prediction").checked) $("show-prediction").dispatchEvent(new Event("change"));
  } catch (e) {
    log.textContent = e.message;
  } finally {
    $("train").disabled = false;
    $("reset-model").disabled = false;
    button.textContent = idleText;
    refreshStatus();
  }
}

$("train").addEventListener("click", () => trainRequest("/api/train", "Обучаю сеть… одна-две минуты", $("train"), "Дообучить сеть"));
$("reset-model").addEventListener("click", () => {
  if (window.confirm("Вернуть базовую модель? Ваша разметка останется, но сеть забудет дообучение.")) {
    trainRequest("/api/model/reset", "Возвращаю базовую модель…", $("reset-model"), "Вернуть базовую модель");
  }
});

// ============================================================ просмотр снимка
const viewer = $("viewer");
const viewerImg = $("viewer-img");
const viewerStage = $("viewer-stage");
const zoomLabel = $("viewer-zoom-reset");

// Холст-обёртка: его размер = размер картинки × масштаб.
const viewerCanvas = document.createElement("div");
viewerCanvas.className = "viewer-canvas";
viewerStage.appendChild(viewerCanvas);
viewerCanvas.appendChild(viewerImg);

const zoom = { scale: 1, min: 0.05, max: 8 };

function applyZoom() {
  const w = viewerImg.naturalWidth || 800;
  const h = viewerImg.naturalHeight || 600;
  viewerCanvas.style.width = Math.round(w * zoom.scale) + "px";
  viewerCanvas.style.height = Math.round(h * zoom.scale) + "px";
  zoomLabel.textContent = Math.round(zoom.scale * 100) + " %";
}

function fitScale() {
  const pad = 32;
  const w = viewerImg.naturalWidth || 800;
  const h = viewerImg.naturalHeight || 600;
  const fitW = Math.max(1, viewerStage.clientWidth - pad) / w;
  const fitH = Math.max(1, viewerStage.clientHeight - pad) / h;
  return Math.max(zoom.min, Math.min(fitW, fitH));
}

function resetZoom() {
  zoom.scale = fitScale();
  applyZoom();
  viewerStage.scrollLeft = (viewerCanvas.offsetWidth - viewerStage.clientWidth) / 2;
  viewerStage.scrollTop = (viewerCanvas.offsetHeight - viewerStage.clientHeight) / 2;
}

function setZoom(next) {
  const clamped = Math.max(zoom.min, Math.min(zoom.max, next));
  if (clamped === zoom.scale) return;
  const centerX = viewerStage.scrollLeft + viewerStage.clientWidth / 2;
  const centerY = viewerStage.scrollTop + viewerStage.clientHeight / 2;
  const ratio = clamped / zoom.scale;
  zoom.scale = clamped;
  applyZoom();
  viewerStage.scrollLeft = centerX * ratio - viewerStage.clientWidth / 2;
  viewerStage.scrollTop = centerY * ratio - viewerStage.clientHeight / 2;
}

function openViewer(src, title) {
  $("viewer-name").textContent = title;
  viewerImg.onload = () => { resetZoom(); };
  viewerImg.src = src;
  viewer.hidden = false;
  document.body.style.overflow = "hidden";
}

function closeViewer() {
  viewer.hidden = true;
  viewerImg.src = "";
  document.body.style.overflow = "";
}

$("viewer-close").addEventListener("click", closeViewer);
$("viewer-zoom-in").addEventListener("click", () => setZoom(zoom.scale * 1.5));
$("viewer-zoom-out").addEventListener("click", () => setZoom(zoom.scale / 1.5));
$("viewer-zoom-reset").addEventListener("click", resetZoom);

window.addEventListener("resize", () => {
  if (!viewer.hidden && viewerImg.complete && viewerImg.naturalWidth) resetZoom();
});

viewerStage.addEventListener("wheel", (e) => {
  if (e.ctrlKey) {
    e.preventDefault();
    const factor = e.deltaY < 0 ? 1.15 : 1 / 1.15;
    setZoom(zoom.scale * factor);
  }
}, { passive: false });

viewerImg.addEventListener("dblclick", (e) => {
  e.preventDefault();
  if (zoom.scale > fitScale() * 1.05) resetZoom();
  else setZoom(3);
});

document.addEventListener("keydown", (e) => {
  if (viewer.hidden) return;
  if (e.key === "Escape") closeViewer();
  if (e.key === "+" || e.key === "=") { e.preventDefault(); setZoom(zoom.scale * 1.25); }
  if (e.key === "-" || e.key === "_") { e.preventDefault(); setZoom(zoom.scale / 1.25); }
  if (e.key === "0") { e.preventDefault(); resetZoom(); }
  const step = 80;
  if (e.key === "ArrowLeft") { e.preventDefault(); viewerStage.scrollLeft -= step; }
  if (e.key === "ArrowRight") { e.preventDefault(); viewerStage.scrollLeft += step; }
  if (e.key === "ArrowUp") { e.preventDefault(); viewerStage.scrollTop -= step; }
  if (e.key === "ArrowDown") { e.preventDefault(); viewerStage.scrollTop += step; }
});

window.addEventListener("beforeunload", (e) => {
  if (label.dirty) { e.preventDefault(); e.returnValue = ""; }
});

refreshStatus();