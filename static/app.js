const imageInput = document.querySelector("#image-input");
const dropzone = document.querySelector("#dropzone");
const uploadEmpty = document.querySelector("#upload-empty");
const previewWrap = document.querySelector("#preview-wrap");
const imagePreview = document.querySelector("#image-preview");
const modelList = document.querySelector("#model-list");
const analyzeButton = document.querySelector("#analyze-button");
const resultPlaceholder = document.querySelector("#result-placeholder");
const resultsContent = document.querySelector("#results-content");
const resultCards = document.querySelector("#result-cards");
const resultCount = document.querySelector("#result-count");
const summaryMessage = document.querySelector("#summary-message");
const errorMessage = document.querySelector("#error-message");

let selectedFile = null;
let previewUrl = null;

function showError(message) {
  errorMessage.textContent = message;
  errorMessage.classList.remove("hidden");
}

function clearError() {
  errorMessage.textContent = "";
  errorMessage.classList.add("hidden");
}

function updateAnalyzeButton() {
  const hasSelection = modelList.querySelectorAll("input[name=model]:checked").length > 0;
  analyzeButton.disabled = !selectedFile || !hasSelection;
}

function setImage(file) {
  clearError();
  if (!file) return;
  if (!file.type.startsWith("image/")) {
    showError("Choose a supported image file.");
    return;
  }
  if (file.size > 15 * 1024 * 1024) {
    showError("The image must be 15 MB or smaller.");
    return;
  }
  selectedFile = file;
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = URL.createObjectURL(file);
  imagePreview.src = previewUrl;
  uploadEmpty.classList.add("hidden");
  previewWrap.classList.remove("hidden");
  updateAnalyzeButton();
}

function createModelOption(model) {
  const row = document.createElement("div");
  row.className = "model-option";
  const label = document.createElement("label");
  const checkbox = document.createElement("input");
  checkbox.type = "checkbox";
  checkbox.name = "model";
  checkbox.value = model.key;
  checkbox.checked = model.available;
  checkbox.disabled = !model.available;
  checkbox.addEventListener("change", updateAnalyzeButton);
  const name = document.createElement("span");
  name.textContent = model.name;
  label.append(checkbox, name);

  const state = document.createElement("span");
  state.className = `model-status${model.available ? " ready" : ""}`;
  state.textContent = model.available ? "MODEL READY" : "MODEL MISSING";
  row.append(label, state);
  return row;
}

async function loadModels() {
  modelList.innerHTML = '<div class="loading-models">Checking for trained models…</div>';
  try {
    const response = await fetch("/api/models");
    if (!response.ok) {
      if (response.status === 404) {
        throw new Error(
          "The Vercel page is deployed, but its prediction backend is not connected yet."
        );
      }
      throw new Error("Could not check model availability.");
    }
    const data = await response.json();
    modelList.replaceChildren(...data.models.map(createModelOption));
    if (!data.models.some((model) => model.available)) {
      showError("Xception model not found. Add your trained Xception model, then refresh.");
    } else {
      clearError();
    }
    updateAnalyzeButton();
  } catch (error) {
    modelList.replaceChildren();
    showError(error.message || "Could not connect to the local prediction server.");
  }
}

function addResultCard(result) {
  const card = document.createElement("article");
  card.className = "result-card";
  const top = document.createElement("div");
  top.className = "card-top";
  const model = document.createElement("span");
  model.className = "card-model";
  model.textContent = result.model;
  const label = document.createElement("strong");
  label.className = `card-label ${String(result.label).toLowerCase()}`;
  label.textContent = result.label;
  top.append(model, label);
  card.append(top);

  if (result.status === "Ready") {
    const score = Number(result.predicted_probability) * 100;
    const scoreRow = document.createElement("div");
    scoreRow.className = "score-row";
    const scoreText = document.createElement("span");
    scoreText.textContent = "Predicted-label score";
    const scoreValue = document.createElement("strong");
    scoreValue.textContent = `${score.toFixed(1)}%`;
    scoreRow.append(scoreText, scoreValue);
    const track = document.createElement("div");
    track.className = "score-track";
    const fill = document.createElement("div");
    fill.className = `score-fill${result.label === "Fake" ? " fake" : ""}`;
    fill.style.width = `${Math.max(0, Math.min(100, score))}%`;
    track.append(fill);
    const fakeRow = document.createElement("div");
    fakeRow.className = "score-row";
    fakeRow.innerHTML = `<span>Fake score</span><strong>${Number(result.fake_probability).toFixed(4)}</strong>`;
    card.append(scoreRow, track, fakeRow);
  }
  const status = document.createElement("div");
  status.className = "card-status";
  status.textContent = result.status;
  card.append(status);
  return card;
}

async function analyze() {
  clearError();
  if (!selectedFile) return;
  const selected = [...modelList.querySelectorAll("input[name=model]:checked")]
    .map((input) => input.value);
  if (!selected.length) {
    showError("Select the Xception model.");
    return;
  }

  const formData = new FormData();
  formData.append("image", selectedFile);
  formData.append("selected_models", JSON.stringify(selected));
  analyzeButton.disabled = true;
  analyzeButton.classList.add("loading");
  analyzeButton.querySelector("span:first-child").textContent = "Analyzing…";
  resultCount.textContent = "RUNNING";
  try {
    const response = await fetch("/api/predict", { method: "POST", body: formData });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "Prediction failed.");
    const cards = data.results.map(addResultCard);
    resultCards.replaceChildren(...cards);
    summaryMessage.textContent = data.message;
    resultPlaceholder.classList.add("hidden");
    resultsContent.classList.remove("hidden");
    resultCount.textContent = `${data.loaded_count} READY`;
  } catch (error) {
    showError(error.message || "Could not complete local inference.");
    resultCount.textContent = "ERROR";
  } finally {
    analyzeButton.classList.remove("loading");
    analyzeButton.querySelector("span:first-child").textContent = "Analyze image";
    updateAnalyzeButton();
  }
}

imageInput.addEventListener("change", (event) => setImage(event.target.files[0]));
dropzone.addEventListener("dragover", (event) => {
  event.preventDefault();
  dropzone.classList.add("drag-over");
});
dropzone.addEventListener("dragleave", () => dropzone.classList.remove("drag-over"));
dropzone.addEventListener("drop", (event) => {
  event.preventDefault();
  dropzone.classList.remove("drag-over");
  setImage(event.dataTransfer.files[0]);
});
analyzeButton.addEventListener("click", analyze);
document.querySelector("#refresh-models").addEventListener("click", loadModels);
loadModels();
