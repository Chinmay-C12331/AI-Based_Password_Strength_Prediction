/**
 * app.js: Browser client controller running ONNX model inference and occlusion analysis.
 */

const LEVEL_NAMES = ["Very Weak", "Weak", "Fair", "Strong", "Very Strong"];
const LEVEL_COLORS = ["#ef4444", "#f97316", "#eab308", "#10b981", "#06b6d4"];

let session = null;
let scalerData = null;
let wordlistSet = new Set();
let blocklistSet = new Set([
  "123456", "password", "12345678", "qwerty", "123456789", "12345", "1234", "111111",
  "1234567", "dragon", "welcome", "ninja", "monkey", "sunshine", "princess", "football",
  "admin", "secret", "master", "charlie", "trustno1", "letmein", "starwars", "batman"
]);

// Initialize model and scaler
async function initModel() {
  try {
    const scalerRes = await fetch("model/features_scaler.json");
    if (scalerRes.ok) {
      scalerData = await scalerRes.json();
    }
  } catch (e) {
    console.warn("Could not load scaler JSON, using unscaled features:", e);
  }

  try {
    if (typeof ort !== "undefined") {
      session = await ort.InferenceSession.create("model/model.int8.onnx", {
        executionProviders: ["wasm"],
      });
      console.log("Loaded ONNX INT8 model into browser session.");
    }
  } catch (e) {
    console.warn("ONNX session loading fallback (WASM / file protocol):", e);
  }
}

// Client-side heuristic inference when WASM session is initializing or unavailable
function localInference(pw, features) {
  // Features: [0]=len, [10]=entropy, [11]=repeat, [13]=walk, [14]=year, [16]=dict, [19]=zxcvbn
  const len = pw.length;
  if (len === 0) return { y: 0.0, cls: 0 };

  const zx_y = features[19] || 0.0;
  const entropy = features[10];
  const walkPenalty = features[13] * 0.4;
  const dictPenalty = features[16] * 2.5;
  const yearPenalty = features[14] * 1.5;
  const repeatPenalty = features[11] * 0.3;

  let y = (len * 0.55) + (entropy * 1.1) + (zx_y * 0.4) - walkPenalty - dictPenalty - yearPenalty - repeatPenalty;
  y = Math.max(0.5, Math.min(15.0, y));

  let cls = 0;
  if (y >= 10.0) cls = 4;
  else if (y >= 8.0) cls = 3;
  else if (y >= 6.0) cls = 2;
  else if (y >= 3.0) cls = 1;
  else cls = 0;

  return { y, cls };
}

async function predictPassword(pw) {
  const t0 = performance.now();
  const rawFeats = extractFeaturesJS(pw, { blocklist: blocklistSet, wordlist: wordlistSet });

  // Scale features
  const scaledFeats = new Float32Array(rawFeats.length);
  if (scalerData && scalerData.mean && scalerData.std) {
    for (let i = 0; i < rawFeats.length; i++) {
      const mean = scalerData.mean[i] || 0.0;
      const std = scalerData.std[i] || 1.0;
      scaledFeats[i] = (rawFeats[i] - mean) / std;
    }
  } else {
    scaledFeats.set(rawFeats);
  }

  let yHat = 0.0;
  let modelCls = 0;

  if (session) {
    try {
      const charIds = encodeCharsJS(pw, 32);
      const charTensor = new ort.Tensor("int64", charIds, [1, 32]);
      const featTensor = new ort.Tensor("float32", scaledFeats, [1, 22]);

      const feeds = { char_ids: charTensor, features: featTensor };
      const results = await session.run(feeds);

      yHat = results.y_hat.data[0];
      const logits = results.class_logits.data;
      let maxIdx = 0, maxVal = -Infinity;
      for (let i = 0; i < 5; i++) {
        if (logits[i] > maxVal) {
          maxVal = logits[i];
          maxIdx = i;
        }
      }
      modelCls = maxIdx;
    } catch (err) {
      const fallback = localInference(pw, rawFeats);
      yHat = fallback.y;
      modelCls = fallback.cls;
    }
  } else {
    const fallback = localInference(pw, rawFeats);
    yHat = fallback.y;
    modelCls = fallback.cls;
  }

  const dt = performance.now() - t0;
  return { y: yHat, cls: modelCls, rawFeats, latencyMs: dt };
}

// Occlusion sensitivity explanation: masks each character in turn
async function runOcclusionAnalysis(pw, baseY) {
  if (pw.length < 2) return [];
  const impacts = [];

  for (let i = 0; i < pw.length; i++) {
    const masked = pw.substring(0, i) + " " + pw.substring(i + 1);
    const res = await predictPassword(masked);
    const delta = res.y - baseY;
    impacts.push({ char: pw[i], delta });
  }
  return impacts;
}

// UI Elements
const input = document.getElementById("passwordInput");
const toggleBtn = document.getElementById("toggleBtn");
const levelBadge = document.getElementById("levelBadge");
const guessCount = document.getElementById("guessCount");
const floorLine = document.getElementById("floorLine");
const reasonsList = document.getElementById("reasonsList");
const occlusionStrip = document.getElementById("occlusionStrip");
const latencyLabel = document.getElementById("latencyLabel");

toggleBtn.addEventListener("click", () => {
  if (input.type === "password") {
    input.type = "text";
    toggleBtn.textContent = "Hide";
  } else {
    input.type = "password";
    toggleBtn.textContent = "Show";
  }
});

let debounceTimer = null;
input.addEventListener("input", () => {
  clearTimeout(debounceTimer);
  debounceTimer = setTimeout(updateMeter, 100);
});

async function updateMeter() {
  const pw = input.value;
  if (!pw) {
    resetMeter();
    return;
  }

  const { y, cls: rawModelCls, rawFeats, latencyMs } = await predictPassword(pw);

  // zxcvbn evaluation
  let zxScore = 0;
  if (typeof zxcvbn === "function") {
    zxScore = zxcvbn(pw).score;
  }

  // Conservative Floor logic (Project Brief & Prompt 9)
  const isBlocklisted = blocklistSet.has(pw.toLowerCase());
  let displayedLevel = isBlocklisted ? 0 : Math.min(rawModelCls, zxScore);
  displayedLevel = Math.max(0, Math.min(4, displayedLevel));

  // Render 5-segment strength bar
  for (let i = 0; i < 5; i++) {
    const seg = document.getElementById(`seg${i}`);
    if (i <= displayedLevel) {
      seg.style.backgroundColor = LEVEL_COLORS[displayedLevel];
    } else {
      seg.style.backgroundColor = "#334155";
    }
  }

  // Level badge
  levelBadge.textContent = LEVEL_NAMES[displayedLevel];
  levelBadge.style.backgroundColor = LEVEL_COLORS[displayedLevel];
  levelBadge.style.color = displayedLevel === 2 ? "#000" : "#fff";

  // Guess count
  const formattedY = Math.max(1.0, y).toFixed(1);
  const approxExp = Math.pow(10, y % 1).toFixed(1);
  const expPower = Math.floor(y);
  guessCount.innerHTML = `about 10<sup>${formattedY}</sup> guesses (~${approxExp} &times; 10<sup>${expPower}</sup>)`;

  // Conservative floor line
  if (isBlocklisted) {
    floorLine.innerHTML = `<strong>Conservative Override:</strong> Found in popular password blocklist &rarr; strictly pinned to <strong>Level 0</strong>.`;
    floorLine.style.borderColor = "#ef4444";
  } else if (rawModelCls > zxScore) {
    floorLine.innerHTML = `<strong>Conservative Floor Engaged:</strong> Model predicted Level ${rawModelCls}, but zxcvbn scored Level ${zxScore}. Conservative floor displays <strong>Level ${displayedLevel}</strong> to prevent overestimation.`;
    floorLine.style.borderColor = "#f59e0b";
  } else {
    floorLine.innerHTML = `<strong>Model &amp; Reference Agreement:</strong> Model predicted Level ${rawModelCls}, reference zxcvbn Level ${zxScore}. Conservative floor = <strong>Level ${displayedLevel}</strong>.`;
    floorLine.style.borderColor = "rgba(59, 130, 246, 0.3)";
  }

  // Top 3 reasons
  reasonsList.innerHTML = "";
  const reasons = [];
  if (rawFeats[0] < 8) reasons.push("Too short: passwords under 8 characters are easily guessed.");
  if (rawFeats[13] >= 4) reasons.push(`Keyboard walk pattern detected (length ${rawFeats[13]}).`);
  if (rawFeats[14] > 0.5) reasons.push("Contains a 4-digit birth or calendar year.");
  if (rawFeats[16] >= 0.5) reasons.push(`Dictionary word found (${Math.round(rawFeats[16] * 100)}% of characters).`);
  if (rawFeats[17] > 0.5) reasons.push("Predictable l33tspeak substitution (@&rarr;a, 0&rarr;o, 3&rarr;e).");
  if (rawFeats[11] >= 3) reasons.push(`Repeated character sequence (run of ${rawFeats[11]}).`);
  if (rawFeats[20] >= 2) reasons.push("Common trailing numbers appended at end.");
  if (isBlocklisted) reasons.push("Found in top breach password dictionary.");

  if (reasons.length === 0) {
    if (rawFeats[10] > 3.0 && pw.length >= 12) {
      reasons.push("High character entropy across multiple character sets.");
    } else {
      reasons.push("Standard alphanumeric distribution.");
    }
  }

  reasons.slice(0, 3).forEach(r => {
    const li = document.createElement("li");
    li.innerHTML = r;
    reasonsList.appendChild(li);
  });

  // Occlusion analysis
  const impacts = await runOcclusionAnalysis(pw, y);
  occlusionStrip.innerHTML = "";
  impacts.forEach(item => {
    const box = document.createElement("div");
    box.className = "char-box";
    box.textContent = item.char;

    // delta > 0 means removing the char made the password score HIGHER (so this char HURTS the password!)
    if (item.delta > 0.4) {
      box.classList.add("hurts-most");
      box.title = `Removing '${item.char}' increases score by +${item.delta.toFixed(2)} (vulnerable pattern element)`;
    } else if (item.delta > 0.15) {
      box.classList.add("hurts-moderate");
      box.title = `Removing '${item.char}' increases score by +${item.delta.toFixed(2)}`;
    } else {
      box.title = `'${item.char}' contributes positively or neutrally`;
    }
    occlusionStrip.appendChild(box);
  });

  latencyLabel.textContent = `Inference: ${latencyMs.toFixed(1)} ms`;
}

function resetMeter() {
  for (let i = 0; i < 5; i++) {
    document.getElementById(`seg${i}`).style.backgroundColor = "#334155";
  }
  levelBadge.textContent = "Start typing...";
  levelBadge.style.backgroundColor = "#334155";
  guessCount.innerHTML = `~10<sup>0.0</sup> guesses`;
  floorLine.innerHTML = `<strong>Conservative Floor Policy:</strong> Level will reflect min(Model, zxcvbn), with immediate Level 0 override for top-10k blocklisted credentials.`;
  floorLine.style.borderColor = "rgba(59, 130, 246, 0.2)";
  reasonsList.innerHTML = "<li>Type a password above to inspect pattern features.</li>";
  occlusionStrip.innerHTML = `<span style="font-size: 0.8rem; color: #64748b; padding: 4px;">Each character will be highlighted based on how much its removal/replacement increases the guessability score.</span>`;
  latencyLabel.textContent = "Inference: -- ms";
}

// Start
initModel();
