/* ============================================================================
   SENTIREEL — Movie Review Sentiment Analysis
   script.js  ·  Frontend logic: counters, cinematic loading, animated results
   ========================================================================== */

(function () {
  "use strict";

  // ── Element references ────────────────────────────────────────────────────
  const reviewInput   = document.getElementById("reviewInput");
  const charCount     = document.getElementById("charCount");
  const wordCount     = document.getElementById("wordCount");
  const analyzeBtn    = document.getElementById("analyzeBtn");
  const clearBtn      = document.getElementById("clearBtn");
  const sampleBtn     = document.getElementById("sampleBtn");

  const loadingPanel  = document.getElementById("loadingPanel");
  const loadingStage  = document.getElementById("loadingStage");
  const loadingBar    = document.getElementById("loadingBar");
  const errorPanel    = document.getElementById("errorPanel");
  const resultsPanel  = document.getElementById("resultsPanel");

  const verdictCard       = document.getElementById("verdictCard");
  const verdictText       = document.getElementById("verdictText");
  const verdictSub        = document.getElementById("verdictSub");
  const ringProgress      = document.getElementById("ringProgress");
  const ensembleConfEl    = document.getElementById("ensembleConfidence");
  const modelBars         = document.getElementById("modelBars");

  const statWords    = document.getElementById("statWords");
  const statChars    = document.getElementById("statChars");
  const statTime     = document.getElementById("statTime");
  const statPolarity = document.getElementById("statPolarity");

  // The fixed circumference of the SVG confidence ring (r = 52).
  const RING_CIRCUMFERENCE = 2 * Math.PI * 52;

  // Human-readable labels for each model key returned by the backend.
  const MODEL_LABELS = {
    textblob:            "TextBlob (rule-based)",
    naive_bayes:         "Naive Bayes",
    logistic_regression: "Logistic Regression",
    svm:                 "Support Vector Machine",
  };

  // Cinematic loading messages shown while the request is in flight.
  const LOADING_STAGES = [
    "Scanning movie review…",
    "Processing language with NLP…",
    "Vectorising with TF-IDF…",
    "Comparing ML models…",
    "Generating final prediction…",
  ];

  const SAMPLE_REVIEW =
    "An absolute triumph of modern cinema. The direction is bold and confident, " +
    "the cinematography is breathtaking, and the lead performance is the finest " +
    "of the year. I was gripped from the first frame to the last and left the " +
    "theatre genuinely moved.";

  // ── Character / word counters ─────────────────────────────────────────────
  function updateCounters() {
    const text = reviewInput.value;
    charCount.textContent = text.length;
    wordCount.textContent = text.trim() ? text.trim().split(/\s+/).length : 0;
  }
  reviewInput.addEventListener("input", updateCounters);

  // ── Helper: animate a number from 0 to a target ───────────────────────────
  function animateNumber(el, target, suffix = "", decimals = 0, duration = 1100) {
    const start = performance.now();
    function tick(now) {
      const progress = Math.min((now - start) / duration, 1);
      const eased = 1 - Math.pow(1 - progress, 3); // ease-out cubic
      const value = (target * eased).toFixed(decimals);
      el.textContent = value + suffix;
      if (progress < 1) requestAnimationFrame(tick);
      else el.textContent = target.toFixed(decimals) + suffix;
    }
    requestAnimationFrame(tick);
  }

  // ── Cinematic loading sequence ────────────────────────────────────────────
  let stageTimer = null;
  function startLoading() {
    errorPanel.hidden = true;
    resultsPanel.hidden = true;
    loadingPanel.hidden = false;
    analyzeBtn.disabled = true;

    let index = 0;
    const step = 100 / LOADING_STAGES.length;
    loadingStage.textContent = LOADING_STAGES[0];
    loadingBar.style.width = step + "%";

    stageTimer = setInterval(() => {
      index = Math.min(index + 1, LOADING_STAGES.length - 1);
      loadingStage.textContent = LOADING_STAGES[index];
      loadingBar.style.width = Math.min((index + 1) * step, 96) + "%";
    }, 520);
  }

  function stopLoading() {
    clearInterval(stageTimer);
    loadingBar.style.width = "100%";
    setTimeout(() => { loadingPanel.hidden = true; analyzeBtn.disabled = false; }, 250);
  }

  // ── Render the results dashboard ──────────────────────────────────────────
  function renderResults(data) {
    loadingPanel.hidden = true; // ensure the loader is gone before results paint
    const isPositive = data.final_prediction === "POSITIVE";

    // Final verdict card
    verdictCard.classList.remove("is-positive", "is-negative");
    verdictCard.classList.add(isPositive ? "is-positive" : "is-negative");
    verdictText.textContent = isPositive ? "POSITIVE REVIEW" : "NEGATIVE REVIEW";
    verdictSub.textContent =
      `${data.agree_count} of ${data.total_models} models agree on this sentiment.`;

    // Confidence ring
    ringProgress.style.strokeDasharray = RING_CIRCUMFERENCE;
    ringProgress.style.strokeDashoffset = RING_CIRCUMFERENCE;
    requestAnimationFrame(() => {
      const offset = RING_CIRCUMFERENCE * (1 - data.ensemble_confidence / 100);
      ringProgress.style.strokeDashoffset = offset;
    });
    animateNumber(ensembleConfEl, data.ensemble_confidence, "%", 0);

    // Model comparison bars
    modelBars.innerHTML = "";
    Object.keys(MODEL_LABELS).forEach((key) => {
      const model = data.models[key];
      const pos = model.prediction === "POSITIVE";

      const row = document.createElement("div");
      row.className = "model-row";
      row.innerHTML = `
        <div class="model-row-top">
          <span class="model-name">
            ${MODEL_LABELS[key]}
            <span class="model-tag ${pos ? "pos" : "neg"}">${model.prediction}</span>
          </span>
          <span class="model-value" data-target="${model.confidence}">0%</span>
        </div>
        <div class="bar-track">
          <div class="bar-fill ${pos ? "pos" : "neg"}" style="width:0%"></div>
        </div>`;
      modelBars.appendChild(row);

      // Animate the bar + its value after a tick so the transition fires.
      const fill = row.querySelector(".bar-fill");
      const valueEl = row.querySelector(".model-value");
      requestAnimationFrame(() => { fill.style.width = model.confidence + "%"; });
      animateNumber(valueEl, model.confidence, "%", 1);
    });

    // Additional stats
    animateNumber(statWords, data.word_count, "", 0, 900);
    animateNumber(statChars, data.char_count, "", 0, 900);
    statTime.textContent = data.processing_time_ms + " ms";
    statPolarity.textContent = data.polarity_score.toFixed(2);

    // Reveal
    resultsPanel.hidden = false;
    resultsPanel.classList.remove("reveal");
    void resultsPanel.offsetWidth; // restart animation
    resultsPanel.classList.add("reveal");
    resultsPanel.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  function showError(message) {
    errorPanel.textContent = message;
    errorPanel.hidden = false;
  }

  // ── Main analyse flow ─────────────────────────────────────────────────────
  async function analyse() {
    const review = reviewInput.value.trim();
    if (review.length < 5) {
      showError("Please enter a movie review of at least 5 characters.");
      return;
    }

    startLoading();
    const requestStart = performance.now();

    try {
      const response = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ review }),
      });

      const data = await response.json();
      if (!response.ok) throw new Error(data.error || "Prediction failed.");

      // Guarantee the cinematic loader is visible for a beat (feels intentional).
      const elapsed = performance.now() - requestStart;
      const minDelay = 1600;
      if (elapsed < minDelay) await new Promise((r) => setTimeout(r, minDelay - elapsed));

      stopLoading();
      renderResults(data);
    } catch (err) {
      stopLoading();
      showError(err.message || "Something went wrong. Is the Flask server running?");
    }
  }

  // ── Event wiring ──────────────────────────────────────────────────────────
  analyzeBtn.addEventListener("click", analyse);
  clearBtn.addEventListener("click", () => {
    reviewInput.value = "";
    updateCounters();
    resultsPanel.hidden = true;
    errorPanel.hidden = true;
    reviewInput.focus();
  });
  sampleBtn.addEventListener("click", () => {
    reviewInput.value = SAMPLE_REVIEW;
    updateCounters();
    reviewInput.focus();
  });

  // Ctrl/Cmd + Enter to analyse
  reviewInput.addEventListener("keydown", (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key === "Enter") analyse();
  });

  updateCounters();
})();
