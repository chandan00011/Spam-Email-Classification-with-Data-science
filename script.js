/* script.js - connects the page to POST /predict in app.py */

document.addEventListener("DOMContentLoaded", function () {
  var emailText = document.getElementById("emailText");
  var analyzeBtn = document.getElementById("analyzeBtn");
  var resetBtn = document.getElementById("resetBtn");
  var charCounter = document.getElementById("charCounter");
  var errorBox = document.getElementById("errorBox");

  var resultCard = document.getElementById("resultCard");
  var verdictLabel = document.getElementById("verdictLabel");
  var verdictNote = document.getElementById("verdictNote");
  var confidenceValue = document.getElementById("confidenceValue");
  var whySection = document.getElementById("whySection");
  var reasonsList = document.getElementById("reasonsList");

  var statWords = document.getElementById("statWords");
  var statChars = document.getElementById("statChars");
  var statLinks = document.getElementById("statLinks");
  var statExcl = document.getElementById("statExcl");

  function showError(message) {
    errorBox.textContent = message;
    errorBox.hidden = false;
  }

  function clearError() {
    errorBox.textContent = "";
    errorBox.hidden = true;
  }

  emailText.addEventListener("input", function () {
    charCounter.textContent = emailText.value.length + " characters";
    if (emailText.value.trim()) {
      clearError();
    }
  });

  // Ctrl + Enter also submits.
  emailText.addEventListener("keydown", function (event) {
    if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
      analyze();
    }
  });

  analyzeBtn.addEventListener("click", analyze);

  resetBtn.addEventListener("click", function () {
    emailText.value = "";
    charCounter.textContent = "0 characters";
    resultCard.hidden = true;
    resultCard.classList.remove("is-spam", "is-ham");
    clearError();
    emailText.focus();
  });

  function analyze() {
    var text = emailText.value.trim();

    if (!text) {
      showError("Enter some email text first.");
      emailText.focus();
      return;
    }

    clearError();
    analyzeBtn.disabled = true;
    analyzeBtn.textContent = "Analyzing…";

    fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email_text: text })
    })
      .then(function (response) {
        return response.json().then(function (data) {
          return { ok: response.ok, data: data };
        });
      })
      .then(function (payload) {
        if (!payload.ok || !payload.data.success) {
          resultCard.hidden = true;
          showError(payload.data.error || "The server could not analyze this email.");
          return;
        }
        render(payload.data);
      })
      .catch(function () {
        resultCard.hidden = true;
        showError("Could not reach the server. Check that app.py is still running.");
      })
      .then(function () {
        analyzeBtn.disabled = false;
        analyzeBtn.textContent = "Analyze email";
      });
  }

  function render(data) {
    resultCard.classList.remove("is-spam", "is-ham");

    if (data.is_spam) {
      resultCard.classList.add("is-spam");
      verdictLabel.textContent = "SPAM DETECTED";
      verdictNote.textContent = "Do not click links or share personal details from this email.";
    } else {
      resultCard.classList.add("is-ham");
      verdictLabel.textContent = "HAM";
      verdictNote.textContent = "This email appears to be legitimate.";
    }

    confidenceValue.textContent = data.confidence.toFixed(1) + "%";

    reasonsList.innerHTML = "";
    if (data.reasons && data.reasons.length) {
      data.reasons.forEach(function (reason) {
        var li = document.createElement("li");
        li.textContent = reason;
        reasonsList.appendChild(li);
      });
      whySection.hidden = false;
    } else {
      whySection.hidden = true;
    }

    statWords.textContent = data.stats.words;
    statChars.textContent = data.stats.characters;
    statLinks.textContent = data.stats.links;
    statExcl.textContent = data.stats.exclamations;

    resultCard.hidden = false;
    resultCard.scrollIntoView({ behavior: "smooth", block: "start" });
  }
});
