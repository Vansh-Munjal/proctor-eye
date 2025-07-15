let candidateId = "";
let trackingStarted = false;
let distractionStart = null;
let totalDistractionTime = 0;
let distractionEvents = [];

function startTracking() {
    candidateId = document.getElementById("candidateIdInput").value.trim();
    if (!candidateId) {
        alert("Please enter a Candidate ID before starting.");
        return;
    }

    document.getElementById("formContainer").style.display = "none";
    document.getElementById("mainApp").style.display = "block";

    startWebcam();
    trackingStarted = true;

    // Begin prediction loop after webcam starts
    startPredictionLoop();
}

function startWebcam() {
    const video = document.getElementById("webcam");
    navigator.mediaDevices.getUserMedia({ video: true }).then(stream => {
        video.srcObject = stream;
    });
}

function startPredictionLoop() {
    const video = document.getElementById("webcam");
    const predictionText = document.getElementById("prediction");

    setInterval(() => {
        if (video.videoWidth === 0 || video.videoHeight === 0) return;

        const canvas = document.createElement("canvas");
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        const ctx = canvas.getContext("2d");

        ctx.translate(canvas.width, 0);
        ctx.scale(-1, 1); // Flip webcam horizontally
        ctx.drawImage(video, 0, 0);

        const imgData = canvas.toDataURL("image/jpeg");

        fetch("/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ image: imgData })
        })
        .then(res => res.json())
        .then(data => {
            predictionText.textContent = `Gaze: ${data.gaze || 'Unknown'} | Head: ${data.head || 'Unknown'}`;

            const isDistracted = data.gaze !== "CENTER" || (data.head === "LEFT" || data.head === "RIGHT");
            const now = new Date();

            if (isDistracted) {
                if (!distractionStart) distractionStart = now;
            } else {
                if (distractionStart) {
                    const end = now;
                    const duration = Math.floor((end - distractionStart) / 1000); // seconds
                    if (duration >= 1) {
                        distractionEvents.push({
                            type: (data.gaze !== "CENTER" ? "Gaze Away" : "Head Turned"),
                            duration: duration,
                            start: distractionStart.toLocaleTimeString()
                        });
                        totalDistractionTime += duration;
                    }
                    distractionStart = null;
                }
            }

           if (data.alert) {
            const alertBox = document.getElementById("cheating-alert");
            alertBox.style.display = "block";

            // Hide the alert after 4 seconds
            setTimeout(() => {
                alertBox.style.display = "none";
            }, 4000);
        }

        })
        .catch(err => {
            console.error("Prediction error:", err);
        });

    }, 1000); // every second
}

function sendDistractionReport(candidateId, totalTime, events) {
    fetch('/log_distraction', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            candidate_id: candidateId,
            total_time: totalTime,
            events: events
        })
    })
    .then(res => res.json())
    .then(data => {
        console.log("✅ Report sent:", data);
    })
    .catch(err => {
        console.error("❌ Report error:", err);
    });
}

// ✅ Send report before window is closed or reloaded
window.addEventListener("beforeunload", function () {
    if (candidateId && trackingStarted) {
        sendDistractionReport(candidateId, totalDistractionTime, distractionEvents);
    }
});
