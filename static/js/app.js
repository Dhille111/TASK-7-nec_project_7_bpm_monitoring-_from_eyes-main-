// Elements
const video = document.getElementById('webcam');
const canvasOverlay = document.getElementById('canvas-overlay');
const ctxOverlay = canvasOverlay.getContext('2d');
const canvasLeftEye = document.getElementById('canvas-left-eye');
const ctxLeftEye = canvasLeftEye.getContext('2d');
const canvasRightEye = document.getElementById('canvas-right-eye');
const ctxRightEye = canvasRightEye.getContext('2d');

const canvasPPG = document.getElementById('canvas-ppg');
const ctxPPG = canvasPPG.getContext('2d');
const canvasRawSensor = document.getElementById('canvas-raw-sensor');
const ctxRawSensor = canvasRawSensor.getContext('2d');

const btnStart = document.getElementById('btn-start');
const btnStop = document.getElementById('btn-stop');
const btnSave = document.getElementById('btn-save');

const chkAudio = document.getElementById('chk-audio');
const inputBpm = document.getElementById('input-bpm');
const saveStatusMsg = document.getElementById('save-status-msg');
const trackingBadge = document.getElementById('tracking-badge');
const alarmBar = document.getElementById('alarm-bar');
const alarmStatusText = document.getElementById('alarm-status-text');
const alarmStatusIcon = document.getElementById('alarm-status-icon');
const liveTimeEl = document.getElementById('live-time');
const bufferFillBar = document.getElementById('buffer-fill-bar');
const bufferCountEl = document.getElementById('buffer-count');

const consensusBpm = document.getElementById('consensus-bpm');
const consensusConf = document.getElementById('consensus-conf');
const heartPulseIcon = document.getElementById('heart-pulse-icon');
const cardHr = document.getElementById('card-hr');
const fusedStatusText = document.getElementById('fused-status-text');

const cnnBpmEl = document.getElementById('cnn-bpm');
const bilstmBpmEl = document.getElementById('bilstm-bpm');
const dspBpmEl = document.getElementById('dsp-bpm');
const dspSnrEl = document.getElementById('dsp-snr');
const valSdnn = document.getElementById('val-sdnn');
const valRmssd = document.getElementById('val-rmssd');

// Web Audio API context for hospital monitor beeps
let audioCtx = null;

// State Variables
let isAcquiring = false;
let faceMesh = null;
const BUFFER_SIZE = 300;

// Sweep Buffers
const rawSignalBuffer = new Array(BUFFER_SIZE).fill(0.0);
const filteredSignalBuffer = new Array(BUFFER_SIZE).fill(0.0);
let writeIndex = 0; // Where the sweep bar writes next
let bufferFilledCount = 0;

let lastPredictTime = 0;

// Browser filter state (0.8Hz - 2.5Hz bandpass)
let x1 = 0, x2 = 0, y1 = 0, y2 = 0;

// Pulse beat detector states
let lastBeatTime = 0;
let isSignalHigh = false;
let runningMax = 1.0;
let runningMin = -1.0;

// Ticking Live Clock
function updateClock() {
    const now = new Date();
    const hrs = String(now.getHours()).padStart(2, '0');
    const mins = String(now.getMinutes()).padStart(2, '0');
    const secs = String(now.getSeconds()).padStart(2, '0');
    liveTimeEl.textContent = `${now.toLocaleDateString()} ${hrs}:${mins}:${secs}`;
}
setInterval(updateClock, 1000);
updateClock();

// Start web audio context
function initAudio() {
    if (!audioCtx) {
        audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    }
}

// Medical Pitch Beep Synthesizer
function playPulseBeep() {
    if (!chkAudio.checked || !audioCtx) return;
    try {
        const osc = audioCtx.createOscillator();
        const gainNode = audioCtx.createGain();
        osc.connect(gainNode);
        gainNode.connect(audioCtx.destination);
        
        osc.frequency.setValueAtTime(800, audioCtx.currentTime); // ICU standard 800Hz beep
        gainNode.gain.setValueAtTime(0.08, audioCtx.currentTime); // Volume
        gainNode.gain.exponentialRampToValueAtTime(0.00001, audioCtx.currentTime + 0.12); // Fast fade
        
        osc.start();
        osc.stop(audioCtx.currentTime + 0.15);
    } catch (e) {
        console.error("Audio beep failed", e);
    }
}

// Zero-Crossing Real-time Pulse Peak Detector (rises above threshold to trigger beeps)
function processBeatDetection(filteredVal) {
    const now = performance.now();
    
    // Slow decay for threshold limits to adapt to breathing/motion changes
    runningMax = runningMax * 0.992 + filteredVal * 0.008;
    runningMin = runningMin * 0.992 + filteredVal * 0.008;
    
    if (filteredVal > runningMax) runningMax = filteredVal;
    if (filteredVal < runningMin) runningMin = filteredVal;
    
    const range = runningMax - runningMin;
    const threshold = runningMin + range * 0.65; // Trigger at 65% of amplitude height
    
    if (filteredVal > threshold && !isSignalHigh) {
        isSignalHigh = true;
        // Enforce a physiological refractory period (e.g. min 350ms between beats -> max 170 BPM)
        if (now - lastBeatTime > 350) {
            lastBeatTime = now;
            triggerBeatIndicator();
        }
    } else if (filteredVal < threshold - range * 0.1) {
        // Hysteresis reset
        isSignalHigh = false;
    }
}

function triggerBeatIndicator() {
    playPulseBeep();
    
    // Flash green heart symbol
    heartPulseIcon.classList.add('pulse-active');
    setTimeout(() => {
        heartPulseIcon.classList.remove('pulse-active');
    }, 120);
}

// MediaPipe Left/Right Eye landmarks
const LEFT_EYE_LANDMARKS = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398];
const RIGHT_EYE_LANDMARKS = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246];

// Setup buttons
btnStart.addEventListener('click', async () => {
    if (isAcquiring) return;
    
    initAudio();
    isAcquiring = true;
    btnStart.disabled = true;
    btnStop.disabled = false;
    btnSave.disabled = true;
    
    // Clear buffers
    writeIndex = 0;
    bufferFilledCount = 0;
    rawSignalBuffer.fill(0.0);
    filteredSignalBuffer.fill(0.0);
    x1 = x2 = y1 = y2 = 0;
    
    trackingBadge.textContent = "LINKING...";
    trackingBadge.className = "tracking-indicator offline";
    fusedStatusText.textContent = "INITIALIZING SENSORS...";
    
    try {
        await startWebcam();
        fusedStatusText.textContent = "WEBCAM ACTIVE - LOADING DETECTORS...";
        initFaceMesh();
        fusedStatusText.textContent = "CALIBRATING RETINA SENSORS... STAY STILL";
        requestAnimationFrame(onFrame);
    } catch (err) {
        console.error("Camera access failed:", err);
        fusedStatusText.textContent = `CRITICAL ERROR: CAMERA LINK FAILED`;
        trackingBadge.textContent = "ERR: CAMERA";
        isAcquiring = false;
        btnStart.disabled = false;
        btnStop.disabled = true;
    }
});

btnStop.addEventListener('click', () => {
    stopAcquisition();
});

btnSave.addEventListener('click', async () => {
    const bpm = inputBpm.value.trim();
    if (!bpm) {
        showSaveStatus("ENTER BPM VALUE", "error");
        return;
    }
    
    if (bufferFilledCount < BUFFER_SIZE) {
        showSaveStatus("BUFFER INCOMPLETE", "error");
        return;
    }

    try {
        btnSave.disabled = true;
        const res = await fetch('/api/save', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                signal: rawSignalBuffer,
                bpm: parseInt(bpm)
            })
        });
        const data = await res.json();
        if (data.success) {
            showSaveStatus(`LOGGED: ${data.filename}`, "success");
            inputBpm.value = "";
        } else {
            showSaveStatus(data.error || "LOG FAIL", "error");
            btnSave.disabled = false;
        }
    } catch (e) {
        showSaveStatus("SERVER COMM ERROR", "error");
        btnSave.disabled = false;
    }
});

function showSaveStatus(msg, type) {
    saveStatusMsg.textContent = msg;
    saveStatusMsg.className = `save-status ${type === 'success' ? 'green-txt' : 'color-red'}`;
    setTimeout(() => {
        saveStatusMsg.textContent = "";
        saveStatusMsg.className = "save-status";
    }, 4000);
}

function stopAcquisition() {
    isAcquiring = false;
    btnStart.disabled = false;
    btnStop.disabled = true;
    btnSave.disabled = true;
    
    const stream = video.srcObject;
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
    }
    video.srcObject = null;
    
    trackingBadge.textContent = "OFFLINE";
    trackingBadge.className = "tracking-indicator offline";
    fusedStatusText.textContent = "MONITOR STANDBY";
    resetAlarmState();
}

async function startWebcam() {
    const constraints = {
        video: {
            width: { ideal: 640 },
            height: { ideal: 480 },
            frameRate: { ideal: 20 }
        },
        audio: false
    };

    const stream = await navigator.mediaDevices.getUserMedia(constraints);
    video.srcObject = stream;
    
    return new Promise((resolve) => {
        video.onloadedmetadata = () => {
            video.play();
            canvasOverlay.width = video.videoWidth;
            canvasOverlay.height = video.videoHeight;
            resolve();
        };
    });
}

async function onFrame() {
    if (!isAcquiring) return;
    try {
        if (video.readyState === video.HAVE_ENOUGH_DATA) {
            await faceMesh.send({ image: video });
        }
    } catch (err) {
        console.error("Face mesh frame send error:", err);
    }
    requestAnimationFrame(onFrame);
}

function initFaceMesh() {
    faceMesh = new FaceMesh({
        locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`
    });

    faceMesh.setOptions({
        maxNumFaces: 1,
        refineLandmarks: true,
        minDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5
    });

    faceMesh.onResults(onFaceMeshResults);
}

function getEyeBoundingBox(landmarks, imgWidth, imgHeight) {
    let minX = imgWidth, maxX = 0, minY = imgHeight, maxY = 0;
    
    landmarks.forEach(lm => {
        const x = lm.x * imgWidth;
        const y = lm.y * imgHeight;
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
    });

    const w = maxX - minX;
    const h = maxY - minY;
    const pad = Math.max(w, h) * 0.4;
    
    return {
        x: Math.max(0, minX - pad),
        y: Math.max(0, minY - pad),
        w: Math.min(imgWidth, w + pad * 2),
        h: Math.min(imgHeight, h + pad * 2)
    };
}

function calculateMeanGreen(ctx, width, height) {
    const imgData = ctx.getImageData(0, 0, width, height);
    const data = imgData.data;
    let sumGreen = 0;
    let pixelCount = 0;

    for (let i = 0; i < data.length; i += 4) {
        sumGreen += data[i + 1]; // Green channel
        pixelCount++;
    }

    return sumGreen / (pixelCount || 1);
}

// Basic real-time digital bandpass filter
function filterSample(sample) {
    const b = [0.1219, 0, -0.1219];
    const a = [1.0, -1.5034, 0.7562];
    
    const output = b[0]*sample - a[1]*y1 - a[2]*y2;
    
    x2 = x1;
    x1 = sample;
    y2 = y1;
    y1 = output;
    
    return output;
}

function onFaceMeshResults(results) {
    if (!isAcquiring) return;

    ctxOverlay.clearRect(0, 0, canvasOverlay.width, canvasOverlay.height);
    ctxOverlay.drawImage(results.image, 0, 0, canvasOverlay.width, canvasOverlay.height);

    if (results.multiFaceLandmarks && results.multiFaceLandmarks.length > 0) {
        trackingBadge.textContent = "ONLINE";
        trackingBadge.className = "tracking-indicator online";
        
        const faceLandmarks = results.multiFaceLandmarks[0];
        
        // Draw full face mesh landmarks as tiny dots
        ctxOverlay.fillStyle = 'rgba(0, 255, 0, 0.35)';
        faceLandmarks.forEach(lm => {
            const x = lm.x * canvasOverlay.width;
            const y = lm.y * canvasOverlay.height;
            ctxOverlay.fillRect(x, y, 1.5, 1.5);
        });

        // Draw left eye boundary path
        ctxOverlay.strokeStyle = 'rgba(0, 255, 255, 0.8)';
        ctxOverlay.lineWidth = 1.5;
        ctxOverlay.beginPath();
        LEFT_EYE_LANDMARKS.forEach((idx, i) => {
            const lm = faceLandmarks[idx];
            const x = lm.x * canvasOverlay.width;
            const y = lm.y * canvasOverlay.height;
            if (i === 0) ctxOverlay.moveTo(x, y);
            else ctxOverlay.lineTo(x, y);
        });
        ctxOverlay.closePath();
        ctxOverlay.stroke();

        // Draw right eye boundary path
        ctxOverlay.beginPath();
        RIGHT_EYE_LANDMARKS.forEach((idx, i) => {
            const lm = faceLandmarks[idx];
            const x = lm.x * canvasOverlay.width;
            const y = lm.y * canvasOverlay.height;
            if (i === 0) ctxOverlay.moveTo(x, y);
            else ctxOverlay.lineTo(x, y);
        });
        ctxOverlay.closePath();
        ctxOverlay.stroke();

        // Draw tracking box for eye regions
        const leftEyeBox = getEyeBoundingBox(LEFT_EYE_LANDMARKS.map(idx => faceLandmarks[idx]), canvasOverlay.width, canvasOverlay.height);
        const rightEyeBox = getEyeBoundingBox(RIGHT_EYE_LANDMARKS.map(idx => faceLandmarks[idx]), canvasOverlay.width, canvasOverlay.height);

        ctxOverlay.strokeStyle = 'rgba(0, 255, 0, 0.5)';
        ctxOverlay.lineWidth = 1;
        ctxOverlay.strokeRect(leftEyeBox.x, leftEyeBox.y, leftEyeBox.w, leftEyeBox.h);
        ctxOverlay.strokeRect(rightEyeBox.x, rightEyeBox.y, rightEyeBox.w, rightEyeBox.h);

        // Crop eye buffers
        canvasLeftEye.width = 64; canvasLeftEye.height = 64;
        ctxLeftEye.drawImage(results.image, leftEyeBox.x, leftEyeBox.y, leftEyeBox.w, leftEyeBox.h, 0, 0, 64, 64);
        
        canvasRightEye.width = 64; canvasRightEye.height = 64;
        ctxRightEye.drawImage(results.image, rightEyeBox.x, rightEyeBox.y, rightEyeBox.w, rightEyeBox.h, 0, 0, 64, 64);

        // Compute mean green channel
        const leftG = calculateMeanGreen(ctxLeftEye, 64, 64);
        const rightG = calculateMeanGreen(ctxRightEye, 64, 64);
        const meanG = (leftG + rightG) / 2.0;

        // 1. Write to buffers using Sweep index
        rawSignalBuffer[writeIndex] = meanG;
        const filteredVal = filterSample(meanG);
        filteredSignalBuffer[writeIndex] = filteredVal;

        // 2. Real-time Peak beat trigger
        processBeatDetection(filteredVal);

        // Increment write pointer
        writeIndex = (writeIndex + 1) % BUFFER_SIZE;
        
        if (bufferFilledCount < BUFFER_SIZE) {
            bufferFilledCount++;
            const fillPercent = (bufferFilledCount / BUFFER_SIZE) * 100;
            bufferFillBar.style.width = `${fillPercent}%`;
            bufferCountEl.textContent = bufferFilledCount;
        }

        // 3. Render Sweep Graph
        renderMedicalSweepCharts();

        // 4. Request Backend Inference
        if (bufferFilledCount === BUFFER_SIZE) {
            btnSave.disabled = false;
            const now = performance.now();
            if (now - lastPredictTime > 1000) {
                lastPredictTime = now;
                sendPredictRequest();
            }
        }
    } else {
        trackingBadge.textContent = "LOST FACE";
        trackingBadge.className = "tracking-indicator offline";
    }
}

// Hospital Patient Monitor Sweep Bar wave drawer
function drawSweepingWaveform(canvas, ctx, signalArray, strokeStyle, isNormalized) {
    canvas.width = canvas.clientWidth;
    canvas.height = canvas.clientHeight;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const step = canvas.width / (BUFFER_SIZE - 1);
    
    let minVal = Math.min(...signalArray);
    let maxVal = Math.max(...signalArray);
    let range = (maxVal - minVal) || 1.0;

    ctx.strokeStyle = strokeStyle;
    ctx.lineWidth = 2.0;
    
    // We draw the wave in two segments: before the write index, and after the erase bar gap.
    // Erase bar gap = 12 samples wide
    const gapSize = 12;
    const gapStart = writeIndex;
    const gapEnd = (writeIndex + gapSize) % BUFFER_SIZE;

    // Segment 1: From writeIndex+gapSize to BUFFER_SIZE
    ctx.beginPath();
    let isFirst = true;
    for (let i = 0; i < BUFFER_SIZE; i++) {
        // Skip the gap region
        const inGap = gapStart < gapEnd 
            ? (i >= gapStart && i < gapEnd)
            : (i >= gapStart || i < gapEnd);

        if (inGap) {
            isFirst = true; // reset path for after gap
            continue;
        }

        const x = i * step;
        const normY = 1.0 - (signalArray[i] - minVal) / range;
        const y = normY * (canvas.height - 20) + 10;

        if (isFirst) {
            ctx.moveTo(x, y);
            isFirst = false;
        } else {
            ctx.lineTo(x, y);
        }
    }
    ctx.stroke();

    // Draw sweep cursor dot
    const cursorX = writeIndex * step;
    ctx.fillStyle = strokeStyle;
    ctx.beginPath();
    ctx.arc(cursorX, canvas.height/2, 3, 0, 2*Math.PI);
    ctx.fill();
}

function renderMedicalSweepCharts() {
    // Green PLETH sweep chart
    drawSweepingWaveform(canvasPPG, ctxPPG, filteredSignalBuffer, '#00ff00', true);
    // Cyan Raw Sensor sweep chart
    drawSweepingWaveform(canvasRawSensor, ctxRawSensor, rawSignalBuffer, '#00ffff', false);
}

// Dynamic Vital Sign Oscillations (Simulates normal breathing/blood pressure drift)
function runVitalFluctuations() {
    if (!isAcquiring) return;
    
    // Systolic 117-123, Diastolic 77-83
    const sys = 117 + Math.floor(Math.random() * 7);
    const dia = 77 + Math.floor(Math.random() * 7);
    const mapVal = Math.round(dia + (sys - dia) / 3);
    
    document.querySelector('.param-cyan .param-value').innerHTML = `${sys}<span class="slash">/</span>${dia}`;
    document.querySelector('.param-cyan .param-source').textContent = `MAP (${mapVal})`;
    
    // Body Temp (36.5 - 36.8)
    const tempVal = (36.5 + Math.random() * 0.4).toFixed(1);
    document.querySelector('.param-white .models-column .model-row:last-child .model-val').textContent = `${tempVal} °C`;
}
setInterval(runVitalFluctuations, 7000);

async function sendPredictRequest() {
    try {
        const res = await fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ signal: rawSignalBuffer })
        });
        const data = await res.json();
        
        if (data.error) {
            fusedStatusText.textContent = `SYS ERR: ${data.error}`;
            return;
        }

        // 1. Update Numeric metrics
        const hrVal = Math.round(data.fused_bpm);
        consensusBpm.textContent = hrVal;
        consensusConf.textContent = `${Math.round(data.confidence_score * 100)}% (${data.confidence_label})`;
        
        cnnBpmEl.textContent = `${Math.round(data.cnn_bpm)} bpm`;
        bilstmBpmEl.textContent = `${Math.round(data.bilstm_bpm)} bpm`;
        dspBpmEl.textContent = `${Math.round(data.dsp_bpm)} bpm`;
        dspSnrEl.textContent = `${data.dsp_snr.toFixed(2)}`;

        valSdnn.textContent = `${Math.round(data.hrv.sdnn * 1000)} ms`;
        valRmssd.textContent = `${Math.round(data.hrv.rmssd * 1000)} ms`;

        // Update bottom right status message
        fusedStatusText.textContent = data.status_msg || `FUSION LOCKED: ${hrVal} BPM (${data.confidence_label})`;

        // 2. Alarm checking: limits are static
        checkAlarmLimits(hrVal);

    } catch (err) {
        console.error("Prediction request failed", err);
        fusedStatusText.textContent = "COMM TIMEOUT - BACKEND UNRESPONSIVE";
    }
}

// Alarm monitoring
function checkAlarmLimits(hr) {
    const highLimit = 120;
    const lowLimit = 50;
    
    // Update the high/low visual limits indicators in the panel
    document.querySelector('.param-limits .lim-val:first-child').textContent = highLimit;
    document.querySelector('.param-limits .lim-val:last-child').textContent = lowLimit;
    
    if (hr < lowLimit || hr > highLimit) {
        alarmBar.className = "alarm-banner banner-alarm";
        alarmStatusIcon.textContent = "🚨";
        alarmStatusText.textContent = `ALARM: HR OUT-OF-RANGE (${hr})`;
        
        // Flash HR card red border
        cardHr.style.borderColor = "#ff0000";
        cardHr.style.boxShadow = "0 0 15px rgba(255, 0, 0, 0.4)";
    } else {
        resetAlarmState();
    }
}

function resetAlarmState() {
    const highLimit = 120;
    const lowLimit = 50;
    
    alarmBar.className = "alarm-banner banner-normal";
    alarmStatusIcon.textContent = "✓";
    alarmStatusText.textContent = `LIMITS OK: HR ${lowLimit} - ${highLimit}`;
    
    cardHr.style.borderColor = "";
    cardHr.style.boxShadow = "";
}
