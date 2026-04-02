const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

let ws = null;
let simInfo = null;
let currentFrame = null;

// Zoom/pan state
let zoom = 1;
let panX = 0;
let panY = 0;
let isPanning = false;
let lastPanX = 0;
let lastPanY = 0;

function connect() {
    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    ws = new WebSocket(`${proto}//${location.host}/ws`);
    ws.binaryType = "arraybuffer";

    ws.onopen = () => {
        document.getElementById("status-bar").textContent = "connected";
    };

    ws.onclose = () => {
        document.getElementById("status-bar").textContent = "disconnected";
        setTimeout(connect, 2000);
    };

    ws.onmessage = (event) => {
        if (event.data instanceof ArrayBuffer) {
            renderFrame(event.data);
        } else {
            const msg = JSON.parse(event.data);
            handleMessage(msg);
        }
    };
}

function renderFrame(buffer) {
    const blob = new Blob([buffer], { type: "image/jpeg" });
    createImageBitmap(blob).then((bmp) => {
        currentFrame = bmp;
        drawFrame();
    });
}

function drawFrame() {
    if (!currentFrame) return;

    // Canvas fills the container
    const wrap = canvas.parentElement;
    canvas.width = wrap.clientWidth;
    canvas.height = wrap.clientHeight;

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.save();

    // Apply zoom and pan
    ctx.translate(canvas.width / 2 + panX, canvas.height / 2 + panY);
    ctx.scale(zoom, zoom);

    // Center the image
    const drawW = currentFrame.width;
    const drawH = currentFrame.height;
    ctx.drawImage(currentFrame, -drawW / 2, -drawH / 2, drawW, drawH);

    ctx.restore();
}

function handleMessage(msg) {
    if (msg.type === "sim_info") {
        simInfo = msg.data;
        updateSimUI(msg.data);
        // Reset zoom/pan on sim switch
        zoom = 1;
        panX = 0;
        panY = 0;
    } else if (msg.type === "metrics") {
        updateMetrics(msg.data);
    } else if (msg.type === "status") {
        if (msg.data.paused) {
            document.getElementById("btn-play").disabled = false;
            document.getElementById("btn-pause").disabled = true;
        } else {
            document.getElementById("btn-play").disabled = true;
            document.getElementById("btn-pause").disabled = false;
        }
    }
}

function updateSimUI(info) {
    if (info.pixelated) {
        canvas.classList.add("pixelated");
    } else {
        canvas.classList.remove("pixelated");
    }

    const container = document.getElementById("params-container");
    container.innerHTML = "";
    for (const [key, param] of Object.entries(info.params || {})) {
        const row = document.createElement("div");
        row.className = "param-row";

        const label = document.createElement("div");
        label.className = "param-label";
        label.innerHTML = `<span>${key}</span><span class="val">${param.default}</span>`;

        const slider = document.createElement("input");
        slider.type = "range";
        slider.min = param.min;
        slider.max = param.max;
        slider.step = param.step;
        slider.value = param.default;

        slider.addEventListener("input", () => {
            label.querySelector(".val").textContent = parseFloat(slider.value).toFixed(4);
            sendParams();
        });

        row.appendChild(label);
        row.appendChild(slider);
        container.appendChild(row);
    }

    const presetsSection = document.getElementById("presets-section");
    const presetSelect = document.getElementById("preset-select");
    if (info.presets && Object.keys(info.presets).length > 0) {
        presetsSection.style.display = "block";
        presetSelect.innerHTML = '<option value="">--</option>';
        for (const name of Object.keys(info.presets)) {
            const opt = document.createElement("option");
            opt.value = name;
            opt.textContent = name;
            presetSelect.appendChild(opt);
        }
    } else {
        presetsSection.style.display = "none";
    }
}

function updateMetrics(data) {
    document.getElementById("m-step").textContent = (data.step_ms || 0).toFixed(1);
    document.getElementById("m-halo").textContent = (data.halo_ms || 0).toFixed(1);
    document.getElementById("m-steps").textContent = data.step_count || 0;
}

function send(msg) {
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify(msg));
    }
}

function sendParams() {
    const params = {};
    document.querySelectorAll("#params-container .param-row").forEach((row) => {
        const key = row.querySelector(".param-label span").textContent;
        const val = parseFloat(row.querySelector("input").value);
        params[key] = val;
    });
    send({ type: "set_params", params });
}

// Convert canvas pixel coordinates to grid coordinates, accounting for zoom/pan
function canvasToGrid(clientX, clientY) {
    const rect = canvas.getBoundingClientRect();
    const cx = clientX - rect.left;
    const cy = clientY - rect.top;

    if (!currentFrame) return { row: 0, col: 0 };

    // Reverse the transform: canvas center + pan, then scale
    const gx = (cx - canvas.width / 2 - panX) / zoom + currentFrame.width / 2;
    const gy = (cy - canvas.height / 2 - panY) / zoom + currentFrame.height / 2;

    return { row: Math.floor(gy), col: Math.floor(gx) };
}

// Buttons
document.getElementById("btn-play").addEventListener("click", () => {
    send({ type: "play" });
    document.getElementById("btn-play").disabled = true;
    document.getElementById("btn-pause").disabled = false;
});

document.getElementById("btn-pause").addEventListener("click", () => {
    send({ type: "pause" });
    document.getElementById("btn-play").disabled = false;
    document.getElementById("btn-pause").disabled = true;
});

document.getElementById("btn-reset").addEventListener("click", () => {
    send({ type: "reset" });
});

// Simulation selector
fetch("/api/simulations").then(r => r.json()).then((sims) => {
    const select = document.getElementById("sim-select");
    select.innerHTML = "";
    for (const sim of sims) {
        const opt = document.createElement("option");
        opt.value = sim.name;
        opt.textContent = sim.description || sim.name;
        select.appendChild(opt);
    }
});

document.getElementById("sim-select").addEventListener("change", (e) => {
    send({ type: "switch_sim", name: e.target.value });
});

// Presets
document.getElementById("preset-select").addEventListener("change", (e) => {
    if (!simInfo || !e.target.value) return;
    const preset = simInfo.presets[e.target.value];
    if (!preset) return;

    document.querySelectorAll("#params-container .param-row").forEach((row) => {
        const key = row.querySelector(".param-label span").textContent;
        if (key in preset) {
            const slider = row.querySelector("input");
            slider.value = preset[key];
            row.querySelector(".val").textContent = preset[key];
        }
    });
    sendParams();
});

// Worker slider
const workerSlider = document.getElementById("worker-slider");
const workerCount = document.getElementById("worker-count");
workerSlider.addEventListener("change", () => {
    const count = parseInt(workerSlider.value);
    workerCount.textContent = count;
    send({ type: "set_workers", count });
});
workerSlider.addEventListener("input", () => {
    workerCount.textContent = workerSlider.value;
});

// Zoom with scroll wheel
canvas.addEventListener("wheel", (e) => {
    e.preventDefault();
    const zoomFactor = e.deltaY > 0 ? 0.9 : 1.1;
    zoom *= zoomFactor;
    zoom = Math.max(0.1, Math.min(50, zoom));
    drawFrame();
}, { passive: false });

// Pan with middle mouse or shift+click drag
canvas.addEventListener("mousedown", (e) => {
    if (e.button === 1 || (e.button === 0 && e.shiftKey)) {
        // Middle click or shift+left click: start panning
        isPanning = true;
        lastPanX = e.clientX;
        lastPanY = e.clientY;
        e.preventDefault();
    } else if (e.button === 0 && !e.shiftKey) {
        // Left click: perturb
        const { row, col } = canvasToGrid(e.clientX, e.clientY);
        send({ type: "perturb", row, col, channel: 0, value: 1.0, radius: 3 });
    } else if (e.button === 2) {
        // Right click: erase
        const { row, col } = canvasToGrid(e.clientX, e.clientY);
        send({ type: "perturb", row, col, channel: 0, value: 0.0, radius: 3 });
    }
});

canvas.addEventListener("mousemove", (e) => {
    if (isPanning) {
        panX += e.clientX - lastPanX;
        panY += e.clientY - lastPanY;
        lastPanX = e.clientX;
        lastPanY = e.clientY;
        drawFrame();
    }
});

canvas.addEventListener("mouseup", (e) => {
    if (e.button === 1 || (e.button === 0 && e.shiftKey)) {
        isPanning = false;
    }
});

canvas.addEventListener("mouseleave", () => {
    isPanning = false;
});

canvas.addEventListener("contextmenu", (e) => e.preventDefault());

// Redraw on window resize
window.addEventListener("resize", drawFrame);

connect();
