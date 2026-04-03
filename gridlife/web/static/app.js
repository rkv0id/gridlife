const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

let ws = null;
let simInfo = null;
let palette = null;
let gridWidth = 0;
let gridHeight = 0;
let imageData = null;

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
            handleBinary(new Uint8Array(event.data));
        } else {
            const msg = JSON.parse(event.data);
            handleMessage(msg);
        }
    };
}

function handleBinary(data) {
    const type = data[0];

    if (type === 0x01) {
        palette = data.slice(1);
    } else if (type === 0x02) {
        const view = new DataView(data.buffer, data.byteOffset);
        const w = view.getUint16(1);
        const h = view.getUint16(3);
        const pixels = data.slice(5);
        renderGrid(w, h, pixels);
    }
}

function renderGrid(w, h, pixels) {
    if (!palette) return;

    gridWidth = w;
    gridHeight = h;

    if (!imageData || imageData.width !== w || imageData.height !== h) {
        imageData = new ImageData(w, h);
    }

    const rgba = imageData.data;
    for (let i = 0; i < pixels.length; i++) {
        const idx = pixels[i] * 3;
        const out = i * 4;
        rgba[out] = palette[idx];
        rgba[out + 1] = palette[idx + 1];
        rgba[out + 2] = palette[idx + 2];
        rgba[out + 3] = 255;
    }

    drawFrame();
}

function drawFrame() {
    const wrap = canvas.parentElement;
    canvas.width = wrap.clientWidth;
    canvas.height = wrap.clientHeight;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (!imageData) return;

    const tmp = document.createElement("canvas");
    tmp.width = gridWidth;
    tmp.height = gridHeight;
    tmp.getContext("2d").putImageData(imageData, 0, 0);

    const scaleX = canvas.width / gridWidth;
    const scaleY = canvas.height / gridHeight;
    const fitScale = Math.min(scaleX, scaleY) * 0.95;

    ctx.save();

    // Nearest-neighbor for crisp cell rendering at any zoom
    ctx.imageSmoothingEnabled = false;

    ctx.translate(canvas.width / 2 + panX, canvas.height / 2 + panY);
    ctx.scale(zoom * fitScale, zoom * fitScale);
    ctx.drawImage(tmp, -gridWidth / 2, -gridHeight / 2);

    ctx.restore();
}

function handleMessage(msg) {
    if (msg.type === "sim_info") {
        simInfo = msg.data;
        updateSimUI(msg.data);
        zoom = 1;
        panX = 0;
        panY = 0;
        updateStatusInfo();
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
    } else if (msg.type === "worker_killed") {
        document.getElementById("status-bar").textContent =
            `Worker ${msg.data.worker_id} killed (${msg.data.num_workers} remaining)`;
    } else if (msg.type === "worker_healed") {
        document.getElementById("status-bar").textContent =
            `Worker healed (${msg.data.num_workers} workers)`;
    } else if (msg.type === "chaos_status") {
        chaosEnabled = msg.data.enabled;
        document.getElementById("btn-chaos").textContent =
            `Auto-Chaos: ${chaosEnabled ? "ON" : "OFF"}`;
    }
}

function updateStatusInfo() {
    if (!simInfo) return;
    document.getElementById("status-info").textContent =
        `${simInfo.width}x${simInfo.height} | ${simInfo.name}`;
}

function updateSimUI(info) {
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

    const simSelect = document.getElementById("sim-select");
    for (const opt of simSelect.options) {
        if (opt.value === info.name) {
            simSelect.value = info.name;
            break;
        }
    }
}

function updateMetrics(data) {
    document.getElementById("m-step").textContent = (data.step_ms || 0).toFixed(1);
    document.getElementById("m-halo").textContent = (data.halo_ms || 0).toFixed(1);
    document.getElementById("m-steps").textContent = data.step_count || 0;

    if (data.num_workers) {
        document.getElementById("worker-slider").value = data.num_workers;
        document.getElementById("worker-count").textContent = data.num_workers;
    }
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

function canvasToGrid(clientX, clientY) {
    const rect = canvas.getBoundingClientRect();
    const cx = clientX - rect.left;
    const cy = clientY - rect.top;

    if (!gridWidth || !gridHeight) return { row: 0, col: 0 };

    const scaleX = canvas.width / gridWidth;
    const scaleY = canvas.height / gridHeight;
    const fitScale = Math.min(scaleX, scaleY) * 0.95;

    const gx = (cx - canvas.width / 2 - panX) / (zoom * fitScale) + gridWidth / 2;
    const gy = (cy - canvas.height / 2 - panY) / (zoom * fitScale) + gridHeight / 2;

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
    if (simInfo) {
        select.value = simInfo.name;
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

// Zoom with scroll wheel, centered on mouse position
canvas.addEventListener("wheel", (e) => {
    e.preventDefault();

    const rect = canvas.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;

    // Mouse position relative to canvas center + pan offset
    const dx = mouseX - canvas.width / 2 - panX;
    const dy = mouseY - canvas.height / 2 - panY;

    const zoomFactor = e.deltaY > 0 ? 0.9 : 1.1;
    const newZoom = Math.max(0.1, Math.min(50, zoom * zoomFactor));

    // Adjust pan so the point under the mouse stays fixed
    panX -= dx * (newZoom / zoom - 1);
    panY -= dy * (newZoom / zoom - 1);

    zoom = newZoom;
    drawFrame();
}, { passive: false });

// Pan with shift+drag or middle mouse drag
canvas.addEventListener("mousedown", (e) => {
    if (e.button === 1 || (e.button === 0 && e.shiftKey)) {
        isPanning = true;
        lastPanX = e.clientX;
        lastPanY = e.clientY;
        e.preventDefault();
    } else if (e.button === 0 && !e.shiftKey) {
        const { row, col } = canvasToGrid(e.clientX, e.clientY);
        send({ type: "perturb", row, col, channel: 0, value: 1.0, radius: 3 });
    } else if (e.button === 2) {
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

window.addEventListener("resize", drawFrame);

// Kill/Heal/Chaos
document.getElementById("btn-kill").addEventListener("click", () => {
    send({ type: "kill_worker" });
});

document.getElementById("btn-heal").addEventListener("click", () => {
    send({ type: "heal_worker" });
});

let chaosEnabled = false;
document.getElementById("btn-chaos").addEventListener("click", () => {
    chaosEnabled = !chaosEnabled;
    send({ type: "chaos", enabled: chaosEnabled });
    document.getElementById("btn-chaos").textContent =
        `Auto-Chaos: ${chaosEnabled ? "ON" : "OFF"}`;
});

connect();
