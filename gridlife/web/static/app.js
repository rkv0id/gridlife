const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

let ws = null;
let simInfo = null;

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
        canvas.width = bmp.width;
        canvas.height = bmp.height;
        ctx.drawImage(bmp, 0, 0);
    });
}

function handleMessage(msg) {
    if (msg.type === "sim_info") {
        simInfo = msg.data;
        updateSimUI(msg.data);
    } else if (msg.type === "metrics") {
        updateMetrics(msg.data);
    }
}

function updateSimUI(info) {
    // Set canvas interpolation mode
    if (info.pixelated) {
        canvas.classList.add("pixelated");
    } else {
        canvas.classList.remove("pixelated");
    }

    // Build param sliders
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

    // Build presets
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

    // Update sliders
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

// Canvas click to perturb
canvas.addEventListener("mousedown", (e) => {
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    const col = Math.floor((e.clientX - rect.left) * scaleX);
    const row = Math.floor((e.clientY - rect.top) * scaleY);
    const value = e.button === 2 ? 0.0 : 1.0;
    send({ type: "perturb", row, col, channel: 0, value, radius: 3 });
});

canvas.addEventListener("contextmenu", (e) => e.preventDefault());

connect();
