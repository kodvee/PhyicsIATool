// Shared helpers for the wizard pages.

async function postJSON(url, body) {
  const r = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}),
  });
  const data = await r.json().catch(() => ({}));
  return { ok: r.ok, data };
}

async function getJSON(url) {
  const r = await fetch(url);
  return r.json();
}

// Validate-and-advance to a stage, then navigate there.
async function advance(toStage) {
  const { ok, data } = await postJSON(`/api/advance/${toStage}`, {});
  if (ok && data.ok) {
    window.location = data.next;
  } else {
    alert(data.error || "Cannot continue yet.");
  }
}

function goBack(n) { window.location = `/stage/${n}`; }

// Interactive frame canvas. Loads /api/frame/<idx> as a background and supports
// either rectangle drawing ("rect") or point clicking ("point"/"points").
// The displayed image represents a region of the full frame `sourceWidth` px
// wide starting at (offsetX, offsetY). Coordinates are mapped back to FULL
// frame pixels as full = offset + display * (sourceWidth / displayedWidth),
// so a viewport crop never changes the reported physical coordinates.
class FrameCanvas {
  constructor(canvasId, mode, opts = {}) {
    this.canvas = document.getElementById(canvasId);
    this.ctx = this.canvas.getContext("2d");
    this.mode = mode;                 // 'rect' | 'point' | 'points'
    this.sourceWidth = opts.sourceWidth || null;  // full-frame px width shown
    this.offsetX = opts.offsetX || 0;
    this.offsetY = opts.offsetY || 0;
    this.raw = opts.raw || false;     // load uncropped full frame (crop tool)
    this.scale = 1;                   // full_px = display_px * scale (set on load)
    this.maxPoints = opts.maxPoints || (mode === "points" ? 2 : 1);
    this.color = opts.color || "#4f8cff";
    this.img = new Image();
    this.rect = null;                 // {x,y,w,h} display px
    this.points = [];                 // [{x,y}] display px
    this.dragging = false;
    this.start = null;
    this.onChange = opts.onChange || (() => {});
    this._pendingFullRect = opts.initialRect || null;

    // ---- in-place playback over the source frames ----
    this.frameCount = opts.frameCount || 0;
    this.fps = opts.fps || 60;
    this.curFrame = opts.startFrame || 0;
    this.playing = false;
    this.playTimer = null;
    this.onFrame = opts.onFrame || (() => {});       // called with each frame index
    this.onPlayState = opts.onPlayState || (() => {}); // called with true/false

    this.img.onload = () => {
      this.canvas.width = this.img.width;
      this.canvas.height = this.img.height;
      this.scale = this.sourceWidth ? this.sourceWidth / this.img.width : 1;
      if (this._pendingFullRect) { this._applyFullRect(this._pendingFullRect); this._pendingFullRect = null; }
      this.redraw();
      // Drive playback off load so we never queue frames faster than they decode.
      if (this.playing) this.playTimer = setTimeout(() => this._advance(), 1000 / this.fps);
    };
    this._bind();
  }

  loadFrame(idx) {
    this.img.src = `/api/frame/${idx}?${this.raw ? "raw=1&" : ""}t=${Date.now()}`;
  }

  goTo(idx) {
    if (this.frameCount) idx = Math.max(0, Math.min(this.frameCount - 1, Math.round(idx)));
    this.curFrame = idx;
    this.loadFrame(idx);
    this.onFrame(idx);
  }
  _advance() {
    if (!this.playing) return;
    if (this.frameCount && this.curFrame >= this.frameCount - 1) { this.pause(); return; }
    this.goTo(this.curFrame + 1);
  }
  play() {
    if (this.frameCount && this.curFrame >= this.frameCount - 1) this.goTo(0);
    this.playing = true; this.onPlayState(true); this._advance();
  }
  pause() { this.playing = false; clearTimeout(this.playTimer); this.onPlayState(false); }
  step(d) { this.pause(); this.goTo(this.curFrame + d); }

  _pos(e) {
    const r = this.canvas.getBoundingClientRect();
    // account for CSS scaling of the canvas element
    const sx = this.canvas.width / r.width;
    const sy = this.canvas.height / r.height;
    return { x: (e.clientX - r.left) * sx, y: (e.clientY - r.top) * sy };
  }

  _bind() {
    if (this.mode === "display") {
      return;  // display-only: no drawing/clicking
    }
    if (this.mode === "rect") {
      this.canvas.addEventListener("mousedown", (e) => {
        this.dragging = true; this.start = this._pos(e); this.rect = null;
      });
      this.canvas.addEventListener("mousemove", (e) => {
        if (!this.dragging) return;
        const p = this._pos(e);
        this.rect = {
          x: Math.min(this.start.x, p.x), y: Math.min(this.start.y, p.y),
          w: Math.abs(p.x - this.start.x), h: Math.abs(p.y - this.start.y),
        };
        this.redraw();
      });
      window.addEventListener("mouseup", () => {
        if (this.dragging) { this.dragging = false; this.onChange(this.fullRect()); }
      });
    } else {
      this.canvas.addEventListener("click", (e) => {
        const p = this._pos(e);
        if (this.points.length >= this.maxPoints) this.points = [];
        this.points.push(p);
        this.redraw();
        this.onChange(this.fullPoints());
      });
    }
  }

  redraw() {
    const c = this.ctx;
    c.clearRect(0, 0, this.canvas.width, this.canvas.height);
    if (this.img.complete) c.drawImage(this.img, 0, 0);
    c.strokeStyle = this.color; c.fillStyle = this.color; c.lineWidth = 2;
    if (this.rect) {
      c.strokeRect(this.rect.x, this.rect.y, this.rect.w, this.rect.h);
      c.fillStyle = "rgba(79,140,255,0.15)";
      c.fillRect(this.rect.x, this.rect.y, this.rect.w, this.rect.h);
    }
    this.points.forEach((p, i) => {
      c.fillStyle = this.color;
      c.beginPath(); c.arc(p.x, p.y, 5, 0, Math.PI * 2); c.fill();
      c.fillText(String(i + 1), p.x + 8, p.y - 8);
    });
  }

  fullRect() {
    if (!this.rect) return null;
    return {
      x: Math.round(this.offsetX + this.rect.x * this.scale),
      y: Math.round(this.offsetY + this.rect.y * this.scale),
      w: Math.round(this.rect.w * this.scale),
      h: Math.round(this.rect.h * this.scale),
    };
  }
  fullPoints() {
    return this.points.map((p) => ({
      x: this.offsetX + p.x * this.scale,
      y: this.offsetY + p.y * this.scale,
    }));
  }
  _applyFullRect(full) {
    this.rect = {
      x: (full.x - this.offsetX) / this.scale, y: (full.y - this.offsetY) / this.scale,
      w: full.w / this.scale, h: full.h / this.scale,
    };
  }
  setRect(full) {
    if (this.scale && this.img.complete) { this._applyFullRect(full); this.redraw(); }
    else { this._pendingFullRect = full; }
  }
}

// Build a play / frame-step / slider control bar inside `mountEl` and wire it to
// `fc` (a FrameCanvas). `onFrame(idx)` is an optional extra callback. Returns an
// object exposing the FrameCanvas so callers can read fc.curFrame, etc.
function attachScrubber(fc, mountEl, opts = {}) {
  const bar = document.createElement("div");
  bar.className = "scrubber";
  bar.innerHTML = `
    <input type="range" class="sb-range" min="0" max="${Math.max(1, fc.frameCount - 1)}" value="${fc.curFrame}" />
    <div class="row" style="margin-top:6px; align-items:center">
      <div style="flex:0"><button type="button" class="sb-play primary">▶ Play</button></div>
      <div style="flex:0"><button type="button" class="sb-prev">− frame</button></div>
      <div style="flex:0"><button type="button" class="sb-next">+ frame</button></div>
      <div style="flex:1" class="kv">frame <span class="sb-num">${fc.curFrame}</span> · <span class="sb-time">0.000</span>s</div>
    </div>`;
  mountEl.appendChild(bar);
  const range = bar.querySelector(".sb-range");
  const playBtn = bar.querySelector(".sb-play");
  const num = bar.querySelector(".sb-num");
  const time = bar.querySelector(".sb-time");

  // Compose with any callbacks the FrameCanvas was constructed with.
  const userOnFrame = fc.onFrame, userOnPlay = fc.onPlayState;
  fc.onFrame = (idx) => {
    range.value = idx; num.textContent = idx; time.textContent = (idx / fc.fps).toFixed(3);
    userOnFrame(idx);
    if (opts.onFrame) opts.onFrame(idx);
  };
  fc.onPlayState = (p) => { playBtn.textContent = p ? "⏸ Pause" : "▶ Play"; userOnPlay(p); };

  range.addEventListener("input", (e) => { fc.pause(); fc.goTo(+e.target.value); });
  playBtn.addEventListener("click", () => { fc.playing ? fc.pause() : fc.play(); });
  bar.querySelector(".sb-prev").addEventListener("click", () => fc.step(-1));
  bar.querySelector(".sb-next").addEventListener("click", () => fc.step(1));
  return bar;
}
