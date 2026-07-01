const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const NAMES = []; let LXY = []; let N = 0; let armed = 0; let cur = 0; let showGrid = true;
let status = []; let bucketSize = 1; let nFrames = 0; let placed = new Set(); let clicks = []; let curH = null;
let residualThreshold = 60.0;   // per-frame readout colour cutoff; set from /api/state
let LINE_NAMES = []; let armedLine = null; let lineClicks = [];
let pendingPoll = null;   // setInterval handle while a background refit is in flight
const LINE_COLORS = {near_touchline:"#ff5ca8", far_touchline:"#5cc8ff",
  own_goal_line:"#ffd95c", opp_goal_line:"#b07cff", midline:"#5cffa8"};
const img = new Image();

// canonical pitch edges (landmark index pairs) for the reprojected overlay
// midline is drawn as two halves through the centre mark: far half [4,6] (halfway_far ->
// center) AND near half [6,5] (center -> halfway_near). halfway_near (5) is not clickable
// but its canonical position is known, so the FULL midline can still be overlaid.
const EDGES = [[0,1],[1,3],[3,2],[2,0],[4,6],[6,5],[9,10],[11,12],[9,11],[10,12],
               [13,14],[15,16],[13,15],[14,16],[17,18],[19,20]];

async function api(path, opts){ const r = await fetch(path, opts); return r.json(); }
function postJSON(path, body){
  return api(path, {method:"POST", headers:{"Content-Type":"application/json"},
                    body: JSON.stringify(body)}); }
function colorFor(s){return s==="green"?"#39d98a":s==="yellow"?"#ffb454":"#e0524d";}

function inv3(m){ // invert a flat 9-array 3x3
  const a=m[0],b=m[1],c=m[2],d=m[3],e=m[4],f=m[5],g=m[6],h=m[7],i=m[8];
  const A=e*i-f*h, B=-(d*i-f*g), C=d*h-e*g;
  const det=a*A+b*B+c*C; if(Math.abs(det)<1e-12) return null;
  const id=1/det;
  return [A*id,(c*h-b*i)*id,(b*f-c*e)*id, B*id,(a*i-c*g)*id,(c*d-a*f)*id,
          C*id,(b*g-a*h)*id,(a*e-b*d)*id];
}
function applyH(m,x,y){ const w=m[6]*x+m[7]*y+m[8];
  return [(m[0]*x+m[1]*y+m[2])/w, (m[3]*x+m[4]*y+m[5])/w]; }
// project with the homogeneous w so callers can drop points BEHIND the camera (w<=0),
// which otherwise flip to garbage pixels ("lines in the sky").
function projW(m,x,y){ const w=m[6]*x+m[7]*y+m[8];
  return [(m[0]*x+m[1]*y+m[2])/w, (m[3]*x+m[4]*y+m[5])/w, w]; }

function renderPalette(){
  const p=document.getElementById("palette");
  p.innerHTML="<h3 style='font-size:12px;color:#9aa4b2'>LANDMARK</h3>";
  for(let i=0;i<N;i++){ if(i===5) continue;
    const d=document.createElement("div");
    d.className="kp"+(i===armed?" armed":"")+(placed.has(i)?" placed":"");
    d.textContent=`${i} ${NAMES[i]||""}`+(placed.has(i)?" ✓":"");
    d.onclick=()=>{armed=i; armedLine=null; renderPalette();}; p.appendChild(d); }
  const lh=document.createElement("h3");
  lh.style.cssText="font-size:12px;color:#9aa4b2;margin-top:10px"; lh.textContent="LINES";
  p.appendChild(lh);
  for(const name of LINE_NAMES){
    const d=document.createElement("div");
    d.className="kp"+(name===armedLine?" armed":"");
    d.textContent=name; d.style.color=LINE_COLORS[name]||"#dfe7ee";
    d.onclick=()=>{armedLine=name; armed=-1; renderPalette();};
    p.appendChild(d);
  }
}
function renderTimeline(){
  const t=document.getElementById("timeline"); t.innerHTML="";
  for(const s of status){const d=document.createElement("div");
    d.style.flex="1"; d.style.background=colorFor(s); t.appendChild(d);}
  const m=document.createElement("div"); m.id="tlmarker"; t.appendChild(m);
  updateMarker();
}
function updateMarker(){
  const m=document.getElementById("tlmarker");
  if(m && nFrames>1) m.style.left=(100*cur/(nFrames-1))+"%";
}
function renderTicks(){
  const el=document.getElementById("ticks"); if(!el || nFrames<2) return;
  el.innerHTML=""; const N=8;
  for(let i=0;i<=N;i++){
    const s=document.createElement("span"); s.className="tick";
    s.textContent=Math.round(i/N*(nFrames-1)); s.style.left=(100*i/N)+"%";
    if(i===0) s.style.transform="none";
    else if(i===N) s.style.transform="translateX(-100%)";
    el.appendChild(s);
  }
}

function drawOverlay(){
  if(!showGrid || !curH || !LXY.length) return;
  const hi=inv3(curH); if(!hi) return;            // pitch -> normalized image
  ctx.strokeStyle="#39d98a"; ctx.lineWidth=1.5; ctx.globalAlpha=0.85;
  const NS=24;                                     // samples per edge
  for(const [a,b] of EDGES){
    const ax=LXY[a][0], ay=LXY[a][1], bx=LXY[b][0], by=LXY[b][1];
    let prev=null;                                 // previous IN-FRONT pixel, or null
    for(let k=0;k<=NS;k++){
      const t=k/NS, px=ax+(bx-ax)*t, py=ay+(by-ay)*t;
      const q=projW(hi, px, py);
      // clip to the part of the edge in front of the camera; a segment that crosses
      // behind the camera BREAKS here instead of drawing a spurious line to the horizon.
      const cur = q[2]>1e-9 ? [q[0]*canvas.width, q[1]*canvas.height] : null;
      if(cur && prev){ ctx.beginPath(); ctx.moveTo(prev[0],prev[1]); ctx.lineTo(cur[0],cur[1]); ctx.stroke(); }
      prev=cur;
    }
  }
  ctx.globalAlpha=1.0;
}

function drawFrame(){
  ctx.clearRect(0,0,canvas.width,canvas.height);
  if(img.complete && img.naturalWidth) ctx.drawImage(img,0,0,canvas.width,canvas.height);
  drawOverlay();
  for(const c of clicks) if(c.frame===cur){
    ctx.fillStyle="#39d98a";
    ctx.beginPath(); ctx.arc(c.x*canvas.width, c.y*canvas.height,6,0,7); ctx.fill();
    ctx.fillStyle="#0f1115"; ctx.font="10px sans-serif";
    ctx.fillText(c.kp_idx, c.x*canvas.width-3, c.y*canvas.height+3);
  }
  for(const lc of lineClicks) if(lc.line_id && lc.frame===cur){
    const cx=lc.x*canvas.width, cy=lc.y*canvas.height, r=6;
    ctx.fillStyle=LINE_COLORS[lc.line_id]||"#5cffa8";
    ctx.beginPath();
    ctx.moveTo(cx,cy-r); ctx.lineTo(cx+r,cy); ctx.lineTo(cx,cy+r); ctx.lineTo(cx-r,cy);
    ctx.closePath(); ctx.fill();
  }
}

async function loadFrame(i){
  cur=i; document.getElementById("frameNum").textContent=i;
  const s=document.getElementById("scrub"); if(+s.value!==i) s.value=i;  // keep slider synced
  updateMarker();
  const fh=await api(`/api/frame_h/${i}`); curH=fh.h;
  const resEl = document.getElementById("res");
  if (fh.residual == null) { resEl.textContent = "—"; resEl.style.color = ""; }
  else {
    resEl.textContent = fh.residual.toFixed(3) + " (" + fh.n_points + " pts)";
    resEl.style.color = fh.residual <= residualThreshold ? "#39d98a" : "#ffb454";
  }
  img.onload=drawFrame; img.src=`/api/frame/${i}?t=${Date.now()}`;
}

function applyState(st){
  N=st.landmark_names.length; for(let i=0;i<N;i++) NAMES[i]=st.landmark_names[i];
  LXY=st.landmark_xy;
  LINE_NAMES=st.line_names||[];
  status = st.status_buckets;
  bucketSize = st.bucket_size;
  nFrames = st.n_frames;
  residualThreshold = st.residual_px_threshold || 60.0;
  maybePoll(st.pending || 0);
  document.getElementById("cov").textContent=Math.round(st.coverage*100)+"%";
  document.getElementById("nclicks").textContent=st.n_clicks;
  document.getElementById("scrub").max=st.n_frames-1;
  document.getElementById("frameMax").textContent=st.n_frames-1;
  renderPalette(); renderTimeline(); renderTicks(); drawFrame();
}

function maybePoll(pending){
  if(pending > 0 && pendingPoll === null){
    // Clicks are non-blocking: the server defers the bundle re-solve to a background
    // worker, so we poll while pending>0 and re-fetch the CURRENT frame's overlay each
    // tick (incl. the drain tick) so the just-placed click's effect appears once the
    // worker drains (~100-300 ms). Tight interval since the backend no longer blocks.
    pendingPoll = setInterval(async () => {
      const st = await api("/api/state");
      applyState(st);                                   // refresh timeline/coverage
      const fh = await api(`/api/frame_h/${cur}`); curH = fh.h; drawFrame();
      if((st.pending || 0) === 0){ clearInterval(pendingPoll); pendingPoll = null; }
    }, 300);
  } else if(pending === 0 && pendingPoll !== null){
    clearInterval(pendingPoll); pendingPoll = null;
  }
}

let dragging = null;  // {kp_idx, c} while dragging an existing same-frame dot
let didDrag = false;

function canvasNorm(e){
  const r = canvas.getBoundingClientRect();
  return [(e.clientX - r.left) / r.width, (e.clientY - r.top) / r.height];
}

canvas.onmousedown = (e) => {
  const [x, y] = canvasNorm(e);
  dragging = null;
  for (let i = clicks.length - 1; i >= 0; i--) {
    const c = clicks[i];
    if (c.frame !== cur) continue;
    const dx = (c.x - x) * canvas.width;
    const dy = (c.y - y) * canvas.height;
    if (Math.hypot(dx, dy) < 10) { dragging = { kp_idx: c.kp_idx, c }; return; }
  }
};

canvas.onmousemove = (e) => {
  if (!dragging) return;
  didDrag = true;
  const [x, y] = canvasNorm(e);
  dragging.c.x = x; dragging.c.y = y;   // live local preview
  drawFrame();
};

canvas.onmouseup = async (e) => {
  if (dragging && didDrag) {
    const [x, y] = canvasNorm(e);
    applyState(await postJSON("/api/nudge",
      { frame: cur, kp_idx: dragging.kp_idx, x, y }));
    const fh = await api(`/api/frame_h/${cur}`); curH = fh.h; drawFrame();
  }
  dragging = null;
};

canvas.onclick = async (e) => {
  if (didDrag) { didDrag = false; return; }   // suppress synthetic click after drag
  if (armed < 0 && !armedLine) return;        // nothing armed — ignore
  const [x, y] = canvasNorm(e);
  if (armedLine) {
    lineClicks.push({ frame: cur, line_id: armedLine, x, y });
    applyState(await postJSON("/api/line_click", { frame: cur, line_id: armedLine, x, y }));
  } else {
    clicks.push({ frame: cur, kp_idx: armed, x, y }); placed.add(armed);
    applyState(await postJSON("/api/click", { frame: cur, kp_idx: armed, x, y }));
  }
  const fh = await api(`/api/frame_h/${cur}`); curH = fh.h; drawFrame();
};

document.getElementById("scrub").oninput=(e)=>loadFrame(+e.target.value);
document.getElementById("timeline").onclick=(e)=>{   // click a colour band -> jump there
  const r=e.currentTarget.getBoundingClientRect();
  const frac=Math.min(1,Math.max(0,(e.clientX-r.left)/r.width));
  if(nFrames>1) loadFrame(Math.round(frac*(nFrames-1)));
};
document.getElementById("undo").onclick=async()=>{
  applyState(await postJSON("/api/undo",{}));
  const cl=await api("/api/clicks"); clicks=cl.clicks; lineClicks=cl.line_clicks||[];
  placed=new Set(clicks.map(c=>c.kp_idx));
  const fh=await api(`/api/frame_h/${cur}`); curH=fh.h; drawFrame();
};
document.getElementById("grid").onclick=()=>{showGrid=!showGrid; drawFrame();};
document.getElementById("export").onclick=async()=>{
  const r=await postJSON("/api/export",{}); alert("Exported to "+r.exported_to);};
function jumpRed(dir){
  let b = Math.floor(cur / bucketSize) + dir;
  while(b >= 0 && b < status.length){
    if(status[b] === "red"){ loadFrame(Math.min(nFrames - 1, b * bucketSize)); return; }
    b += dir;
  }
}
document.getElementById("nextRed").onclick=()=>jumpRed(1);
document.getElementById("prevRed").onclick=()=>jumpRed(-1);
window.onkeydown=(e)=>{ if(e.key>="0"&&e.key<="9"){armed=+e.key; armedLine=null; renderPalette();} };

(async()=>{
  const cl = await api("/api/clicks");
  clicks = cl.clicks; lineClicks = cl.line_clicks || [];
  placed = new Set(clicks.map(c=>c.kp_idx));
  applyState(await api("/api/state"));
  loadFrame(0);
})();
