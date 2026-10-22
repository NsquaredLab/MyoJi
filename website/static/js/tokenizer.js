(() => {
  const widget = document.querySelector('.flow-widget');
  if (!widget) return;
  const $ = s => widget.querySelector(s), $$ = s => [...widget.querySelectorAll(s)];
  const examples = JSON.parse(document.querySelector('#tokenizer-examples').textContent);
  const timing = JSON.parse(document.querySelector('#tokenizer-geometry').textContent);
  const reduced = matchMedia('(prefers-reduced-motion: reduce)');
  const names = ['Window', 'Features', 'Match', 'Token', 'Lookup'];
  const copy = {
    learned: [
      ['Start with a short history of muscle activity.', 'For each electrode, the tokenizer uses the latest 480 ms of EMG context. The final 160 ms patch ends at the current update; future samples are never used.', '480 ms', 'of causal signal history'],
      ['Describe the signal with a learned feature vector.', 'A shared causal encoder turns this history into 256 continuous values. Its weights were pretrained through spectral reconstruction, without hand-pose labels.', '256', 'continuous feature values'],
      ['Find the closest vector in the shared codebook.', 'Compare all 256 components against each of 1,024 entries. Entry 93 has the smallest squared Euclidean distance for this recorded update. Click a row to inspect a candidate.', '93', 'the nearest codebook entry'],
      ['Replace the feature vector with its index.', 'Entry 93 becomes the token. With 1,024 entries, a packed index needs 10 bits; every electrode produces a new token every 20 ms.', '10 bits', 'instead of 256 float values'],
      ['Recover the codebook vector at the receiver.', 'The receiver looks up entry 93 in the same codebook. A separately trained decoder uses the history of retrieved vectors across electrodes to estimate 3D hand pose.', '256', 'retrieved codebook values']
    ],
    spar: [
      ['Start with a local patch of muscle signals.', 'SPAR uses a causal 160 ms window from four neighbouring electrodes. Their spectra and relationships describe the local electrical activity. The four input traces here are schematic.', '160 ms', 'from a four-electrode patch'],
      ['Extract spectral and spatial features.', 'Six frequency bands each supply three retained matrix-log features. After standardization, these form an 18-component vector for the centre electrode.', '18', 'values: six bands × three'],
      ['Find the closest centroid in the SPAR codebook.', 'The participant-specific codebook contains 1,024 centroids fitted on training features without hand-pose labels. Comparing all 18 components selects entry 239 for this update.', '239', 'the nearest codebook entry'],
      ['Send the index in the same 10-bit format.', 'Token 239 replaces the 18 continuous feature values. Its packed index uses 10 bits, and a fresh token is produced every 20 ms for each electrode.', '10 bits', 'instead of 18 float values'],
      ['Look up the spectral–spatial vector.', 'The receiver retrieves centroid 239 and restores structural zeros for missing neighbours. This interior electrode has none. The decoder then uses the vector history across electrodes to estimate 3D hand pose.', '18', 'retrieved codebook values']
    ]
  };
  const palettes = Object.fromEntries(Object.entries(examples).map(([key, ex]) => [key, Math.max(...ex.vector.map(Math.abs), ...ex.candidates.flatMap(c => c.vector.map(Math.abs)))]));
  let method = 'learned', progress = reduced.matches ? .98 : 0, playing = !reduced.matches;
  let phase = -1, inspected = examples.learned.token, visible = false, raf = 0, last = 0;
  let featureCells = [], receiverCells = [], rowNodes = [], bitNodes = [], routes = [], previewKey = '';
  const clamp = value => Math.max(0, Math.min(1, value));
  const smooth = (a, b, value) => { const x = clamp((value-a)/(b-a)); return x*x*(3-2*x); };
  const getPhase = () => Math.min(4, timing.boundaries.slice(1).findIndex(boundary => progress < boundary) < 0 ? 4 : timing.boundaries.slice(1).findIndex(boundary => progress < boundary));
  const gray = value => {
    const x = clamp(.5 + .5*value/palettes[method]);
    return `rgb(${Math.round(239-x*199)} ${Math.round(239-x*199)} ${Math.round(235-x*190)})`;
  };
  const vectorHTML = vector => vector.map((value,i) => `<span class="feature-cell" style="background:${gray(value)}" title="Component ${i+1}: ${value}"></span>`).join('');
  function wavePath(wave, centre, height) {
    const limit = Math.max(...wave.map(Math.abs))*1.1;
    return wave.map((v,i) => `${i?'L':'M'}${(12+i*196/(wave.length-1)).toFixed(2)},${(centre-v/limit*height/2).toFixed(2)}`).join(' ');
  }
  function setExample() {
    const ex = examples[method], win = ex.candidates.find(c => c.id === ex.token);
    inspected = ex.token;
    $('[data-input-kind]').textContent = method === 'learned' ? 'Recorded EMG' : '4 channels · schematic';
    $('[data-support]').textContent = `${ex.context_ms} ms context`;
    $('[data-patch]').textContent = method === 'learned' ? '160 ms final patch' : 'Local spectral coupling';
    
    $('[data-encoder-name]').textContent = method === 'learned' ? 'Causal encoder' : '6 bands × 3 features';
    
    $('[data-feature-count]').textContent = ex.dimensions;
    $('[data-feature-grid]').innerHTML = vectorHTML(ex.vector);
    $('[data-feature-grid]').dataset.dimensions = ex.dimensions;
    $('[data-feature-grid]').setAttribute('aria-label', `Recorded ${ex.dimensions}-dimensional input feature vector`);
    $('[data-receiver-grid]').innerHTML = vectorHTML(win.vector);
    $('[data-receiver-grid]').dataset.dimensions = ex.dimensions;
    $('[data-receiver-grid]').setAttribute('aria-label', `Retrieved codebook entry ${ex.token}, ${ex.dimensions} components`);
    featureCells = $$('[data-feature-grid] .feature-cell');
    receiverCells = $$('[data-receiver-grid] .feature-cell');
    const maximum = Math.max(...ex.candidates.map(c => c.distance));
    $('.codebook-rows').innerHTML = ex.ranking.map(id => {
      const candidate = ex.candidates.find(c => c.id === id);
      return `<button type="button" class="code-row" data-candidate="${id}" data-nearest="${id===ex.token}" aria-label="Inspect token ${id}, squared distance ${candidate.distance.toFixed(4)}${id===ex.token?', nearest match':''}" aria-pressed="false"><span>${id}</span><div class="distance-track"><div class="distance-fill" style="width:${candidate.distance/maximum*100}%"></div></div><span class="code-distance">${candidate.distance.toFixed(2)}</span></button>`;
    }).join('');
    rowNodes = $$('.code-row');
    $('[data-winner-label]').textContent = `${ex.token} selected`;
    $('[data-output-token]').textContent = ex.token;
    $('[data-packet-token]').textContent = ex.token;
    $('[data-lookup-label]').textContent = 'Receiver lookup';
    $('.token-bits').innerHTML = ex.bits.split('').map(bit => `<span data-on="${bit==='1'}">${bit}</span>`).join('');
    $('.token-bits').setAttribute('aria-label', `Token ${ex.token} as ten binary digits: ${ex.bits}`);
    bitNodes = $$('.token-bits span');
    $('[data-readout]').textContent = `${ex.dimensions} values → token ${ex.token} → 10 bits`;
    $('[data-provenance]').textContent = ex.provenance;
    $('[data-source-note]').textContent = method === 'learned' ? 'The waveform is the recorded 480 ms input context, displayed with min/max pairs to preserve peaks. Feature cells and receiver cells show every component in its original order.' : 'The four input traces illustrate a local electrode patch; they are schematic. The displayed 18-component input vector, codebook entries, distances, and token 239 are recorded values.';
    $('[data-time-start]').textContent = `−${ex.context_ms} ms`;
    $('[data-window]').setAttribute('x', method === 'learned' ? 143 : 12);
    $('[data-window]').setAttribute('width', method === 'learned' ? 65 : 196);
    let paths;
    if (method === 'learned') paths = [wavePath(ex.waveform,75,100)];
    else {
      // Explicitly schematic channel traces; recorded SPAR values begin at the vector.
      paths = Array.from({length:4},(_,c) => wavePath(Array.from({length:180},(_,i) => Math.sin(i*.82+c*.9)*Math.sin(i*.11+c*.4) + .35*Math.cos(i*1.73+c)), 28+c*30, 23));
    }
    $('[data-wave-background]').innerHTML = paths.map(d => `<path d="${d}"/>`).join('');
    $('[data-wave-foreground]').innerHTML = paths.map(d => `<path d="${d}"/>`).join('');
    $('.signal-chart').setAttribute('aria-label', method === 'learned' ? 'Recorded normalized EMG waveform over 480 milliseconds; the final 160 milliseconds are shaded' : 'Four schematic EMG traces illustrating a 160-millisecond neighbouring-electrode patch');
    $$('[data-packet-cell]').forEach((cell,i) => cell.setAttribute('fill', gray(ex.vector[i])));
    previewKey = '';
    updateCandidate();
    phase = -1;
    measureRoutes();
    draw();
  }
  function updateCandidate(announce=false) {
    const ex = examples[method], candidate = ex.candidates.find(c => c.id === inspected), win = ex.candidates.find(c => c.id === ex.token);
    $('[data-inspected-token]').innerHTML = `${candidate.id} <span>token ID</span>`;
    $('[data-distance]').textContent = candidate.distance.toFixed(4);
    $('[data-candidate-foot]').textContent = candidate.id === ex.token ? 'This is the nearest of all 1,024 entries.' : `The nearest match is token ${ex.token} (distance ${win.distance.toFixed(4)}). Transmission keeps that match.`;
    $('[data-candidate-label]').textContent = `Codebook entry ${candidate.id}`;
    $('[data-vector-dimensions]').textContent = ex.dimensions;
    $('.return-match').disabled = candidate.id === ex.token;
    rowNodes.forEach(row => {
      const selected = Number(row.dataset.candidate) === inspected;
      row.dataset.inspected = String(selected && $('.flow-inspector').open);
      row.setAttribute('aria-pressed', String(selected && $('.flow-inspector').open));
    });
    const key = `${method}:${inspected}`;
    if (key !== previewKey) {
      previewKey = key;
      ['input','candidate'].forEach(name => {
        const vector = name === 'input' ? ex.vector : candidate.vector, node = $(`[data-preview="${name}"]`);
        node.dataset.dimensions = ex.dimensions;
        node.innerHTML = vectorHTML(vector);
        node.setAttribute('aria-label', `${name === 'input' ? 'Input vector' : `Codebook entry ${candidate.id}`}, all ${ex.dimensions} components. Exact values are available below.`);
      });
      $('.component-table').innerHTML = '<table><thead><tr><th scope="col">Component</th><th scope="col">Input</th><th scope="col">Codebook</th></tr></thead><tbody>'+ex.vector.map((value,i) => `<tr><td>${i+1}</td><td>${value}</td><td>${candidate.vector[i]}</td></tr>`).join('')+'</tbody></table>';
    }
    if (announce) $('[data-announcement]').textContent = `Inspecting token ${candidate.id}, squared distance ${candidate.distance.toFixed(4)}${candidate.id === ex.token ? ', nearest match' : ''}.`;
  }
  function setPlay(next) {
    playing = next && !reduced.matches;
    $('.flow-play').dataset.playing = String(playing);
    $('.flow-play').setAttribute('aria-label', reduced.matches ? 'Next stage' : playing ? 'Pause animation' : 'Play animation');
    $('[data-play-icon]').textContent = reduced.matches ? '→' : playing ? 'Ⅱ' : '▷';
    $('[data-play-label]').textContent = reduced.matches ? 'Next' : playing ? 'Pause' : 'Play';
    run();
  }
  function renderPhase(announce=false) {
    phase = getPhase();
    const data = copy[method][phase];
    widget.dataset.phase = phase;
    $$('[data-step]').forEach(button => button.setAttribute('aria-pressed', Number(button.dataset.step) === phase));
    $$('[data-method]').forEach(button => button.setAttribute('aria-pressed', button.dataset.method === method));
    $$('[data-panel]').forEach(panel => panel.dataset.active = String(Number(panel.dataset.panel) === Math.min(3,phase)));
    $('[data-step-kicker]').textContent = `0${phase+1} / ${names[phase].toUpperCase()}`;
    $('[data-step-title]').textContent = data[0];
    $('[data-step-copy]').textContent = data[1];
    $('[data-equation-main]').textContent = data[2];
    $('[data-equation-note]').textContent = data[3];
    $('[data-progress-label]').textContent = `0${phase+1} / ${names[phase]}`;
    $('#flow-progress').setAttribute('aria-valuetext', `${names[phase]}: ${data[0]}`);
    if (announce) $('[data-announcement]').textContent = `${examples[method].name}. ${data[0]}`;
  }
  function bezier(a,b,vertical=false) {
    return vertical ? [a,{x:a.x,y:(a.y+b.y)/2},{x:b.x,y:(a.y+b.y)/2},b] : [a,{x:(a.x+b.x)/2,y:a.y},{x:(a.x+b.x)/2,y:b.y},b];
  }
  const pathOf = ([a,b,c,d]) => `M${a.x},${a.y}C${b.x},${b.y} ${c.x},${c.y} ${d.x},${d.y}`;
  function pointAt(route,t) {
    const q = 1-t;
    return {x:q*q*q*route[0].x+3*q*q*t*route[1].x+3*q*t*t*route[2].x+t*t*t*route[3].x,y:q*q*q*route[0].y+3*q*q*t*route[1].y+3*q*t*t*route[2].y+t*t*t*route[3].y};
  }
  function measureRoutes() {
    const pipeline = $('.pipeline'), origin = pipeline.getBoundingClientRect();
    if (!origin.width) return;
    const k = origin.width / pipeline.offsetWidth;   // on-screen scale (a scaled phone slide), 1 elsewhere
    const rect = node => {const r=node.getBoundingClientRect();return {x:(r.x-origin.x)/k,y:(r.y-origin.y)/k,w:r.width/k,h:r.height/k};};
    const centre = node => {const r=rect(node);return {x:r.x+r.w/2,y:r.y+r.h/2};};
    const panels = $$('[data-panel]').map(rect);
    const links = [0,1,2].map(i => {
      const a=panels[i],b=panels[i+1],sameRow=Math.abs(a.y-b.y)<5;
      if (sameRow) return bezier({x:a.x+a.w+1,y:a.y+a.h/2},{x:b.x-3,y:b.y+b.h/2});
      const startX = a.w>b.w*1.5 ? b.x+b.w/2 : a.x+a.w/2;
      const endX = b.w>a.w*1.5 ? a.x+a.w/2 : b.x+b.w/2;
      return bezier({x:startX,y:a.y+a.h+1},{x:endX,y:b.y-3},true);
    });
    $$('[data-link]').forEach((path,i)=>path.setAttribute('d',pathOf(links[i])));
    const wave=centre($('.signal-chart')),features=centre($('[data-feature-grid]'));
    const nearest=centre($('.code-row[data-nearest=true]')),output=centre($('.token-badge'));
    routes=[bezier(wave,features,Math.abs(panels[0].y-panels[1].y)>5),bezier(features,nearest,Math.abs(panels[1].y-panels[2].y)>5),bezier(nearest,output,Math.abs(panels[2].y-panels[3].y)>5)];
    drawPackets();
  }
  function packet(node, routeIndex, start, end) {
    const t=clamp((progress-start)/(end-start));
    const opacity=smooth(0,.13,t)*(1-smooth(.87,1,t));
    node.style.opacity=opacity;
    if (!routes[routeIndex]) return;
    const p=pointAt(routes[routeIndex],smooth(0,1,t));
    node.setAttribute('transform',`translate(${p.x.toFixed(2)} ${p.y.toFixed(2)})`);
  }
  function drawPackets() {
    if (progress<.4) packet($('[data-feature-packet]'),0,.15,.27);
    else packet($('[data-feature-packet]'),1,.4,.48);
    packet($('[data-token-packet]'),2,.655,.78);
  }
  function draw() {
    if (getPhase() !== phase) renderPhase();
    const t=progress;
    const scan=smooth(0,.15,t);
    $('[data-wave-clip]').setAttribute('width',196*scan);
    $('[data-scan]').setAttribute('x1',12+196*scan);
    $('[data-scan]').setAttribute('x2',12+196*scan);
    $('[data-scan]').style.opacity=1-smooth(.16,.2,t);
    featureCells.forEach((cell,i)=>{const start=.255+i/featureCells.length*.095;cell.style.opacity=.08+.92*smooth(start,start+.025,t);});
    rowNodes.forEach((row,i)=>{
      const amount=smooth(.47+i*.015,.505+i*.015,t);
      row.style.opacity=.22+.78*amount;
      row.querySelector('.distance-fill').style.transform=`scaleX(${amount})`;
    });
    $('.winner-label').style.opacity=smooth(.6,.64,t);
    $('.token-output').style.opacity=.08+.92*smooth(.76,.79,t);
    bitNodes.forEach((bit,i)=>bit.style.opacity=.12+.88*smooth(.775+i*.004,.795+i*.004,t));
    $('.receiver-lookup').style.opacity=.2+.8*smooth(.84,.88,t);
    receiverCells.forEach((cell,i)=>{const start=.85+i/receiverCells.length*.09;cell.style.opacity=.08+.92*smooth(start,start+.025,t);});
    $('#flow-progress').value=Math.round(progress*1000);
    drawPackets();
  }
  function seek(value,announce=true) {
    setPlay(false);
    progress=clamp(value);
    draw();
    if (announce) renderPhase(true);
  }
  function tick(now) {
    const dt=last?Math.min((now-last)/1000,1/30):0;
    last=now;
    if (playing && !reduced.matches) {
      progress+=dt/timing.duration;
      if(progress>1)progress=0;
      draw();
    }
    raf=playing&&visible&&!document.hidden&&!reduced.matches?requestAnimationFrame(tick):0;
  }
  function run() {
    if(!raf&&playing&&visible&&!document.hidden&&!reduced.matches){last=0;raf=requestAnimationFrame(tick);}
  }
  $('.flow-play').addEventListener('click',()=>{
    if(reduced.matches) seek(timing.stops[(phase+1)%5]);
    else {if(progress>=1)progress=0;setPlay(!playing);}
  });
  $$('[data-step]').forEach(button=>button.addEventListener('click',()=>seek(timing.stops[Number(button.dataset.step)])));
  $('#flow-progress').addEventListener('pointerdown',()=>setPlay(false));
  $('#flow-progress').addEventListener('keydown',()=>setPlay(false));
  $('#flow-progress').addEventListener('input',event=>seek(Number(event.target.value)/1000,false));
  $('#flow-progress').addEventListener('change',()=>renderPhase(true));
  $$('[data-method]').forEach(button=>button.addEventListener('click',()=>{
    method=button.dataset.method;
    if(playing)progress=0;
    setExample();
    renderPhase(true);
  }));
  $('.codebook-rows').addEventListener('click',event=>{
    const row=event.target.closest('[data-candidate]');
    if(!row)return;
    seek(timing.stops[2],false);
    inspected=Number(row.dataset.candidate);
    $('.flow-more').open=true;
    $('.flow-inspector').open=true;
    updateCandidate(true);
  });
  $('.flow-inspector').addEventListener('toggle',()=>{if($('.flow-inspector').open)setPlay(false);updateCandidate();});
  $('.flow-more').addEventListener('toggle',()=>{if($('.flow-more').open)setPlay(false);});
  $('.return-match').addEventListener('click',()=>{inspected=examples[method].token;updateCandidate(true);});
  new ResizeObserver(measureRoutes).observe($('.pipeline'));
  new IntersectionObserver(entries=>{
    visible=entries[entries.length-1].isIntersecting;
    if(!visible&&raf){cancelAnimationFrame(raf);raf=0;}
    if(visible)run();
  },{rootMargin:'80px 0px'}).observe(widget);
  document.addEventListener('visibilitychange',()=>{
    if(document.hidden&&raf){cancelAnimationFrame(raf);raf=0;}
    else run();
  });
  reduced.addEventListener('change',()=>{
    if(reduced.matches){setPlay(false);progress=timing.stops[getPhase()];draw();}
    setPlay(playing);
  });
  setExample();
  setPlay(playing);
  document.fonts.ready.then(measureRoutes);
  widget.dataset.ready = 'true';
})();
