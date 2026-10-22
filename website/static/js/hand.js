(() => {
  // 02 Results: measured and decoded hands of participant 6 (static/data/hand.json, scripts/export_hand.py).
  const widget = document.querySelector('.hand-widget');
  if (!widget) return;
  const $ = s => widget.querySelector(s), $$ = s => [...widget.querySelectorAll(s)];
  const reduced = matchMedia('(prefers-reduced-motion: reduce)');
  const input = $('#hand-time'), play = $('.umap-play');
  const traces = [];
  let data = null, loading = false, move = Number(widget.dataset.movement || 0), t = 0, playing = !reduced.matches, visible = false, raf = 0, last = 0, hold = 0;

  // Landmarks at playback position t (recorded updates, 0 to 199), drawn between the two updates around it.
  function pose(values, at) {
    const k = Math.min(Math.floor(at), 198), a = at - k, o = k * 40, p = new Array(40);
    for (let i = 0; i < 40; i++) p[i] = (values[o + i] + a * (values[o + 40 + i] - values[o + i])).toFixed(1);
    return p;
  }
  const wrist = () => `M${data.wrist[0]} ${data.wrist[1]}`;
  const bones = p => data.chains.map(chain => wrist() + chain.map(j => `L${p[2 * j]} ${p[2 * j + 1]}`).join('')).join('');
  function joints(p) {
    let d = wrist() + 'h0';
    for (let j = 0; j < 20; j++) d += `M${p[2 * j]} ${p[2 * j + 1]}h0`;
    return d;
  }

  // Fig. 5A-style trace of the movement's fingertip distance, with two to five round ticks.
  function trace(m) {
    const all = m.trace.measured.concat(m.trace.tokens, m.trace.continuous);
    let lo = Math.min(...all), hi = Math.max(...all);
    const pad = .12 * (hi - lo);
    lo -= pad; hi += pad;
    const y = v => (150 * (hi - v) / (hi - lo)).toFixed(1);
    const line = vs => vs.map((v, i) => `${i ? 'L' : 'M'}${(i * 1000 / 199).toFixed(1)} ${y(v)}`).join('');
    const span = hi - lo, base = 10 ** Math.floor(Math.log10(span / 5));
    const step = [1, 2, 5, 10].map(k => k * base).find(s => span / s <= 5);
    const ticks = [];
    for (let v = Math.ceil(lo / step) * step; v <= hi; v += step) ticks.push({label: String(Math.round(v)), y: y(v), top: `${(100 * (hi - v) / (hi - lo)).toFixed(2)}%`});
    return {measured: line(m.trace.measured), tokens: line(m.trace.tokens), continuous: line(m.trace.continuous),
      ticks, grid: ticks.map(tick => `M0 ${tick.y}H1000`).join('')};
  }

  function draw() {
    const m = data.movements[move], measured = pose(m.measured, t);
    $$('[data-bones=measured]').forEach(path => path.setAttribute('d', bones(measured)));
    $$('[data-joints]').forEach(path => path.setAttribute('d', joints(measured)));
    for (const arm of ['tokens', 'continuous']) $(`[data-bones=${arm}]`).setAttribute('d', bones(pose(m[arm], t)));
    $('[data-playhead]').setAttribute('d', `M${(t * 1000 / 199).toFixed(1)} 0V150`);
    const seconds = (t * .02).toFixed(2);
    input.value = Math.round(t);
    input.setAttribute('aria-valuetext', `${seconds} seconds`);
    $('[data-time]').textContent = `${seconds} s`;
  }

  function setMove(index) {
    move = index;
    const m = data.movements[move], tr = traces[move] ??= trace(m);
    $$('[data-move]').forEach(b => b.setAttribute('aria-pressed', Number(b.dataset.move) === move));
    $('[data-move-name]').textContent = m.name;
    $('[data-measure]').textContent = m.measure;
    for (const arm of ['tokens', 'continuous']) $(`[data-error=${arm}]`).textContent = m.error_mm[arm].toFixed(1);
    for (const kind of ['measured', 'tokens', 'continuous']) $(`[data-trace=${kind}]`).setAttribute('d', tr[kind]);
    $('[data-grid]').setAttribute('d', tr.grid);
    $('[data-ticks]').replaceChildren(...tr.ticks.map(tick => {
      const label = document.createElement('span');
      label.style.top = tick.top;
      label.textContent = tick.label;
      return label;
    }));
    $$('.hand-svg').forEach((svg, i) => svg.setAttribute('aria-label', `${m.name}: hand decoded from ${i ? 'continuous SPAR' : 'SPAR tokens'} over the measured hand`));
    draw();
  }

  function tick(now) {
    const dt = last ? Math.min(now - last, 100) : 0;   // ms; a hidden tab resumes without a jump
    last = now;
    if (t >= 199) {                                    // hold the last update, then loop
      hold += dt;
      if (hold > 700) { hold = 0; t = 0; }
    } else {
      hold = 0;
      t = Math.min(199, t + dt / 20);                  // one recorded update every 20 ms
    }
    draw();
    raf = playing && visible && !document.hidden ? requestAnimationFrame(tick) : 0;
  }
  function run() { if (!raf && data && playing && visible && !document.hidden) { last = 0; raf = requestAnimationFrame(tick); } }
  function stop() { if (raf) { cancelAnimationFrame(raf); raf = 0; } }
  function setPlay(next) {
    playing = next;
    play.setAttribute('aria-label', playing ? 'Pause the hand animation' : 'Play the hand animation');
    play.firstElementChild.textContent = playing ? 'Ⅱ' : '▷';
    if (playing) { if (t >= 199) t = 0; run(); } else stop();
  }

  function load() {
    loading = true;
    fetch('static/data/hand.json')
      .then(response => { if (!response.ok) throw new Error(response.status); return response.json(); })
      .then(json => { data = json; widget.dataset.ready = 'true'; setMove(move); run(); })
      .catch(() => { $('[data-status]').textContent = 'The recorded hand poses could not be loaded.'; });
  }

  $$('[data-move]').forEach(button => button.addEventListener('click', () => data && setMove(Number(button.dataset.move))));
  play.addEventListener('click', () => setPlay(!playing));
  input.addEventListener('pointerdown', () => setPlay(false));
  input.addEventListener('keydown', () => setPlay(false));
  input.addEventListener('input', () => { if (!data) return; setPlay(false); t = Number(input.value); hold = 0; draw(); });
  // Load the poses (~0.8 MB) only when the figure comes near; animate only while it is on screen.
  new IntersectionObserver(entries => { if (entries.some(e => e.isIntersecting) && !loading) load(); }, {rootMargin: '600px 0px'}).observe(widget);
  new IntersectionObserver(entries => { visible = entries[entries.length - 1].isIntersecting; visible ? run() : stop(); }, {rootMargin: '80px 0px'}).observe(widget);
  document.addEventListener('visibilitychange', () => document.hidden ? stop() : run());
  setPlay(playing);
})();
