(() => {
  // 04 Representations: SPAR patches on forearm grid 1 (static/data/spar-patch.json, scripts/export_spar_patch.py).
  const widget = document.querySelector('.patch-widget');
  if (!widget) return;
  const $ = s => widget.querySelector(s), $$ = s => [...widget.querySelectorAll(s)];
  const reduced = matchMedia('(prefers-reduced-motion: reduce)');
  const input = $('#patch-electrode'), play = $('.umap-play');
  const S = 40, STEP = 650;   // electrode spacing in drawing units (circles have radius 12); ms per electrode
  let data = null, loading = false, centre = 28, playing = !reduced.matches, visible = false, timer = 0, cells = [];

  const place = e => ({r: Math.floor((e - 1) / 8), c: (e - 1) % 8});
  const point = e => { const p = place(e); return [p.c * S, p.r * S]; };
  const ring = g => `M${g[0] - 12} ${g[1]}a12 12 0 1 0 24 0a12 12 0 1 0 -24 0`;
  let sites = '';
  for (let e = 1; e <= 64; e++) sites += ring(point(e));
  $('[data-sites]').setAttribute('d', sites);

  // Straight arrow between two electrodes, ending in a filled head.
  function arrow(p, q) {
    const dx = q[0] - p[0], dy = q[1] - p[1], len = Math.hypot(dx, dy), ux = dx / len, uy = dy / len;
    const tip = [q[0] - ux * 13, q[1] - uy * 13], base = [tip[0] - ux * 10, tip[1] - uy * 10];
    return {line: `M${p[0] + ux * 12} ${p[1] + uy * 12}L${base[0]} ${base[1]}`,
      head: `M${tip[0]} ${tip[1]}L${base[0] - uy * 5} ${base[1] + ux * 5}L${base[0] + uy * 5} ${base[1] - ux * 5}Z`};
  }
  // Dashed arc from a missing neighbour's position g to its substitute t, hugging the patch on
  // side [sx, sy], in the gap between electrode rows or columns; the head points into t.
  function hook(g, t, [sx, sy]) {
    const at = (p, k) => `${p[0] + sx * k} ${p[1] + sy * k}`, end = [t[0] + sx * 22, t[1] + sy * 22];
    return {line: `M${at(g, 12)}C${at(g, 20)} ${at(t, 24)} ${at(t, 22)}`,
      head: `M${at(t, 13)}L${end[0] - sy * 5} ${end[1] + sx * 5}L${end[0] + sy * 5} ${end[1] - sx * 5}Z`};
  }
  // Auxiliary link: curved into the grid when a straight line would cross another patch member.
  function auxiliary(ce, ae) {
    const C = point(ce), A = point(ae), cr = place(ce), ar = place(ae), dr = ar.r - cr.r, dc = ar.c - cr.c;
    if ((dr === 0 || dc === 0) && Math.abs(dr + dc) === 2) {
      const n = dr === 0 ? [0, cr.r < 4 ? 1 : -1] : [cr.c < 4 ? 1 : -1, 0];
      const k = [(C[0] + A[0]) / 2 + n[0] * 34, (C[1] + A[1]) / 2 + n[1] * 34];
      const from = p => { const vx = k[0] - p[0], vy = k[1] - p[1], l = Math.hypot(vx, vy); return `${p[0] + vx / l * 12} ${p[1] + vy / l * 12}`; };
      return `M${from(C)}Q${k[0]} ${k[1]} ${from(A)}`;
    }
    const len = Math.hypot(A[0] - C[0], A[1] - C[1]), ux = (A[0] - C[0]) / len, uy = (A[1] - C[1]) / len;
    return `M${C[0] + ux * 12} ${C[1] + uy * 12}L${A[0] - ux * 12} ${A[1] - uy * 12}`;
  }

  function draw() {
    const info = data.electrodes[centre - 1], [ce, ae, he, ve] = info.members;
    const C = point(ce), H = point(he), V = point(ve);
    let edges = '', heads = '', sub = '', ghosts = '';
    for (const q of [H, V]) { const a = arrow(C, q); edges += a.line; heads += a.head; }
    const ghost = (name, g) => {
      const label = $(`[data-ghost=${name}]`);
      label.style.display = g ? '' : 'none';
      if (g) { label.setAttribute('x', g[0]); label.setAttribute('y', g[1]); ghosts += ring(g); }
    };
    // A missing neighbour: a ghost at its position and a dashed arc to the substitute, on the side
    // away from the solid neighbour arrows.
    if (!info.right) { const hk = hook([C[0] + S, C[1]], H, info.down ? [0, -1] : [0, 1]); sub += hk.line; heads += hk.head; }
    if (!info.down) { const hk = hook([C[0], C[1] + S], V, info.right ? [-1, 0] : [1, 0]); sub += hk.line; heads += hk.head; }
    ghost('h', info.right ? null : [C[0] + S, C[1]]);
    ghost('v', info.down ? null : [C[0], C[1] + S]);
    for (const [key, value] of [['ghosts', ghosts], ['aux', auxiliary(ce, ae)], ['sub', sub], ['edges', edges], ['heads', heads]]) $(`[data-${key}]`).setAttribute('d', value);
    const labels = {h: info.right ? 'h' : 'h′', v: info.down ? 'v' : 'v′'};
    for (const [key, e] of [['c', ce], ['a', ae], ['h', he], ['v', ve]]) {
      const [x, y] = point(e);
      $(`[data-member=${key}]`).setAttribute('transform', `translate(${x} ${y})`);
      $(`[data-n=${key}]`).textContent = e;
    }
    for (const key of ['h', 'v']) {
      $(`[data-member=${key}] text`).textContent = labels[key];
      $(`[data-l=${key}]`).textContent = labels[key];
    }
    const p = place(ce);
    $('[data-centre]').textContent = ce;
    $('[data-place]').textContent = `Row ${p.r + 1}, column ${p.c + 1}`;
    let status = 'Right and lower neighbours present: both couplings are kept in every band.';
    if (!info.right && !info.down) status = 'Corner: h′ and v′ replace both missing neighbours, and both couplings are set to zero.';
    else if (!info.right) status = 'No right neighbour: h′ is the left neighbour, and the horizontal couplings are set to zero.';
    else if (!info.down) status = 'No lower neighbour: v′ is the upper neighbour, and the vertical couplings are set to zero.';
    if (p.c === 0 || ae !== ce - 1) status += ' The auxiliary is the nearest unused electrode.';
    $('[data-status]').textContent = status;
    cells.forEach(({cell, bar, text, band, k}) => {
      const zeroed = (k === 1 && !info.right) || (k === 2 && !info.down), v = info.features[3 * band + k];
      const w = Math.min(Math.abs(v), 4) / 4 * 23;   // bar zone: left half of the cell, zero at 25 %
      cell.classList.toggle('zero', zeroed);
      text.textContent = zeroed ? '0, zeroed' : `${v < 0 ? '−' : '+'}${Math.abs(v).toFixed(2)}`;
      bar.style.left = `${v < 0 ? 25 - w : 25}%`;
      bar.style.width = zeroed ? '0' : `${w}%`;
    });
    $('[data-token]').textContent = info.token;
    $('[data-bits]').textContent = info.token.toString(2).padStart(10, '0');
    input.value = ce;
    input.setAttribute('aria-valuetext', `Electrode ${ce} of 64`);
    $('[data-of]').textContent = `${ce}/64`;
    $$('[data-electrode]').forEach(b => b.setAttribute('aria-pressed', Number(b.dataset.electrode) === ce));
    $('.patch-grid').setAttribute('aria-label', `Electrode ${ce} patch: horizontal neighbour ${he}${info.right ? '' : ' (substituted)'}, vertical neighbour ${ve}${info.down ? '' : ' (substituted)'}, auxiliary ${ae}`);
  }

  function schedule() {
    clearTimeout(timer);
    if (data && playing && visible && !document.hidden) timer = setTimeout(() => { centre = centre % 64 + 1; draw(); schedule(); }, STEP);
  }
  function setPlay(next) {
    playing = next;
    play.setAttribute('aria-label', playing ? 'Pause the patch animation' : 'Play the patch animation');
    play.firstElementChild.textContent = playing ? 'Ⅱ' : '▷';
    schedule();
  }
  function show(e) { if (!data) return; setPlay(false); centre = e; draw(); }

  function load() {
    loading = true;
    fetch('static/data/spar-patch.json')
      .then(response => { if (!response.ok) throw new Error(response.status); return response.json(); })
      .then(json => {
        data = json;
        const table = $('[data-table]');
        data.bands.forEach((band, b) => {
          const label = document.createElement('span');
          label.className = 'patch-band';
          label.textContent = band;
          table.append(label);
          for (let k = 0; k < 3; k++) {
            const cell = document.createElement('span'), bar = document.createElement('b'), text = document.createElement('span');
            cell.className = `patch-cell${k ? '' : ' centre'}`;
            cell.append(bar, text);
            table.append(cell);
            cells.push({cell, bar, text, band: b, k});
          }
        });
        widget.dataset.ready = 'true';
        draw();
        schedule();
      })
      .catch(() => { $('[data-status]').textContent = 'The recorded update could not be loaded.'; });
  }

  $$('[data-electrode]').forEach(button => button.addEventListener('click', () => show(Number(button.dataset.electrode))));
  play.addEventListener('click', () => setPlay(!playing));
  input.addEventListener('input', () => show(Number(input.value)));
  new IntersectionObserver(entries => {
    visible = entries[entries.length - 1].isIntersecting;
    if (visible && !loading) load();
    schedule();
  }, {rootMargin: '300px 0px'}).observe(widget);
  document.addEventListener('visibilitychange', schedule);
  setPlay(playing);
})();
