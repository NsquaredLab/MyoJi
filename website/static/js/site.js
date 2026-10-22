(() => {
  // Hero: the token grid (recorded SPAR tokens, static/data/token-stream.json) and the payload bars
  // share one electrode count. On load the grid grows from 16 to 320 electrodes; the tokens then
  // advance one recorded update every 200 ms (10x slower than real time).
  const $ = s => document.querySelector(s);
  const slider = $('#channels'), grid = $('#token-grid'), play = $('#grid-play');
  if (slider && grid) {
    const rawMax = 320 * 16 * 2048, link = 1.4e6;
    const fmt = x => Math.round(x).toLocaleString('en-US');
    const rate = bits => bits >= 1e6 ? `${(bits / 1e6).toFixed(2)} Mbit/s` : `${Math.round(bits / 1e3)} kbit/s`;
    const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;
    const cells = [];
    let data = null, n = reduced ? 320 : 16, t = 49, playing = !reduced, growing = !reduced, visible = true;

    const layout = () => {
      const cols = Math.ceil(Math.sqrt(n));
      const box = Math.min(grid.parentElement.clientWidth, grid.parentElement.clientHeight);
      const side = box * (140 + 400 * Math.sqrt(n / 320)) / 560;  // 540 px of 560 at 320 electrodes
      const cell = Math.floor((side - (cols - 1) * 2) / cols);
      while (cells.length < n) cells.push(grid.appendChild(document.createElement('span')));
      while (cells.length > n) cells.pop().remove();
      grid.style.gridTemplateColumns = `repeat(${cols}, ${cell}px)`;
      grid.style.gridAutoRows = `${cell}px`;
      grid.style.fontSize = cell >= 20 ? `${Math.max(8, Math.round(cell * .34))}px` : '0';
      cells[27]?.classList.add('mark');                 // electrode 28, the tokenizer figure's example
    };
    const paint = () => {
      if (!data) return;
      const ids = data.tokens[t], act = data.activity[t];
      cells.forEach((c, i) => {
        c.textContent = ids[i];
        c.style.setProperty('--a', act[i] / 9);
        c.classList.toggle('dark', act[i] >= 5);
      });
      $('#tok28').textContent = ids[27];
      $('#bits28').textContent = ids[27].toString(2).padStart(10, '0');
      $('#update-no').textContent = `Update ${t + 1} of 100`;
    };
    const setN = next => {
      n = next;
      slider.value = n;
      const raw = n * 16 * 2048, packed = n * 10 * 50;
      $('#channel-count').textContent = $('[data-n]').textContent = n;
      $('#raw-bits').textContent = fmt(raw * .02);
      $('#tok-bits').textContent = fmt(n * 10);
      $('#raw-rate').textContent = rate(raw);
      $('#packed-rate').textContent = rate(packed);
      $('#raw-fill').style.width = `${(100 * raw / rawMax).toFixed(2)}%`;
      $('#packed-fill').style.width = `${Math.max(.3, 100 * packed / rawMax).toFixed(2)}%`;
      $('#fit-text').textContent = raw <= link
        ? `Raw EMG fits the assumed link at ${n} electrodes (up to 42); tokens fit up to 2,800. Payload only.`
        : 'Raw EMG exceeds the assumed link above 42 electrodes; tokens fit up to 2,800. Payload only.';
      layout();
      paint();
    };
    const setPlaying = on => {
      playing = on;
      play.textContent = on ? 'Pause' : 'Play';
      play.setAttribute('aria-pressed', String(on));
    };

    slider.addEventListener('input', () => { growing = false; setN(Number(slider.value)); });
    play.addEventListener('click', () => setPlaying(!playing));
    new IntersectionObserver(([entry]) => { visible = entry.isIntersecting; }).observe(grid);
    addEventListener('resize', layout);
    setInterval(() => {
      if (!visible || document.hidden) return;
      if (growing) { setN(Math.min(320, n + 16)); growing = n < 320; }
      if (playing && data) { t = (t + 1) % 100; paint(); }
    }, 200);
    fetch('static/data/token-stream.json').then(r => r.json()).then(d => { data = d; paint(); });
    setPlaying(playing);
    setN(n);
  }

  // Slides: count the chapter on screen, and mark each slide once it is reached so its content
  // fades in (and the error bars grow). The CSS only hides content when this class is set.
  const slides = [...document.querySelectorAll('.slide')];
  if (slides.length && 'IntersectionObserver' in window) {
    document.documentElement.classList.add('slides-js');
    const total = String(slides.length).padStart(2, '0');
    const seen = new IntersectionObserver(entries => entries.forEach(entry => {
      if (!entry.isIntersecting) return;
      entry.target.classList.add('in');
      const i = slides.indexOf(entry.target) + 1;
      $('#slide-no').textContent = `${String(i).padStart(2, '0')} / ${total}`;
    }), { threshold: .15 });
    slides.forEach(slide => seen.observe(slide));

    // Desktop: page scrolling is off (swiss.css); the buttons and paging keys glide one slide.
    // A slide taller than the window (the intro on short screens) first shows its remainder.
    const desktop = matchMedia('(min-width: 900px) and (min-height: 640px)');
    const motion = matchMedia('(prefers-reduced-motion: reduce)');
    const prev = $('#slide-prev'), next = $('#slide-next');
    const head = () => desktop.matches ? $('.site-header').offsetHeight : 0;     // the header stays on top (swiss.css)
    const top = i => i ? slides[i].getBoundingClientRect().top + scrollY - head() : 0;   // the first slide includes the header
    const current = () => slides.reduce((best, _, i) => Math.abs(top(i) - scrollY) < Math.abs(top(best) - scrollY) ? i : best, 0);
    const glide = y => scrollTo({ top: y, behavior: motion.matches ? 'auto' : 'smooth' });
    const go = step => {
      const i = current(), r = slides[i].getBoundingClientRect(), start = i ? r.top - head() : -scrollY;
      if (step > 0 && r.bottom > innerHeight + 2) return glide(scrollY + Math.min(innerHeight, r.bottom - innerHeight));
      if (step < 0 && start < -2) return glide(scrollY + Math.max(-innerHeight, start));
      glide(top(Math.min(slides.length - 1, Math.max(0, i + step))));
    };
    const ends = () => {
      prev.disabled = scrollY < 2;
      next.disabled = scrollY + innerHeight > document.documentElement.scrollHeight - 2;
    };
    prev.addEventListener('click', () => go(-1));
    next.addEventListener('click', () => go(1));
    addEventListener('scroll', ends, { passive: true });
    addEventListener('resize', ends);
    ends();
    addEventListener('keydown', e => {
      if (!desktop.matches || e.defaultPrevented || e.altKey || e.ctrlKey || e.metaKey) return;
      if (!e.target.closest?.('.slide-nav') && e.target.closest?.('input, textarea, select, button, [contenteditable], [tabindex]')) return;
      const step = { ArrowDown: 1, PageDown: 1, ' ': e.shiftKey ? -1 : 1, ArrowUp: -1, PageUp: -1 }[e.key];
      if (!step) return;
      e.preventDefault();
      go(step);
    });
  }

  // Paper: copy the citation as plain text (the colour spans drop out of textContent).
  $('#copy-citation')?.addEventListener('click', async event => {
    const button = event.currentTarget;
    const code = $('#bibtex');
    try {
      await navigator.clipboard.writeText(code.textContent);
      button.textContent = 'Copied';
      $('#copy-status').textContent = 'BibTeX copied to clipboard.';
    } catch {
      const range = document.createRange();
      range.selectNodeContents(code);
      window.getSelection().removeAllRanges();
      window.getSelection().addRange(range);
      $('#copy-status').textContent = 'Citation selected. Press Control+C or Command+C to copy.';
    }
  });
})();
