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
        ? `Raw EMG fits this throughput at ${n} electrodes (up to 42); tokens fit up to 2,800. Payload only.`
        : 'Raw EMG exceeds this throughput above 42 electrodes; tokens fit up to 2,800. Payload only.';
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
    // Smaller screens at least 500 px tall (phones held upright, tablets) get the same slides as a
    // deck: one slide per screen, its content (wrapped in .fit) scaled down to fit, moved by the
    // buttons or a swipe.
    const desktop = matchMedia('(min-width: 900px) and (min-height: 640px)');
    const motion = matchMedia('(prefers-reduced-motion: reduce)');
    const root = document.documentElement, header = $('.site-header');
    const prev = $('#slide-prev'), next = $('#slide-next');
    const deck = () => root.classList.contains('deck');
    const head = () => desktop.matches || deck() ? header.offsetHeight : 0;     // the header stays on top (swiss.css)
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

    // Deck: wrap each slide's content once. A slide taller than the screen is laid out wider (width / s)
    // and scaled by s, so it fills the screen's width; s is the largest scale whose height still fits
    // (binary search, because wrapping and figure sizes change with the width).
    const fits = [];
    const fitOne = box => {
      const slide = box.parentElement, style = getComputedStyle(slide);
      const room = slide.clientHeight - parseFloat(style.paddingTop) - parseFloat(style.paddingBottom);
      const width = slide.clientWidth;
      box.style.width = box.style.transform = '';
      let scale = 1;
      if (box.offsetHeight > room) {
        let lo = .3, hi = 1;
        for (let k = 0; k < 8; k++) {
          const s = (lo + hi) / 2;
          box.style.width = `${width / s}px`;
          if (box.offsetHeight * s <= room) lo = s; else hi = s;
        }
        box.style.width = `${width / lo}px`;
        scale = lo;
      }
      // centre the (scaled) content in the room between the header and the slide buttons
      const lift = Math.max(0, (room - box.offsetHeight * scale) / 2);
      box.style.transform = `translateY(${lift}px) scale(${scale})`;
    };
    const fit = () => fits.forEach(fitOne);
    // re-fit a slide whose content changes size (figures load their data late); the fitted width
    // reports one more change, which yields the same result and ends there
    const sizer = 'ResizeObserver' in window ? new ResizeObserver(entries => entries.forEach(e => fitOne(e.target))) : null;
    const setMode = () => {
      const on = !desktop.matches && innerHeight >= 500;   // landscape phones keep the scrolling page
      if (on === deck()) return;
      const i = current();
      root.classList.toggle('deck', on);
      if (on) slides.forEach(slide => {
        const box = document.createElement('div');
        box.className = 'fit';
        box.append(...slide.childNodes);
        slide.append(box);
        fits.push(box);
        sizer?.observe(box);
      });
      else fits.splice(0).forEach(box => { sizer?.unobserve(box); box.replaceWith(...box.childNodes); });
      scrollTo(0, top(i));
    };
    const settle = () => {
      setMode();
      if (!deck()) return ends();
      root.style.setProperty('--head', `${header.offsetHeight}px`);
      fit();
      scrollTo(0, top(current()));   // keep a whole slide in view after a rotation or resize
      ends();
    };
    addEventListener('resize', settle);
    settle();

    // Deck: a vertical or sideways swipe moves one slide (a moved touch never taps a button or link);
    // two-finger zoom and the sliders keep their touches.
    let touch = null;
    addEventListener('touchstart', e => {
      touch = deck() && e.touches.length === 1 && !e.target.closest('input, select, textarea, [role="slider"]')
        ? { x: e.touches[0].clientX, y: e.touches[0].clientY } : null;
    }, { passive: true });
    addEventListener('touchmove', e => { if (touch && e.touches.length === 1) e.preventDefault(); }, { passive: false });
    addEventListener('touchend', e => {
      if (!touch) return;
      const dx = e.changedTouches[0].clientX - touch.x, dy = e.changedTouches[0].clientY - touch.y;
      touch = null;
      if (Math.max(Math.abs(dx), Math.abs(dy)) < 50) return;
      go(Math.abs(dy) >= Math.abs(dx) ? (dy < 0 ? 1 : -1) : (dx < 0 ? 1 : -1));
    });
    addEventListener('keydown', e => {
      if (!(desktop.matches || deck()) || e.defaultPrevented || e.altKey || e.ctrlKey || e.metaKey) return;
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
