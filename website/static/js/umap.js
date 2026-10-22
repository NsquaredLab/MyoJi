(() => {
  const widget = document.querySelector('.umap-widget');
  if (!widget) return;
  const data = JSON.parse(document.querySelector('#umap-data').textContent);
  const $ = s => widget.querySelector(s), $$ = s => [...widget.querySelectorAll(s)];
  const canvas = $('.umap-canvas'), ctx = canvas.getContext('2d');
  const cloud = document.createElement('canvas');
  const reduced = matchMedia('(prefers-reduced-motion: reduce)');
  let history = 480, phase = .25, selected = -1, playing = !reduced.matches;
  let width = 1, height = 1, dpr = 1, visible = false, raf = 0, last = 0;
  const dark = () => document.documentElement.dataset.theme === 'dark';
  const colors = isDark => data.colors.map(hex => {
    if (!isDark) return hex;
    return `rgb(${[1,3,5].map(i=>Math.round(parseInt(hex.slice(i,i+2),16)*.72+255*.28)).join(',')})`;
  });
  const palette = isDark => {
    const fixed = {paper:isDark?'#181818':'#fafaf8', ink:isDark?'#ecece8':'#202020', muted:isDark?'#acaca6':'#616161', line:isDark?'#41413d':'#d1d1cc'};
    if (isDark !== dark()) return fixed;   // video frames for the other theme keep the fixed palette
    const css = getComputedStyle(document.documentElement);   // follow the host page's colours
    return Object.fromEntries(Object.entries(fixed).map(([key, value]) => [key, css.getPropertyValue(`--${key}`).trim() || value]));
  };
  function projection(view, box) {
    const [x0,y0,x1,y1] = view.bounds;
    const scale = Math.min(box.w/(x1-x0),box.h/(y1-y0));
    const x = box.x+(box.w-(x1-x0)*scale)/2, y = box.y+(box.h-(y1-y0)*scale)/2;
    return p => [x+(p[0]-x0)*scale,y+(y1-p[1])*scale];
  }
  function path(context, points, project) {
    context.beginPath();
    points.forEach((p,i)=>{const q=project(p);if(i)context.lineTo(...q);else context.moveTo(...q);});
  }
  function point(route, p) {
    const n=route.length-1, t=Math.max(0,Math.min(1,p))*n, i=Math.min(n-1,Math.floor(t)), f=t-i;
    return [route[i][0]*(1-f)+route[i+1][0]*f,route[i][1]*(1-f)+route[i+1][1]*f];
  }
  function drawCloud(context, box, ms, isDark, focus=-1, scale=1) {
    const view=data.windows[ms],project=projection(view,box),inks=colors(isDark);
    for(let c=0;c<8;c++){
      context.fillStyle=inks[c];context.globalAlpha=focus<0||focus===c ? .21 : .035;
      context.beginPath();
      view.points.forEach((p,i)=>{if(data.labels[i]!==c)return;const [x,y]=project(p),r=1.1*scale;context.moveTo(x+r,y);context.arc(x,y,r,0,Math.PI*2);});
      context.fill();
    }
    context.globalAlpha=1;
  }
  function drawRoutes(context, box, ms, p, isDark, focus=-1, scale=1) {
    const view=data.windows[ms],project=projection(view,box),inks=colors(isDark),theme=palette(isDark);
    view.routes.forEach((route,c)=>{
      const active=focus<0||focus===c;
      context.lineJoin='round';context.lineCap='round';context.strokeStyle=inks[c];
      context.globalAlpha=active?.32:.06;context.lineWidth=1.25*scale;
      path(context,route,project);context.stroke();
      if(!active)return;
      const end=Math.floor(p*(route.length-1));
      context.globalAlpha=.9;context.lineWidth=2.5*scale;
      path(context,[...route.slice(0,end+1),point(route,p)],project);context.stroke();
      const q=project(point(route,p)),start=project(route[0]);
      context.fillStyle=theme.paper;context.globalAlpha=1;
      context.beginPath();context.arc(...q,6.5*scale,0,Math.PI*2);context.fill();
      context.fillStyle=inks[c];context.beginPath();context.arc(...q,4.2*scale,0,Math.PI*2);context.fill();
      context.strokeStyle=inks[c];context.lineWidth=1*scale;context.beginPath();context.arc(...start,2.5*scale,0,Math.PI*2);context.stroke();
    });
    context.globalAlpha=1;
    // Names occupy the centres of the groups; a background stroke keeps them
    // readable over dense points without changing the scientific coordinates.
    context.textAlign='center';context.textBaseline='middle';
    context.font=`500 ${Math.max(12,14*scale)}px "Switzer Variable",Arial,sans-serif`;
    view.centres.forEach((centre,c)=>{
      if(focus>=0&&focus!==c)return;
      const [x,y]=project(centre);
      context.lineWidth=5*scale;context.strokeStyle=theme.paper;context.strokeText(data.names[c],x,y);
      context.fillStyle=theme.ink;context.fillText(data.names[c],x,y);
    });
    context.textAlign='left';context.textBaseline='alphabetic';
  }
  function rebuild() {
    const rect=canvas.getBoundingClientRect();
    width=rect.width;height=rect.height;dpr=Math.min(devicePixelRatio||1,2);
    if(!width||!height)return;
    canvas.width=cloud.width=Math.round(width*dpr);canvas.height=cloud.height=Math.round(height*dpr);
    const cache=cloud.getContext('2d');cache.setTransform(dpr,0,0,dpr,0,0);
    drawCloud(cache,{x:0,y:10,w:width,h:height-20},history,dark(),selected,width<600?.8:1);
    draw();
  }
  function draw() {
    ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,width,height);
    ctx.drawImage(cloud,0,0,width,height);
    drawRoutes(ctx,{x:0,y:10,w:width,h:height-20},history,phase,dark(),selected,width<600?.9:1);
    const percent=Math.round(phase*100);
    $('[data-phase-number]').textContent=percent;
    $('#umap-progress').value=Math.round(phase*1000);
    $('#umap-progress').setAttribute('aria-valuetext',`${percent} percent of repetition`);
    canvas.setAttribute('aria-label',`${selected<0?'Eight movements':data.names[selected]}, ${history?`${history} milliseconds of token history`:'single update'}, mean repetition phase ${percent} percent`);
  }
  function setPlay(next) {
    playing=next&&!reduced.matches;
    $('.umap-play').dataset.playing=playing;
    $('.umap-play').setAttribute('aria-label',reduced.matches?'Next repetition phase':playing?'Pause movement animation':'Play movement animation');
    $('.umap-play span').textContent=reduced.matches?'→':playing?'Ⅱ':'▷';
    if(!playing&&raf){cancelAnimationFrame(raf);raf=0;}
    run();
  }
  function seek(p) {setPlay(false);phase=Math.max(0,Math.min(1,p));draw();}
  function setHistory(ms) {
    if(!data.windows[ms])throw new Error('Unknown saved history window');
    history=Number(ms);
    $$('[data-history]').forEach(b=>b.setAttribute('aria-pressed',Number(b.dataset.history)===history));
    $('[data-umap-announcement]').textContent=history?`${history} milliseconds of token history`:'Single update';
    rebuild();
  }
  function setSelected(value) {
    selected=Number(value);
    $$('[data-movement]').forEach(b=>b.setAttribute('aria-pressed',Number(b.dataset.movement)===selected));
    $('.umap-selection').textContent=selected<0?'All movements':data.names[selected];
    rebuild();
  }
  function tick(now) {
    const dt=last?Math.min((now-last)/1000,.05):0;last=now;
    phase=(phase+dt/8)%1;draw();
    raf=playing&&visible&&!document.hidden?requestAnimationFrame(tick):0;
  }
  function run(){if(!raf&&playing&&visible&&!document.hidden){last=0;raf=requestAnimationFrame(tick);}}
  $('.umap-play').addEventListener('click',()=>reduced.matches?seek((Math.floor(phase*4+1)%5)/4):setPlay(!playing));
  $('#umap-progress').addEventListener('pointerdown',()=>setPlay(false));
  $('#umap-progress').addEventListener('keydown',()=>setPlay(false));
  $('#umap-progress').addEventListener('input',event=>seek(Number(event.target.value)/1000));
  $$('[data-history]').forEach(button=>button.addEventListener('click',()=>setHistory(button.dataset.history)));
  $$('[data-movement]').forEach(button=>button.addEventListener('click',()=>setSelected(button.dataset.movement)));
  $('.umap-details').addEventListener('toggle',()=>{if($('.umap-details').open)setPlay(false);});
  new ResizeObserver(rebuild).observe(canvas);
  new MutationObserver(rebuild).observe(document.documentElement,{attributes:true,attributeFilter:['data-theme']});
  new IntersectionObserver(entries=>{visible=entries[entries.length-1].isIntersecting;if(!visible&&raf){cancelAnimationFrame(raf);raf=0;}if(visible)run();},{rootMargin:'80px 0px'}).observe(widget);
  document.addEventListener('visibilitychange',()=>{if(document.hidden&&raf){cancelAnimationFrame(raf);raf=0;}else run();});
  reduced.addEventListener('change',()=>{if(reduced.matches)setPlay(false);else setPlay(playing);});
  // Deterministic, UI-free slide rendering uses exactly the same points/routes.
  // The exporter supplies phase explicitly; no video frame depends on wall time.
  function presentation(p, ms=480, isDark=false) {
    const frame=document.createElement('canvas');frame.width=1920;frame.height=1080;
    const c=frame.getContext('2d'),theme=palette(isDark),inks=colors(isDark);
    c.fillStyle=theme.paper;c.fillRect(0,0,1920,1080);
    c.fillStyle=theme.ink;c.font='540 62px "Switzer Variable"';c.fillText('Movement through token space',72,102);
    c.font='400 25px "Switzer Variable"';c.fillStyle=theme.muted;
    c.fillText(`${ms?`${ms} ms of token history`:'Single update'}  ·  Participant 1  ·  320 electrodes`,74,150);
    c.strokeStyle=theme.line;c.lineWidth=1;c.beginPath();c.moveTo(72,180);c.lineTo(1848,180);c.stroke();
    const box={x:90,y:192,w:1230,h:710};
    drawCloud(c,box,ms,isDark,-1,1.5);drawRoutes(c,box,ms,p,isDark,-1,1.8);
    c.fillStyle=theme.muted;c.font='400 24px "Switzer Variable"';c.fillText('Repetition phase',1440,350);
    c.fillStyle=theme.ink;c.font='500 124px "Switzer Variable"';c.fillText(`${Math.round(p*100)}%`,1432,490);
    c.font='400 26px "Switzer Variable"';c.fillText('Phase-aligned mean routes',1440,565);
    c.fillStyle=theme.muted;c.font='400 23px "Switzer Variable"';c.fillText('19 repetitions per movement',1440,609);
    c.fillText('Playback slowed',1440,645);
    data.names.forEach((name,i)=>{const x=74+i*220;c.fillStyle=inks[i];c.beginPath();c.arc(x,934,5,0,Math.PI*2);c.fill();c.fillStyle=theme.ink;c.font='400 23px "Switzer Variable"';c.fillText(name,x+15,941);});
    c.strokeStyle=theme.line;c.lineWidth=3;c.beginPath();c.moveTo(72,984);c.lineTo(1848,984);c.stroke();
    c.strokeStyle=theme.ink;c.beginPath();c.moveTo(72,984);c.lineTo(72+1776*p,984);c.stroke();
    c.fillStyle=theme.ink;c.beginPath();c.arc(72+1776*p,984,7,0,Math.PI*2);c.fill();
    c.fillStyle=theme.muted;c.font='400 22px "Switzer Variable"';c.fillText('0%',72,1034);c.textAlign='center';c.fillText('50%',960,1034);c.textAlign='right';c.fillText('100%',1848,1034);
    return frame.toDataURL('image/png');
  }
  window.emgUmap={seek,setHistory,setSelected,presentation,getState:()=>({history,phase,selected,playing,visible,frames:data.labels.length})};
  document.fonts.ready.then(()=>{rebuild();widget.dataset.ready='true';});
  setPlay(playing);rebuild();
})();
