import {readFileSync, writeFileSync} from 'node:fs';
import {dirname, join} from 'node:path';
import {fileURLToPath} from 'node:url';
import {createHash} from 'node:crypto';
import {umapFragment} from './umap_fragment.mjs';

const root=join(dirname(fileURLToPath(import.meta.url)), '..');
const read=file=>readFileSync(join(root,file),'utf8');
const examples=JSON.parse(read('static/data/examples.json'));
const temporal=umapFragment(JSON.parse(read('static/data/umap.json')));
const initial=examples.learned;
const stages=['Window','Features','Match','Token','Lookup'];
const geometry={duration:16, boundaries:[0,.18,.4,.65,.83,1], stops:[.15,.37,.63,.82,.98]};
const wavePath=(wave,x,y,w,h)=>{
 const limit=Math.max(...wave.map(Math.abs))*1.1;
 return wave.map((v,i)=>`${i?'L':'M'}${(x+i*w/(wave.length-1)).toFixed(2)},${(y-v/limit*h/2).toFixed(2)}`).join(' ');
};
const learnedPath=wavePath(initial.waveform,12,75,196,100);
const signalSvg=`<svg class="signal-chart" viewBox="0 0 220 140" preserveAspectRatio="none" role="img" aria-label="Recorded EMG waveform over 480 milliseconds">
 <defs><clipPath id="signal-reveal"><rect data-wave-clip x="12" y="10" width="0" height="126"/></clipPath></defs>
 <path class="signal-grid" d="M12 25H208 M12 75H208 M12 125H208 M12 15V135 M77 15V135 M143 15V135 M208 15V135"/>
 <rect class="signal-window" data-window x="143" y="14" width="65" height="122" rx="3"/>
 <g data-wave-background><path d="${learnedPath}"/></g>
 <g data-wave-foreground clip-path="url(#signal-reveal)"><path d="${learnedPath}"/></g>
 <line data-scan x1="12" x2="12" y1="14" y2="136"/>
</svg><div class="signal-axis"><span data-time-start>−480 ms</span><span>now</span></div>`;
const fragment=`<div class="tokenizer-widget iso flow-widget" data-phase="0">
 <div class="widget-top"><span class="flow-meta">One update · 50 Hz</span><div class="widget-tools"><div class="method-switch" role="group" aria-label="Tokenizer representation"><button type="button" data-method="learned" aria-pressed="true">Learned encoder</button><button type="button" data-method="spar" aria-pressed="false">SPAR</button></div><button class="theme-button" data-theme-toggle type="button" aria-label="Switch to dark mode" aria-pressed="false"><span aria-hidden="true">◐</span></button></div></div>
 <div class="flow-plate iso-plate">
  <div class="pipeline">
   <svg class="flow-links" aria-hidden="true"><defs><marker id="flow-arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M1 1L7 4L1 7"/></marker></defs>${[0,1,2].map(i=>`<path data-link="${i}" marker-end="url(#flow-arrow)"/>`).join('')}<g data-feature-packet>${Array.from({length:8},(_,i)=>`<rect x="${i%4*5-9}" y="${Math.floor(i/4)*6-5}" width="4" height="5" rx="1" data-packet-cell="${i}"/>`).join('')}</g><g data-token-packet><rect x="-23" y="-14" width="46" height="28" rx="8"/><text text-anchor="middle" y="4" data-packet-token>93</text></g></svg>
   <section class="flow-panel signal-panel" data-panel="0" aria-label="EMG input">
    <div class="flow-panel-heading"><h3>Signal</h3></div>
    <p class="flow-subtitle" data-input-kind>One electrode · recorded</p>
    ${signalSvg}
    <div class="signal-support"><span data-support>480 ms context</span><span data-patch>160 ms final patch</span></div>
   </section>
   <section class="flow-panel feature-panel" data-panel="1" aria-label="Feature extraction">
    <div class="flow-panel-heading"><h3>Features</h3></div>
    <p class="flow-subtitle" data-encoder-name>Shared causal encoder</p>
    <div class="feature-field" data-feature-grid role="img" aria-label="Recorded 256-dimensional feature vector"></div>
    <div class="feature-count"><strong data-feature-count>256</strong><span>values</span></div>
   </section>
   <section class="flow-panel codebook-panel" data-panel="2" aria-label="Nearest codebook entry">
    <div class="flow-panel-heading"><h3>Codebook</h3></div>
    <p class="flow-subtitle">1,024 entries</p>
    <div class="codebook-key"><span>ID</span><span title="Squared Euclidean distance">Distance²</span></div>
    <div class="codebook-rows" role="group" aria-label="Inspect codebook entries"></div>
    <div class="winner-label"><span class="winner-dot"></span><span data-winner-label>Nearest entry: 93</span></div>
   </section>
   <section class="flow-panel output-panel" data-panel="3" aria-label="Token transmission and receiver lookup">
    <div class="flow-panel-heading"><h3>Token</h3></div>
    <p class="flow-subtitle">10-bit index</p>
    <div class="token-output"><div class="token-badge"><span class="sr-only">TOKEN ID</span><strong data-output-token>93</strong></div><span class="token-size">10 <span>bits</span></span></div>
    <div class="token-bits" aria-label="Token in binary"></div>
    <div class="receiver-lookup"><div class="receiver-label"><span aria-hidden="true">↓</span> <span data-lookup-label>Look up entry 93</span></div><div class="receiver-vector" data-receiver-grid role="img" aria-label="Retrieved codebook vector"></div><p>→ Pose decoder</p></div>
   </section>
  </div>
 </div>
 <div class="flow-transport"><button type="button" class="flow-play" aria-label="Pause animation" data-playing="true"><span data-play-icon aria-hidden="true">Ⅱ</span><span data-play-label class="sr-only">Pause</span></button><div class="flow-timeline"><label for="flow-progress" class="sr-only">Scrub the transformation</label><input id="flow-progress" type="range" min="0" max="1000" step="1" value="0" aria-valuetext="Window: the EMG context"><div class="flow-steps" role="group" aria-label="Transformation stages">${stages.map((name,i)=>`<button type="button" data-step="${i}" aria-pressed="${i===0}">${name}</button>`).join('')}</div></div><span class="flow-speed" title="Replays one recorded update in slow motion">Slowed</span><span data-progress-label class="sr-only">01 / Window</span></div>
 <div class="flow-caption-row"><p data-readout>256 values → token 93 → 10 bits</p><span>Drag to explore</span></div>
 <details class="flow-more"><summary>Details <span>+</span></summary>
 <div class="flow-explanation"><p class="step-kicker" data-step-kicker>01 / WINDOW</p><div><h3 data-step-title>Start with a short history of muscle activity.</h3><p data-step-copy></p></div><div class="flow-equation"><span data-equation-main>480 ms</span><span data-equation-note>of causal signal history</span></div></div>
 <details class="flow-inspector"><summary>Compare the feature vectors <span>+</span></summary><div class="inspector-body"><div><div class="panel-label">INSPECTED CODEBOOK ENTRY</div><div class="live-value" data-inspected-token>93 <span>token ID</span></div><p class="inspector-distance">Squared distance <strong data-distance>97.2990</strong></p><p class="panel-foot" data-candidate-foot></p><button type="button" class="return-match">Return to nearest match <span aria-hidden="true">↗</span></button></div><div><div class="vector-pair"><div><span>Input vector</span><div class="vector-preview" data-preview="input" role="img"></div></div><div><span data-candidate-label>Codebook entry 93</span><div class="vector-preview" data-preview="candidate" role="img"></div></div></div><p class="vector-legend">All <span data-vector-dimensions>256</span> components · shared grayscale</p><details class="component-details"><summary>View all component values</summary><div class="component-table" tabindex="0" role="region" aria-label="Input and candidate component values"></div></details></div></div></details>
 <details class="example-details"><summary>About this recorded example <span>+</span></summary><p data-provenance></p><p data-source-note></p><p>Each replay follows the same update. The numerical vectors, distances, and token IDs come from the recorded examples; the animation illustrates the processing order. Matching uses all feature dimensions and all 1,024 entries. The eight displayed candidates are ordered by distance. Grayscale uses a shared symmetric scale within each example.</p></details>
 </details>
 <p class="sr-only" aria-live="polite" data-announcement></p>
</div><script type="application/json" id="tokenizer-examples">${JSON.stringify(examples)}</script><script type="application/json" id="tokenizer-geometry">${JSON.stringify(geometry)}</script>`;
const theme=read('static/js/theme.js');
const versionAssets=html=>html.replace(/(href|src)="(static\/(?:css|js)\/[^"?]+)"/g,(_,attr,path)=>`${attr}="${path}?v=${createHash('sha256').update(read(path)).digest('hex').slice(0,10)}"`);
writeFileSync(join(root,'index.html'),versionAssets(read('index.template.html').replace('{{ANATOMY}}',()=>fragment).replace('{{UMAP}}',()=>temporal).replace('{{THEME}}',()=>theme)));
const embeddedFonts=read('static/css/fonts.css').replace(/url\('\.\.\/fonts\/([^']+)'\)/g,(_,file)=>`url('data:font/woff2;base64,${readFileSync(join(root,'static/fonts',file)).toString('base64')}')`);
const styles=[embeddedFonts,...['static/css/site.css','static/css/flow.css'].map(read)].join('\n');
writeFileSync(join(root,'tokenizer.html'),`<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="theme-color" content="#f9f9f6"><title>From EMG signal to token</title><script>${theme}</script><style>${styles}</style></head><body class="standalone"><main>${fragment}</main><script>${read('static/js/tokenizer.js')}</script></body></html>`);
writeFileSync(join(root,'umap.html'),`<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Movement through token space</title><script>${theme}</script><style>${embeddedFonts}\n${read('static/css/site.css')}\n${read('static/css/umap.css')}</style></head><body class="standalone"><main><header class="umap-standalone-header"><h1>Movement through token space</h1><button class="theme-button" data-theme-toggle type="button" aria-label="Switch to dark mode" aria-pressed="false">◐</button></header>${temporal}</main><script>${read('static/js/umap.js')}</script></body></html>`);

// A complete, light-background vector counterpart of the interactive pipeline.
const limit=Math.max(...initial.vector.map(Math.abs),...initial.candidates.flatMap(c=>c.vector.map(Math.abs)));
const gray=v=>{const n=Math.max(0,Math.min(1,.5+.5*v/limit));return `rgb(${Math.round(239-n*199)},${Math.round(239-n*199)},${Math.round(235-n*190)})`;};
const cells=(values,x,y,w,h,cols)=>values.map((v,i)=>`<rect x="${x+i%cols*w/cols}" y="${y+Math.floor(i/cols)*h/(values.length/cols)}" width="${w/cols-.5}" height="${h/(values.length/cols)-.5}" fill="${gray(v)}"/>`).join('');
const text=(x,y,value,css='')=>`<text x="${x}" y="${y}" ${css}>${value}</text>`;
const rect=(x,y,w,h,css='')=>`<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="8" ${css}/>`;
let art=rect(0,0,1120,400,'fill="#fafaf8"');
art+=text(26,30,'FROM EMG SIGNAL TO TOKEN','font-size="12" letter-spacing="2"');
for(let i=0;i<4;i++){
 const x=26+i*273;
 art+=`<path d="M${x} 55h249" stroke="#d1d1cc"/>`;
 art+=text(x+16,83,`${i+1}. ${['Signal','Features','Codebook','Token'][i]}`,'font-size="16" font-weight="600"');
 if(i<3)art+=`<path d="M${x+251} 207h19m-5-4l5 4-5 4" fill="none" stroke="#6e7d74"/>`;
}
art+=text(42,106,'Recorded · one electrode','class="note"');
art+=`<rect x="189" y="125" width="65" height="125" rx="4" fill="#eaf0fa"/><path d="${wavePath(initial.waveform,42,187,213,108)}" fill="none" stroke="#365ca8" stroke-width="1"/>`;
art+=text(42,278,'−480 ms','class="note"')+text(220,278,'now','class="note"')+text(42,330,'160 ms final patch','class="note"');
art+=text(315,106,'Shared causal encoder','class="note"')+cells(initial.vector,345,127,160,160,16)+text(345,330,'256 continuous values','class="note"');
art+=text(588,106,'1,024 entries · eight nearest shown','class="note"');
const max=Math.max(...initial.candidates.map(c=>c.distance));
initial.ranking.forEach((id,i)=>{
 const c=initial.candidates.find(c=>c.id===id), y=133+i*24;
 art+=text(588,y+11,String(id),'font-size="12"')+`<rect x="624" y="${y}" width="${c.distance/max*106}" height="12" rx="2" fill="${i===0?'#5a8069':'#d7ded8'}"/>`+text(738,y+11,c.distance.toFixed(2),'font-size="12"');
});
art+=text(588,344,'Nearest entry: 93','font-size="12" fill="#42624c"');
art+=text(861,106,'Only the index is transmitted','class="note"')+rect(889,126,90,65,'fill="#ecf3ed" stroke="#b8cdbd"')+text(912,170,'93','font-size="36" fill="#42624c" font-weight="600"')+text(990,166,'10 bits','font-size="14"');
initial.bits.split('').forEach((v,i)=>{art+=rect(866+i*21,207,18,24,`fill="${v==='1'?'#42624c':'#f4f5f1'}"`)+text(872+i*21,223,v,`font-size="12" fill="${v==='1'?'white':'#626b75'}"`);});
art+=text(861,261,'↓ Look up codebook entry 93','class="note"')+cells(initial.candidates.find(c=>c.id===93).vector,866,274,204,51,32)+text(861,347,'Vector → hand-pose decoder','class="note"');
writeFileSync(join(root,'static/images/tokenizer.svg'),`<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="400" viewBox="0 0 1120 400" fill="#222b27" role="img" aria-label="Recorded EMG is encoded into 256 features, matched to codebook entry 93, and transmitted as the ten-bit index 0001011101. The receiver looks up its codebook vector."><style>${embeddedFonts}text{font-family:'Switzer Variable',Arial,sans-serif}.note{font-size:13px;fill:#616161}</style>${art}</svg>`);
writeFileSync(join(root,'static/data/geometry.json'),JSON.stringify(geometry,null,2)+'\n');
console.log('Built 2D signal-to-token pipeline, standalone HTML, and static SVG.');
