// Render the same browser drawing deterministically to 1080p H.264 for slides.
// Requires local Chrome and ffmpeg; no browser library or network service.
import {spawn} from 'node:child_process';
import {mkdtempSync,mkdirSync,rmSync,writeFileSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {join,dirname} from 'node:path';
import {fileURLToPath,pathToFileURL} from 'node:url';
import {once} from 'node:events';

const root=join(dirname(fileURLToPath(import.meta.url)),'..');
const out=join(root,'static/videos');
const qa=join(root,'../tmp/project-page');
mkdirSync(out,{recursive:true});mkdirSync(qa,{recursive:true});
const profile=mkdtempSync(join(tmpdir(),'emg-video-'));
const chrome=spawn(process.env.CHROME_PATH||'/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',[
 '--headless=new','--remote-debugging-port=0',`--user-data-dir=${profile}`,
 '--no-first-run','--no-default-browser-check','--allow-file-access-from-files','about:blank'
]);
let socket,encoder;
try {
 const port=await new Promise((resolve,reject)=>{
  let output='';const timeout=setTimeout(()=>reject(new Error('Chrome startup timeout')),20000);
  chrome.stderr.on('data',chunk=>{output+=chunk;const m=output.match(/DevTools listening on ws:\/\/[^:]+:(\d+)\//);if(m){clearTimeout(timeout);resolve(Number(m[1]));}});
  chrome.on('error',reject);chrome.on('exit',code=>{clearTimeout(timeout);reject(new Error(`Chrome exited: ${code}`));});
 });
 const targets=await (await fetch(`http://127.0.0.1:${port}/json/list`)).json();
 socket=new WebSocket(targets.find(t=>t.type==='page').webSocketDebuggerUrl);
 await new Promise((resolve,reject)=>{socket.addEventListener('open',resolve,{once:true});socket.addEventListener('error',reject,{once:true});});
 let id=0;const pending=new Map();
 socket.addEventListener('message',event=>{
  const message=JSON.parse(event.data);if(!message.id)return;
  const task=pending.get(message.id);if(!task)return;pending.delete(message.id);
  if(message.error)task.reject(new Error(message.error.message));else task.resolve(message.result);
 });
 const call=(method,params={})=>new Promise((resolve,reject)=>{const request=++id;pending.set(request,{resolve,reject});socket.send(JSON.stringify({id:request,method,params}));});
 const evaluate=async expression=>{
  const result=await call('Runtime.evaluate',{expression,awaitPromise:true,returnByValue:true});
  if(result.exceptionDetails)throw new Error(result.exceptionDetails.exception?.description||result.exceptionDetails.text);
  return result.result.value;
 };
 await call('Page.enable');
 await call('Emulation.setEmulatedMedia',{features:[{name:'prefers-reduced-motion',value:'reduce'}]});
 await call('Page.navigate',{url:pathToFileURL(join(root,'umap.html')).href});
 for(let i=0;i<100;i++){
  if(await evaluate("document.querySelector('.umap-widget')?.dataset.ready === 'true'"))break;
  if(i===99)throw new Error('UMAP initialization timed out');
  await new Promise(resolve=>setTimeout(resolve,100));
 }
 await evaluate('document.fonts.ready.then(()=>emgUmap.seek(0))');
 const fps=24;
 const jobs=[{name:'umap-phase-light',dark:false,windows:[480],seconds:8},
             {name:'umap-phase-dark',dark:true,windows:[480],seconds:8},
             {name:'umap-history',dark:false,windows:[0,120,240,480,720,960],seconds:4}];
 for(const job of jobs){
  const filename=join(out,`${job.name}.mp4`);
  encoder=spawn(process.env.FFMPEG_PATH||'ffmpeg',['-y','-loglevel','error','-f','image2pipe','-vcodec','png','-framerate',String(fps),'-i','pipe:0','-an','-c:v','libx264','-preset','fast','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',filename]);
  let encoderError='';encoder.stderr.on('data',chunk=>encoderError+=chunk);
  const completion=new Promise((resolve,reject)=>{encoder.on('error',reject);encoder.on('close',code=>code===0?resolve():reject(new Error(`ffmpeg ${code}: ${encoderError}`)));});
  for(const ms of job.windows){
   for(let frame=0;frame<fps*job.seconds;frame++){
    const phase=frame/(fps*job.seconds-1);
    const url=await evaluate(`emgUmap.presentation(${phase},${ms},${job.dark})`);
    const buffer=Buffer.from(url.split(',')[1],'base64');
    if(!encoder.stdin.write(buffer))await once(encoder.stdin,'drain');
    if(frame===Math.round(fps*job.seconds*.35))writeFileSync(join(qa,`${job.name}-${ms}-slide.png`),buffer);
   }
   console.log(`${job.name}: ${ms} ms complete`);
  }
  encoder.stdin.end();await completion;encoder=undefined;
  console.log(`Wrote ${filename}`);
 }
} finally {
 if(encoder)encoder.kill();
 if(socket)socket.close();chrome.kill();
 await once(chrome,'exit').catch(()=>{});
 rmSync(profile,{recursive:true,force:true});
}
