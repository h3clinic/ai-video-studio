// Developer-only capture helper. Not part of the app UI, bridge or packaged product.
const fs=require('node:fs')
const path=require('node:path')
const {spawn}=require('node:child_process')
const {randomUUID}=require('node:crypto')
let current=null
let last={status:'idle'}
function status(){return current?{status:current.stopping?'encoding':'recording',seconds:Math.floor((Date.now()-current.started)/1000),frames:current.frames}:last}
async function start(window,workspace,durationSeconds=60){
  if(!Number.isInteger(durationSeconds)||durationSeconds<1||durationSeconds>600)throw new Error('Bounded recording duration required')
  if(current)throw new Error('Recording already active')
  const bin=path.join(workspace,'work/.venv/Lib/site-packages/imageio_ffmpeg/binaries')
  const executable=fs.readdirSync(bin).find(name=>/^ffmpeg-win-x86_64-v[\d.]+\.exe$/.test(name))
  if(!executable)throw new Error('Local recording encoder unavailable')
  const root=path.join(workspace,'outputs/gaussian_vector_prototype/artifacts/studio/recordings',randomUUID().replaceAll('-',''))
  fs.mkdirSync(root,{recursive:true})
  const output=path.join(root,'studio-demo.mp4')
  const child=spawn(path.join(bin,executable),['-hide_banner','-loglevel','error','-f','image2pipe','-framerate','10','-vcodec','png','-use_wallclock_as_timestamps','1','-i','pipe:0','-an','-r','10','-fps_mode','cfr','-vf','pad=ceil(iw/2)*2:ceil(ih/2)*2','-c:v','libx264','-threads','2','-preset','veryfast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',output],{windowsHide:true,stdio:['pipe','ignore','ignore']})
  const recording={child,root,output,started:Date.now(),frames:0,timestamps:[],capturing:false,stopping:false,timer:null,limit:null}
  current=recording
  child.stdin.on('error',()=>{void stop()})
  child.on('error',()=>{last={status:'failed',message:'Recording encoder unavailable'};clearInterval(recording.timer);clearTimeout(recording.limit);if(current===recording)current=null})
  child.on('close',code=>{
    clearInterval(recording.timer);clearTimeout(recording.limit)
    const report={status:code===0&&recording.frames>0?'completed':'failed',output,frames:recording.frames,fps:10,wallSeconds:(Date.now()-recording.started)/1000,capturedFrameTimesMs:recording.timestamps,source:'Only AI Video Studio webContents; no desktop, other windows or microphone',audio:false}
    fs.writeFileSync(path.join(root,'recording.json'),JSON.stringify(report,null,2))
    last={status:report.status,output,capturedFrames:report.frames,seconds:report.wallSeconds};if(current===recording)current=null
  })
  recording.timer=setInterval(async()=>{
    if(recording.capturing||recording.stopping||window.isDestroyed())return
    recording.capturing=true
    try {
      const captured=await window.webContents.capturePage()
      if(!recording.stopping&&!captured.isEmpty()){
        if(child.stdin.writableLength>8*1024*1024){void stop();return}
        child.stdin.write(captured.toPNG());recording.frames++;recording.timestamps.push(Date.now()-recording.started)
      }
    } catch {void stop()} finally {recording.capturing=false}
  },100)
  recording.limit=setTimeout(()=>{void stop()},durationSeconds*1000)
  last={status:'recording'};return status()
}
async function stop(){
  if(!current)return last
  const c=current;if(c.stopping)return status()
  c.stopping=true;clearInterval(c.timer);clearTimeout(c.limit);c.child.stdin.end()
  // Bounded local encoding cleanup; no unrelated process is touched.
  const deadline=setTimeout(()=>c.child.kill(),10000);c.child.once('close',()=>clearTimeout(deadline))
  return status()
}
module.exports={start,stop,status}
