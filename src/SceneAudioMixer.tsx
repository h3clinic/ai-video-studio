import {useEffect,useRef,useState} from 'react'
type Row=Record<string,any>
export default function SceneAudioMixer({job,onExport,busy}:{job:Row;onExport:(gains:number[])=>void;busy:boolean}) {
  const tracks:Row[]=(job.workers||[]).filter((w:Row)=>w.roleId==='sound-agents'&&w.status==='completed'&&w.execution==='elevenlabs_sound'&&w.mediaId)
  const [levels,setLevels]=useState<Record<string,number>>({})
  const gains=tracks.map(w=>levels[w.id]??(/ambient|environment/i.test(w.partId)?0.25:0.55))
  const allSoundWorkers:Row[]=(job.workers||[]).filter((w:Row)=>w.roleId==='sound-agents')
  const ready=tracks.length>=3&&tracks.length===allSoundWorkers.length
  const [message,setMessage]=useState('')
  const video=useRef<HTMLVideoElement>(null)
  const audio=useRef<(HTMLAudioElement|null)[]>([])
  const pause=()=>audio.current.forEach(a=>a?.pause())
  const sync=()=>audio.current.forEach(a=>{if(a&&video.current&&a.readyState>0){a.currentTime=Math.min(video.current.currentTime,Number.isFinite(a.duration)?a.duration:0)}})
  const play=async()=>{
    sync();setMessage('')
    const outcomes=await Promise.allSettled(audio.current.filter((a):a is HTMLAudioElement=>!!a).map(a=>a.play()))
    if(outcomes.some(o=>o.status==='rejected')){video.current?.pause();pause();setMessage('Audio unavailable. Press play to retry.')}
  }
  useEffect(()=>{audio.current.forEach((a,i)=>{if(a)a.volume=gains[i]??0.5})},[gains])
  useEffect(()=>()=>{audio.current.forEach(a=>a?.pause())},[job.id])
  return <section aria-label="Scene sound mixer" style={{padding:14,borderBottom:'1px solid var(--line)'}}>
    <h3 style={{margin:'0 0 10px'}}>Sound mix</h3>
    <video ref={video} controls preload="metadata" playsInline style={{width:'100%',borderRadius:8}} src={`gaussian-media://artifact/latest_video_${job.id}`} onPlay={()=>void play()} onPause={pause} onSeeking={sync} onEnded={pause}/>
    {tracks.map((w,i)=><div key={w.id} style={{padding:'8px 0'}}>
      <audio ref={el=>{audio.current[i]=el}} preload="auto" src={`gaussian-media://artifact/audio_${w.mediaId}`}/>
      <label style={{display:'flex',gap:10,alignItems:'center'}}><span style={{flex:1,fontSize:12}}>{w.name}</span><input aria-label={`${w.name} volume`} type="range" min="0" max="1" step="0.05" value={gains[i]} onChange={e=>setLevels(old=>({...old,[w.id]:Number(e.target.value)}))}/><output style={{fontSize:11,minWidth:28}}>{Math.round(gains[i]*100)}%</output></label>
    </div>)}
    <button disabled={busy||!ready} onClick={()=>onExport(gains)} style={{padding:'8px 12px',border:'1px solid var(--line)',borderRadius:6,background:'var(--panel)',color:'var(--text)'}}>Export with sound</button>
    <p style={{fontSize:11,color:'var(--muted)'}}>Original video + generated audio · mix/export uses no API credits</p>
    {message&&<p role="alert">{message}</p>}
  </section>
}
