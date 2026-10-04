import {useState} from 'react'
type Row=Record<string,any>
export default function LiveSwarm({job,onAssign,artwork}:{job:Row;onAssign:(role:string,part:string)=>void;artwork:(role:string)=>Row|undefined}) {
  const workers:Row[]=Array.isArray(job.workers)?job.workers:[]
  const [selected,setSelected]=useState<string|null>(null)
  const [animate,setAnimate]=useState(()=>!window.matchMedia('(prefers-reduced-motion: reduce)').matches)
  const chosen=workers.find(w=>w.id===selected)
  const regions=workers.filter(w=>Array.isArray(w.region)&&w.region.length===4)
  return <section className="live-swarm" aria-label="Live agent team">
    <div className="swarm-heading"><div><small>LIVE TEAM</small><h3>Scene crew</h3></div><button onClick={()=>setAnimate(!animate)}>{animate?'Pause animation':'Animate'}</button></div>
    <details><summary>Team brief</summary><p>{job.teamSummary||job.stage}</p></details>
    <div className="swarm-counts"><span>{workers.filter(w=>w.status==='completed').length}/{workers.length} tasks done</span>{workers.some(w=>w.status==='running')&&<span>{workers.filter(w=>w.status==='running').length} working</span>}<span>{workers.filter(w=>w.execution==='elevenlabs_sound').length} audio layers</span></div>
    {workers.some(w=>w.error==='auth_failed'&&w.roleId==='sound-agents')&&<p role="alert" style={{color:'#e6a667'}}>ElevenLabs key rejected (401). Update it in Environment.</p>}
    <div className="swarm-grid">{workers.map(w=>{const art=artwork(w.roleId);return <button className={`swarm-worker ${w.status}`} key={w.id} aria-pressed={selected===w.id} onClick={()=>setSelected(w.id)}>
      {art&&<img src={animate?art.gif:art.thumb} alt="Looping role artwork"/>}<strong>{w.name}</strong><span>{w.partId} · {w.status}</span><small>{w.execution==='elevenlabs_sound'?'Audio ready':w.execution==='specialist_brief'?'Brief ready':w.status==='failed'?'Request failed':w.status==='blocked'?'Needs attention':'Awaiting execution'}</small>
    </button>})}</div>
    {chosen&&<div className="swarm-detail"><strong>{chosen.name} → {chosen.partId}</strong><p>{chosen.task}</p>{chosen.execution==='specialist_brief'&&<small>Agent brief · not verified scene geometry</small>}<p>{chosen.summary}</p>{chosen.requirements?.length>0&&<p>Still needed: {chosen.requirements.join(' · ')}</p>}
      <small>{chosen.startedAt?`Started ${new Date(chosen.startedAt).toLocaleTimeString()}`:'Not started'}{typeof chosen.seconds==='number'?` · ${chosen.seconds.toFixed(2)} s`:''}</small>
      {chosen.execution==='elevenlabs_sound'&&<audio controls preload="none" src={`gaussian-media://artifact/audio_${chosen.mediaId}`}/>}
      <button onClick={()=>onAssign(chosen.roleId,chosen.partId)}>Assign follow-up</button>
    </div>}
    {regions.length>0&&<div><div className="swarm-region-frame"><img src={`gaussian-media://artifact/reference_${job.id}`} alt="Source reference with proposed image regions"/>{regions.map(w=><button key={w.id} aria-label={`Select proposed region: ${w.partId}`} style={{left:`${w.region[0]*100}%`,top:`${w.region[1]*100}%`,width:`${(w.region[2]-w.region[0])*100}%`,height:`${(w.region[3]-w.region[1])*100}%`}} onClick={()=>setSelected(w.id)}><span>{w.partId}</span></button>)}</div><small>Proposed 2D selection · not bound to splats</small></div>}
    <p className="swarm-disclosure">Outputs: briefs + separate audio. Video unchanged.</p>
  </section>
}
