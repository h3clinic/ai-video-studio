import { useEffect, useState } from 'react'

type Row = Record<string, any>
type Artwork = { id: string; agent: string; gif: string; thumb: string; desc: string }
const ROLE_LABELS:Record<string,string>={'idea-model':'Scene planning','video-controller':'Task coordination','crawler':'Reference research','decision':'Edit decisions','vision-sensor':'Image inspection','section-leads':'Section coordination','field-agents':'Appearance parameters','surroundings-list':'Scene inventory','kind-checkers':'Object checks','mini-object-agents':'Placement and contact','task-assigner':'Part assignments','vector-agents':'Motion and transforms','sound-agents':'Sound effects','verifiers':'Result review'}
export function spotlightActivity(jobs: Row[], stale: boolean) {
  const running=jobs.filter(j=>j.status==='running')
  const queued=jobs.filter(j=>j.status==='queued')
  return {running,queued,job:running[0]||queued[0],status:stale?'Status unavailable':running.length?'Working now':queued.length?'Queued':'Ready for a task'}
}
export default function AgentSpotlight({ agents, roleIds, jobs, selected, onSelect, onHistory, stale }: {
  agents: Artwork[]; roleIds: Record<string,string>; jobs: Row[]; selected?: Artwork | null;
  onSelect: (agent: Artwork) => void; onHistory: () => void; stale: boolean
}) {
  const [animate, setAnimate] = useState(() => !window.matchMedia('(prefers-reduced-motion: reduce)').matches)
  const [clock, setClock] = useState(Date.now())
  const {running,queued,job,status}=spotlightActivity(jobs,stale)
  const active = agents.find(a=>roleIds[a.id]===job?.agentId) || (job ? agents.find(a=>a.id==='50') : null)
  const featured = active || selected || agents[0]
  useEffect(()=>{ if(!job || stale)return; const id=window.setInterval(()=>setClock(Date.now()),1000);return()=>clearInterval(id) },[job?.id,stale])
  const start = job?.startedAt ? Date.parse(job.startedAt) : NaN
  const elapsed = Number.isFinite(start) ? Math.max(0,Math.floor((clock-start)/1000)) : null
  return <section className="agent-spotlight" aria-label="Agent spotlight" data-active-agent={active ? roleIds[active.id] : ''}>
    <div className="agent-spotlight-art"><img src={animate?featured.gif:featured.thumb} alt={`${featured.agent} animated role artwork`} />
      <button onClick={()=>setAnimate(!animate)} aria-pressed={animate}>{animate?'Pause artwork':'Animate artwork'}</button></div>
    <div className="agent-spotlight-copy">
      <div className="agent-eyebrow"><span className={running.length&&!stale?'agent-live-dot':'agent-idle-dot'}/>{status}{!stale&&elapsed!==null&&running.length>0?` · ${elapsed}s elapsed`:''}</div>
      <h2>{featured.agent}</h2>
      <p role="status">{stale?'Connection lost.':running.length?job?.name||'Task in progress.':queued.length?'Waiting to start.':'Choose an agent or start a team.'}</p>
      <div className="agent-spotlight-detail">{job?.partId?`Part: ${job.partId}`:ROLE_LABELS[roleIds[featured.id]]||'Scene tasks'}</div>
      <div className="agent-spotlight-actions"><button onClick={()=>onSelect(featured)}>Assign task</button><button onClick={onHistory}>Agent timeline →</button><span>{running.length} running · {queued.length} queued</span></div>
    </div>
  </section>
}
