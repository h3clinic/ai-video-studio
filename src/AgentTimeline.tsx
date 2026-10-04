import { useEffect, useMemo, useState } from 'react'
import { Identity } from 'spacetimedb'
import { Clock3, GitBranch, History, Users, RefreshCw, Play, Save, ShieldCheck, AlertTriangle } from 'lucide-react'
import { useCollaboration } from './useSpacetime'

type Row=Record<string,any>
export type SharedAgentTask={agentId:string;partId:string;prompt:string}
type Props={projectId:string;jobs:Row[];recentRunIds?:string[];agents:Row[];parts:Row[];dark:boolean;onRun?:(task:SharedAgentTask)=>void|Promise<void>}
const ROLES=['idea-model','video-controller','crawler','decision','vision-sensor','section-leads','field-agents','surroundings-list','kind-checkers','mini-object-agents','task-assigner','vector-agents','sound-agents','verifiers']
const TERMINAL=new Set(['completed','failed','blocked','interrupted','cancelled','rejected'])
const short=(value:string)=>value ? `${value.slice(0,8)}…${value.slice(-6)}` : 'Unknown'
const instant=(value:unknown)=>{
  if(!value) return 'Not recorded'
  const date=new Date(typeof value==='bigint'?Number(value):value as string)
  return Number.isFinite(date.getTime()) ? date.toLocaleString() : 'Not recorded'
}
const duration=(ms:number|null)=>ms===null?'Not measured':ms<1000?`${Math.round(ms)} ms`:`${(ms/1000).toFixed(2)} s`
const utc=(value:unknown)=>{const ms=typeof value==='string'?Date.parse(value):NaN;return Number.isFinite(ms)?new Date(ms).toISOString():''}
function runtime(job:Row):number|null {
  return typeof job.seconds==='number'&&Number.isFinite(job.seconds)&&job.seconds>=0 ? job.seconds*1000 : null
}
function queue(job:Row):number|null {
  if(!job.startedAt||!(job.queuedAt||job.createdAt)) return null
  const diff=Date.parse(job.startedAt)-Date.parse(job.queuedAt||job.createdAt)
  return Number.isFinite(diff)&&diff>=0?diff:null
}
function agentFor(job:Row):string {
  const id=job.agentId||job.agent||job.request?.agentId
  return ROLES.includes(id)?id:job.kind==='plan'?'idea-model':job.kind==='gemini_reference'?'mini-object-agents':'video-controller'
}
const unsaved=new Map<string,{value:string;revision:bigint}>()

export function timingScope(jobs:Row[], recentRunIds:string[], all:boolean):Row[] {
  if(all||!recentRunIds.length) return jobs
  const ids=new Set(recentRunIds)
  return jobs.filter(j=>ids.has(j.id)||ids.has(j.parentRunId))
}

export default function AgentTimeline({projectId,jobs,recentRunIds=[],agents,parts,dark,onRun}:Props) {
  const shared=useCollaboration()
  const project=shared.projects.find(p=>p.id===projectId)
  const members=shared.members.filter(m=>m.projectId===projectId)
  const mine=members.find(m=>m.identity.toHexString()===shared.identity)
  const canEdit=mine&&['owner','editor'].includes(mine.role)
  const [agentId,setAgentId]=useState('vector-agents'),[partId,setPartId]=useState('scene'),[field,setField]=useState('instruction')
  const [value,setValue]=useState(''),[expectedRevision,setExpectedRevision]=useState(0n),[dirty,setDirty]=useState(false)
  const [busy,setBusy]=useState(false),[message,setMessage]=useState(''),[error,setError]=useState('')
  const [name,setName]=useState('Studio owner'),[invite,setInvite]=useState(''),[inviteName,setInviteName]=useState('Collaborator'),[inviteRole,setInviteRole]=useState('editor')
  const [showAllJobs,setShowAllJobs]=useState(false)
  const key=`${projectId}:${agentId}:${partId}:${field}`
  const current=shared.drafts.find(d=>d.key===key)
  const currentRevision=current?.revision??0n
  useEffect(()=>{
    const draft=unsaved.get(key)
    setValue(draft?.value??current?.value??'');setExpectedRevision(draft?.revision??currentRevision);setDirty(!!draft)
    setMessage('');setError('')
  },[key])
  useEffect(()=>{
    if(!dirty) {setValue(current?.value??'');setExpectedRevision(currentRevision)}
  },[current?.value,currentRevision,dirty])
  const allLocalJobs=useMemo(()=>jobs.filter(j=>j.projectId===projectId).slice().sort((a,b)=>Date.parse(b.createdAt||0)-Date.parse(a.createdAt||0)),[jobs,projectId])
  const localJobs=timingScope(allLocalJobs,recentRunIds,showAllJobs)
  const measured=localJobs.filter(j=>runtime(j)!==null)
  const longest=measured.reduce<Row|null>((best,j)=>(!best||runtime(j)!>runtime(best)!)?j:best,null)
  const edits=shared.edits.filter(e=>e.projectId===projectId).sort((a,b)=>Number(b.id-a.id))
  const timings=shared.timings.filter(t=>t.projectId===projectId)
  const partOptions:Row[]=[...new Map([...parts.filter(p=>p.id&&p.id!=='scene'),...shared.drafts.filter(d=>d.projectId===projectId&&d.partId!=='scene').map(d=>({id:d.partId,label:d.partId}))].map(p=>[p.id,p])).values()]
  const events:Row[]=localJobs.flatMap<Row>(job=>(Array.isArray(job.events)?job.events:[]).map((event:Row)=>({...event,jobId:job.id}))).sort((a,b)=>Date.parse(b.at)-Date.parse(a.at)).slice(0,40)
  const panel:React.CSSProperties={border:'1px solid var(--line)',borderRadius:12,padding:14,background:'var(--panel)'}
  const muted:React.CSSProperties={fontSize:11,color:'var(--muted)',lineHeight:1.6}
  const input:React.CSSProperties={width:'100%',boxSizing:'border-box',background:'var(--canvas)',color:'var(--text)',border:'1px solid var(--line)',borderRadius:7,padding:8,fontSize:12}
  const button:React.CSSProperties={display:'inline-flex',gap:6,alignItems:'center',border:'1px solid var(--line)',borderRadius:7,padding:'7px 10px',background:'var(--hover)',color:'var(--text)',fontSize:11,cursor:'pointer'}
  const act=async(fn:()=>Promise<unknown>,success:string)=>{
    setBusy(true);setError('');setMessage('')
    try {await fn();setMessage(success)} catch(err) {
      const text=err instanceof Error?err.message:'Operation failed'
      setError(/revision conflict/i.test(text)?'Someone edited this part. Review the latest version, then merge your changes.':/membership|owner/i.test(text)?'Your project role does not allow this action.':/secret|credential/i.test(text)?'Do not put API keys into shared history. Use owner settings.':'The change was not confirmed. Check the connection and retry; nothing was marked successful.')
    } finally {setBusy(false)}
  }
  const roleName=(id:string)=>agents.find(a=>a.id===id)?.name||id.replace(/-/g,' ')
  const displayIdentity=(identity:{toHexString():string})=>members.find(m=>m.identity.isEqual(identity as Identity))?.displayName||short(identity.toHexString())
  const save=()=>act(async()=>{
    if(!shared.reducers) throw new Error('disconnected')
    await shared.reducers.editPart({projectId,agentId,partId,field,value,expectedRevision,operationId:crypto.randomUUID()})
    unsaved.delete(key);setDirty(false)
  },'Draft saved. Run it to apply the instruction.')
  const shareTimings=()=>act(async()=>{
    if(!shared.reducers) throw new Error('disconnected')
    for(const job of localJobs.filter(j=>TERMINAL.has(j.status)).slice(0,100)) {
      const ms=runtime(job)
      await shared.reducers.recordTiming({projectId,jobId:String(job.id),agentId:agentFor(job),partId:job.partId||'',status:job.status,
        queuedAt:utc(job.queuedAt||job.createdAt),startedAt:utc(job.startedAt),finishedAt:utc(job.finishedAt),runtimeMs:BigInt(Math.round(ms??0)),hasRuntime:ms!==null,sequence:job.events?.length?Number(job.events[job.events.length-1].sequence)+1:1})
    }
  },'Completed job measurements synchronized. DB receipt time and source measurements remain separate.')

  return <div data-testid="agent-timeline" style={{padding:16,display:'grid',gap:14,color:'var(--text)',colorScheme:dark?'dark':'light'}}>
    <section style={panel}>
      <div style={{display:'flex',justifyContent:'space-between',alignItems:'center',gap:12}}><h3 style={{margin:0,fontSize:16,display:'flex',gap:8,alignItems:'center'}}><GitBranch size={18}/> Agent Observatory</h3>
        <span style={{fontSize:11,color:shared.state==='connected'?'#34b879':'var(--muted)'}}>{shared.state==='connected'?'● Live database':shared.state==='connecting'?'Connecting…':'○ Local history only'}</span></div>
      <p style={muted}>Timings, shared drafts and edit history.</p>
      {recentRunIds.length>0&&<select aria-label="Timing history scope" value={showAllJobs?'all':'recent'} onChange={e=>setShowAllJobs(e.target.value==='all')} style={{...input,marginBottom:10}}><option value="recent">Latest run and team</option><option value="all">All project history ({allLocalJobs.length} records)</option></select>}
      <div style={{display:'grid',gridTemplateColumns:'repeat(3,minmax(0,1fr))',gap:8}}>
        {[['Recorded jobs',String(localJobs.length)],['Measured runtimes',String(measured.length)],['Shared editors',String(members.filter(m=>m.role!=='viewer').length)]].map(([label,total])=><div key={label} style={{background:'var(--hover)',borderRadius:8,padding:10}}><div style={{fontSize:20,fontWeight:600}}>{total}</div><div style={muted}>{label}</div></div>)}
      </div>
      {longest&&<p style={muted}><Clock3 size={12} style={{verticalAlign:'middle'}}/> Longest reported task: <strong>{roleName(agentFor(longest))}</strong> · {duration(runtime(longest))} · {longest.status}{longest.reused?' · reused result':''}</p>}
    </section>

    <section style={panel}>
      <h4 style={{margin:'0 0 10px',display:'flex',gap:8,alignItems:'center'}}><Users size={16}/> Shared workspace</h4>
      <details style={muted}><summary>Local collaboration</summary>{shared.endpoint} · Same-device browser profiles only. LAN and internet sharing are not enabled.</details>
      {shared.error&&<p role="status" style={{...muted,color:'#d99238'}}>{shared.error}</p>}
      {shared.state!=='connected'&&<button style={button} disabled={shared.state==='connecting'} onClick={()=>shared.reconnect().catch(()=>{})}><RefreshCw size={13}/> Reconnect database</button>}
      {shared.identity&&<details style={{margin:'10px 0',fontSize:11}}><summary>My authenticated device identity · {short(shared.identity)}</summary><code style={{display:'block',padding:8,overflowWrap:'anywhere',userSelect:'all'}}>{shared.identity}</code><div style={muted}>Share this public identity with the project owner—not your session token or API keys.</div></details>}
      {!project&&shared.state==='connected'&&<div style={{display:'grid',gap:8,marginTop:10}}><input aria-label="Collaboration display name" value={name} onChange={e=>setName(e.target.value)} maxLength={80} style={input}/><button style={button} disabled={busy||!projectId} onClick={()=>act(async()=>{await shared.reducers!.createWorkspace({projectId,title:`Studio project ${projectId.slice(0,12)}`,displayName:name})},'Private shared workspace created for this project. No prior videos, prompts or credentials were uploaded.')}><ShieldCheck size={13}/> Enable sharing for this project</button><div style={muted}>If this project belongs to someone else, ask the owner to add your device identity. Do not create a duplicate.</div></div>}
      {project&&<>
        <div style={{display:'flex',gap:6,flexWrap:'wrap',margin:'10px 0'}}>{members.map(m=><span key={m.key} title={m.identity.toHexString()} style={{fontSize:11,padding:'5px 8px',borderRadius:20,background:'var(--hover)'}}>{m.displayName} · {m.role}</span>)}</div>
        {mine?.role==='owner'&&<details><summary style={{fontSize:12,cursor:'pointer'}}>Invite or change a collaborator</summary><div style={{display:'grid',gap:8,paddingTop:10}}><input aria-label="Collaborator identity" placeholder="64-character public device identity" value={invite} onChange={e=>setInvite(e.target.value)} maxLength={64} style={input}/><input aria-label="Collaborator name" value={inviteName} onChange={e=>setInviteName(e.target.value)} maxLength={80} style={input}/><select aria-label="Collaborator role" value={inviteRole} onChange={e=>setInviteRole(e.target.value)} style={input}><option value="editor">Editor</option><option value="viewer">Viewer</option><option value="removed">Remove access</option></select><button disabled={busy||!/^[a-f0-9]{64}$/i.test(invite)} style={button} onClick={()=>act(async()=>{await shared.reducers!.setMember({projectId,identity:Identity.fromString(invite),role:inviteRole,displayName:inviteName})},'Membership updated by the server.')}>Update membership</button></div></details>}
      </>}
    </section>

    <section style={panel}>
      <h4 style={{margin:'0 0 10px'}}>Per-agent part editor</h4>
      <div style={{display:'grid',gridTemplateColumns:'1fr 1fr',gap:8}}><select aria-label="Edit responsible agent" value={agentId} onChange={e=>setAgentId(e.target.value)} style={input}>{ROLES.map(id=><option key={id} value={id}>{roleName(id)}</option>)}</select><select aria-label="Edit Gaussian part" value={partId} onChange={e=>setPartId(e.target.value)} style={input}><option value="scene">Scene-wide</option>{partOptions.map(p=><option key={p.id} value={p.id}>{p.label||p.name||p.id}</option>)}</select></div>
      <select aria-label="Part edit field" value={field} onChange={e=>setField(e.target.value)} style={{...input,marginTop:8}}>{['instruction','appearance','motion','review'].map(f=><option key={f}>{f}</option>)}</select>
      <textarea aria-label="Shared part instruction" maxLength={2000} rows={4} value={value} placeholder="Describe this agent's change to this part. No credentials." onChange={e=>{setValue(e.target.value);setDirty(true);unsaved.set(key,{value:e.target.value,revision:expectedRevision})}} style={{...input,marginTop:8,resize:'vertical'}}/>
      <div style={muted}>Server revision {currentRevision.toString()} · {current?`last edited ${instant(current.updatedAt)} by ${displayIdentity(current.updatedBy)}`:'No shared edit yet'}{dirty?' · Unsaved local draft':''}</div>
      {dirty&&expectedRevision!==currentRevision&&<p role="alert" style={{fontSize:12,color:'#d99238'}}><AlertTriangle size={13}/> A collaborator committed revision {currentRevision.toString()}. Your draft will not overwrite it.</p>}
      <div style={{display:'flex',flexWrap:'wrap',gap:8,marginTop:10}}><button style={button} disabled={busy||!canEdit||!value.trim()||!dirty} onClick={save}><Save size={13}/> Commit draft</button><button style={button} onClick={()=>{unsaved.delete(key);setDirty(false);setValue(current?.value??'');setExpectedRevision(currentRevision)}}>Load latest</button>{onRun&&<button style={button} disabled={busy||!canEdit||dirty||!current?.value} onClick={()=>act(async()=>{await onRun({agentId,partId,prompt:current!.value})},'Committed instruction sent to the execution queue. Check the actual job outcome below.')}><Play size={13}/> Run committed instruction</button>}</div>
      {!canEdit&&<p style={muted}>Enable sharing and obtain an editor role to commit changes. Reading job history remains available offline.</p>}
      {message&&<p role="status" style={{...muted,color:'#34b879'}}>{message}</p>}{error&&<p role="alert" style={{...muted,color:'#d99238'}}>{error}</p>}
    </section>

    <section style={panel}>
      <div style={{display:'flex',justifyContent:'space-between',alignItems:'center',gap:8}}><h4 style={{margin:0}}>Agent timing & bottlenecks</h4><button style={button} disabled={busy||!canEdit||!localJobs.some(j=>TERMINAL.has(j.status))} onClick={shareTimings}>Sync measured jobs</button></div>
      <details style={muted}><summary>About timings</summary>Task-reported queue and runtime; database receipt times are separate. Missing measurements stay blank.</details>
      {localJobs.length===0&&<div style={muted}>No jobs recorded for this project yet.</div>}
      {localJobs.slice(0,30).map(job=><details key={job.id} style={{borderTop:'1px solid var(--line)',padding:'9px 0'}}><summary style={{fontSize:12,cursor:'pointer'}}>{roleName(agentFor(job))} · {job.status} · {duration(runtime(job))}</summary><div style={{...muted,paddingTop:8}}>Part: {job.partId||'scene'}<br/>Queued: {instant(job.queuedAt||job.createdAt)}<br/>Started: {instant(job.startedAt)}<br/>Finished: {instant(job.finishedAt)}<br/>Queue wait: {duration(queue(job))}<br/>Runtime: {duration(runtime(job))}<br/>Job: {job.id}</div></details>)}
      {timings.length>0&&<details style={{marginTop:10}}><summary style={{fontSize:12}}>Shared measurements ({timings.length})</summary>{timings.map(t=><div key={t.key} style={{...muted,padding:'7px 0'}}>{roleName(t.agentId)} · {t.sourceStatus} · {t.sourceHasRuntime?duration(Number(t.sourceRuntimeMs)):'Not measured'}<br/>Source finish {instant(t.sourceFinishedAt)} · DB received {instant(t.observedAt)}</div>)}</details>}
      {events.length>0&&<details style={{marginTop:10}}><summary style={{fontSize:12}}>Actual job event stream ({events.length})</summary>{events.map((event,i)=><div key={`${event.jobId}-${event.sequence}-${i}`} style={{...muted,padding:'6px 0'}}>{instant(event.at)} · {event.agentId||'coordinator'} · {event.status}<br/>{String(event.stage||'').slice(0,200)}</div>)}</details>}
    </section>

    <section style={panel}><h4 style={{display:'flex',gap:8,alignItems:'center',margin:'0 0 10px'}}><History size={16}/> Shared edit history</h4>
      {!edits.length&&<div style={muted}>No edits yet.</div>}
      {edits.slice(0,50).map(edit=><details key={edit.id.toString()} style={{borderTop:'1px solid var(--line)',padding:'9px 0'}}><summary style={{fontSize:12,cursor:'pointer'}}>{roleName(edit.agentId)} / {edit.partId} · {edit.field} · r{edit.revision.toString()}</summary><div style={muted}>{instant(edit.editedAt)} · {displayIdentity(edit.editedBy)}</div><div style={{fontSize:12,whiteSpace:'pre-wrap',overflowWrap:'anywhere',padding:8}}><span style={{color:'var(--muted)'}}>Before: {edit.oldValue||'(empty)'}</span><br/>After: {edit.newValue}</div></details>)}
    </section>
  </div>
}
