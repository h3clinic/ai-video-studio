import { useState, useRef, useEffect } from 'react'
import { Plus, Send, Loader2, Sun, Moon, Trash2, Pencil, Check, X, PanelRightClose, PanelRight, Play } from 'lucide-react'
import { useTheme } from '@/contexts/ThemeContext'
import { useSpacetime } from './useSpacetime'
import { pickColor } from './spacetime'
import AgentSpotlight from './AgentSpotlight'
import AgentTimeline from './AgentTimeline'
import ElevenLabsSettings from './ElevenLabsSettings'
import LiveSwarm from './LiveSwarm'
import SceneAudioMixer from './SceneAudioMixer'

interface AgentNote {
  agentId: string
  note: string
}

interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  agents?: AgentNote[]
}

interface Chat {
  id: string
  title: string
  messages: Message[]
  createdAt: number
}

interface Artifact {
  id: string
  agent: string
  model: string
  desc: string
  sizeKB: number
  video: string
  thumb: string
  gif: string
  output: string
}

const AGENTS: Artifact[] = [
  { id: '49', agent: 'Idea Model',             model: 'Qwen3-4B',     desc: 'Writes the script; parent of Crawler and Decision agents', sizeKB: 50,  video: '/videos/agent-49.mp4', thumb: '/thumbs/agent-49.jpg', gif: '/gifs/agent-49.gif', output: 'Script outline with scene breakdown, character descriptions, and narration cues' },
  { id: '50', agent: 'Video Controller',        model: 'code',         desc: 'Owns the 60s budget and times every stage',               sizeKB: 217, video: '/videos/agent-50.mp4', thumb: '/thumbs/agent-50.jpg', gif: '/gifs/agent-50.gif', output: 'Timeline manifest: stage durations, sync points, and render order' },
  { id: '51', agent: 'Crawler Agent',           model: 'Qwen3-4B',    desc: 'Writes the image search queries',                         sizeKB: 27,  video: '/videos/agent-51.mp4', thumb: '/thumbs/agent-51.jpg', gif: '/gifs/agent-51.gif', output: 'Search query list for reference images matching each scene' },
  { id: '52', agent: 'Decision Agent',          model: 'Qwen3-4B',    desc: 'Picks photos; answers fur, ground and bone-name forms',   sizeKB: 46,  video: '/videos/agent-52.mp4', thumb: '/thumbs/agent-52.jpg', gif: '/gifs/agent-52.gif', output: 'Selected reference photos with fur/ground/bone attribute answers' },
  { id: '53', agent: 'Vision Sensor',           model: 'Qwen3.5-0.8B', desc: 'Answers one yes/no question per photo',                  sizeKB: 248, video: '/videos/agent-53.mp4', thumb: '/thumbs/agent-53.jpg', gif: '/gifs/agent-53.gif', output: 'Binary yes/no judgments for each photo validation check' },
  { id: '54', agent: 'Section Leads',           model: 'Qwen3-4B',    desc: 'One per section: look, material, gait, ground texture',   sizeKB: 84,  video: '/videos/agent-54.mp4', thumb: '/thumbs/agent-54.jpg', gif: '/gifs/agent-54.gif', output: 'Section specs: look profile, material list, gait model, ground texture map' },
  { id: '55', agent: 'Field Agents',            model: 'Qwen3-4B',    desc: 'One tiny agent per number or word inside a section',      sizeKB: 368, video: '/videos/agent-55.mp4', thumb: '/thumbs/agent-55.jpg', gif: '/gifs/agent-55.gif', output: 'Filled attribute values for every field in each section' },
  { id: '56', agent: 'Surroundings List Agent', model: 'Qwen3-4B',    desc: 'Lists the kinds of object and their proportions',         sizeKB: 74,  video: '/videos/agent-56.mp4', thumb: '/thumbs/agent-56.jpg', gif: '/gifs/agent-56.gif', output: 'Object inventory with types, counts, and proportion ratios' },
  { id: '57', agent: 'Kind Checkers',           model: 'Qwen3.5-0.8B', desc: 'One per kind: "is this a physical object?"',             sizeKB: 42,  video: '/videos/agent-57.mp4', thumb: '/thumbs/agent-57.jpg', gif: '/gifs/agent-57.gif', output: 'Physical-object verification flags per object kind' },
  { id: '58', agent: 'Mini Object Agents',      model: 'Qwen3.5-0.8B', desc: 'One per surrounding object: its place and size',         sizeKB: 56,  video: '/videos/agent-58.mp4', thumb: '/thumbs/agent-58.jpg', gif: '/gifs/agent-58.gif', output: 'Position coordinates and bounding dimensions for each object' },
  { id: '59', agent: 'Task Assigner',           model: 'Qwen3.5-0.8B', desc: 'Decides which Vector Agents to spawn',                  sizeKB: 128, video: '/videos/agent-59.mp4', thumb: '/thumbs/agent-59.jpg', gif: '/gifs/agent-59.gif', output: 'Vector Agent spawn plan mapping body parts to V1-V7' },
  { id: '60', agent: 'Vector Agents',           model: 'Qwen3.5-0.8B', desc: 'V1-V7: each drives one body part movement',             sizeKB: 245, video: '/videos/agent-60.mp4', thumb: '/thumbs/agent-60.jpg', gif: '/gifs/agent-60.gif', output: 'Motion vectors for each body part across all keyframes' },
  { id: '61', agent: 'Sound Agents',            model: 'Qwen3.5-0.8B', desc: 'Choose which contacts make sound',                      sizeKB: 68,  video: '/videos/agent-61.mp4', thumb: '/thumbs/agent-61.jpg', gif: '/gifs/agent-61.gif', output: 'Sound event list: footstep timing, impact points, surface types' },
  { id: '62', agent: 'Verifiers',               model: 'laws',         desc: 'One per decision: checks against physics or maths law',  sizeKB: 60,  video: '/videos/agent-62.mp4', thumb: '/thumbs/agent-62.jpg', gif: '/gifs/agent-62.gif', output: 'Verification verdicts with pass/fail per physics and maths law' },
]

const AGENT_COUNTS: Record<string, string> = {
  'Video Controller': '1',
  'Idea Model': '1',
  'Crawler Agent': '1',
  'Decision Agent': '1',
  'Vision Sensor': '1',
  'Section Leads': '4',
  'Field Agents': '~35',
  'Surroundings List Agent': '1',
  'Kind Checkers': '5',
  'Mini Object Agents': '72',
  'Task Assigner': '1',
  'Vector Agents': '7',
  'Sound Agents': '1 lead',
  'Verifiers': '44',
}

// Original role GIFs identify specialists; they are never project output media.
const AGENT_ROLE_IDS: Record<string, string> = {
  '49': 'idea-model', '50': 'video-controller', '51': 'crawler',
  '52': 'decision', '53': 'vision-sensor', '54': 'section-leads',
  '55': 'field-agents', '56': 'surroundings-list', '57': 'kind-checkers',
  '58': 'mini-object-agents', '59': 'task-assigner', '60': 'vector-agents',
  '61': 'sound-agents', '62': 'verifiers',
}

const WELCOME: Message = {
  id: 'welcome',
  role: 'assistant',
  content:
    "What would you like to change?",
}

function makeId() {
  return Math.random().toString(36).slice(2, 10)
}

function ActiveAgentBanner({ agentId, note }: { agentId: string; note: string; isDark: boolean }) {
  const agent = AGENTS.find(a => a.id === agentId)
  if (!agent) return null

  return (
    <div style={{
      display: 'flex', gap: 16, padding: '14px 16px', borderRadius: 8,
      border: '1.5px solid var(--line-strong)',
      background: 'var(--panel)',
      alignItems: 'center',
      animation: 'agentSlideIn 0.4s ease-out',
    }}>
      <img
        src={agent.gif}
        alt={agent.agent}
        style={{
          width: 80, height: 45, borderRadius: 6, objectFit: 'cover', flexShrink: 0,
        }}
      />
      <div style={{ flex: 1, minWidth: 0 }}>
        <div style={{ display: 'flex', alignItems: 'baseline', gap: 8 }}>
          <span style={{ fontSize: 14, fontWeight: 700 }}>{agent.agent}</span>
          <span style={{ fontSize: 11, color: 'var(--faint)', fontFamily: 'monospace' }}>
            {agent.model}{AGENT_COUNTS[agent.agent] !== '1' ? ` x${AGENT_COUNTS[agent.agent]}` : ''}
          </span>
        </div>
        <div style={{ fontSize: 12, color: 'var(--muted)', marginTop: 5, lineHeight: 1.5 }}>
          {note}
        </div>
      </div>
    </div>
  )
}

function DoneAgentRow({ agentId }: { agentId: string; isDark: boolean }) {
  const agent = AGENTS.find(a => a.id === agentId)
  if (!agent) return null

  return (
    <div style={{
      display: 'flex', alignItems: 'center', gap: 8, padding: '3px 0',
      opacity: 0.55, fontSize: 11, color: 'var(--muted)',
    }}>
      <img
        src={agent.gif}
        alt=""
        style={{ width: 24, height: 14, borderRadius: 2, objectFit: 'cover' }}
      />
      <span>{agent.agent}</span>
      <span style={{ color: 'var(--faint)', marginLeft: 'auto', fontSize: 10 }}>done</span>
    </div>
  )
}

type PanelTab = 'video' | 'contents' | 'environment' | 'timeline'

async function hashPassword(email: string, password: string): Promise<string> {
  const enc = new TextEncoder()
  const key = await crypto.subtle.importKey('raw', enc.encode(password), 'PBKDF2', false, ['deriveBits'])
  const bits = await crypto.subtle.deriveBits(
    { name: 'PBKDF2', hash: 'SHA-256', salt: enc.encode('ai-video-studio:' + email), iterations: 100000 },
    key, 256,
  )
  return Array.from(new Uint8Array(bits), b => b.toString(16).padStart(2, '0')).join('')
}

const gateInput: React.CSSProperties = {
  width: '100%', padding: '9px 12px', border: '1px solid var(--line)',
  borderRadius: 4, background: 'var(--panel)', color: 'var(--text)',
  fontSize: 13, outline: 'none',
}
const gateLabel: React.CSSProperties = { fontSize: 11, fontWeight: 600, color: 'var(--muted)', display: 'block', marginBottom: 4 }
const gatePrimary: React.CSSProperties = {
  width: '100%', padding: '10px', border: '1px solid #dedede',
  borderRadius: 4, background: '#dedede', color: '#111',
  fontSize: 13, fontWeight: 600, cursor: 'pointer',
}
const gateLink: React.CSSProperties = {
  width: '100%', padding: '8px', border: 'none', borderRadius: 4, background: 'transparent',
  color: 'var(--faint)', fontSize: 12, cursor: 'pointer', marginTop: 8,
}

type GateStep = 'signin' | 'details' | 'forgot' | 'code' | 'password'

function SignInGate({ stdb }: { stdb: ReturnType<typeof useSpacetime> }) {
  const [step, setStep] = useState<GateStep>('signin')
  const [purpose, setPurpose] = useState<'signup' | 'reset'>('signup')
  const [email, setEmail] = useState('')
  const [accessCode, setAccessCode] = useState('')
  const [emailCode, setEmailCode] = useState('')
  const [password, setPassword] = useState('')
  const [confirmPassword, setConfirmPassword] = useState('')
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const [busy, setBusy] = useState(false)

  const cleanEmail = email.trim().toLowerCase()

  const go = (next: GateStep) => {
    setError('')
    setNotice('')
    setPassword('')
    setConfirmPassword('')
    setEmailCode('')
    setStep(next)
  }

  // Server calls either throw or return a message to show; '' means success.
  const run = async (action: () => Promise<string | void>, onSuccess?: () => void) => {
    if (busy) return
    if (!stdb.reducers || !stdb.procedures) {
      setError('Cannot reach the server. Try again in a moment.')
      return
    }
    setError('')
    setNotice('')
    setBusy(true)
    try {
      const message = await action()
      if (message) setError(message)
      else onSuccess?.()
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setBusy(false)
    }
  }

  const sendCode = (p: 'signup' | 'reset', onSuccess: () => void) =>
    run(() => stdb.procedures!.requestCode({ email: cleanEmail, purpose: p, accessCode }), onSuccess)

  const submit = () => {
    if (step === 'signin') {
      if (!cleanEmail || !password) return setError('Enter your email and password')
      run(async () => stdb.reducers!.signIn({ email: cleanEmail, clientHash: await hashPassword(cleanEmail, password) }))
    } else if (step === 'details' || step === 'forgot') {
      const p = step === 'details' ? 'signup' : 'reset'
      sendCode(p, () => { setPurpose(p); go('code') })
    } else if (step === 'code') {
      run(() => stdb.procedures!.verifyCode({ email: cleanEmail, code: emailCode }), () => go('password'))
    } else {
      if (password.length < 8) return setError('Password must be at least 8 characters')
      if (password !== confirmPassword) return setError('Passwords do not match')
      run(async () => {
        const args = { email: cleanEmail, clientHash: await hashPassword(cleanEmail, password) }
        await (purpose === 'signup' ? stdb.reducers!.signUp(args) : stdb.reducers!.resetPassword(args))
      })
    }
  }

  const onEnter = (e: React.KeyboardEvent) => { if (e.key === 'Enter') submit() }

  const subtitle = {
    signin: 'Sign in to start creating',
    details: 'Create your account',
    forgot: 'Reset your password',
    code: 'Check your email',
    password: purpose === 'signup' ? 'Create a password' : 'Choose a new password',
  }[step]

  const submitLabel = {
    signin: 'Sign In',
    details: 'Send Verification Code',
    forgot: 'Send Reset Code',
    code: 'Verify',
    password: purpose === 'signup' ? 'Create Account' : 'Reset Password',
  }[step]

  return (
    <div style={{
      width: '100%', height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center',
      background: 'var(--canvas)', fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
    }}>
      <div style={{
        width: 400, background: 'var(--chrome)', border: '1px solid var(--line)',
        borderRadius: 10, padding: '40px 32px',
      }}>
        <div style={{ textAlign: 'center', marginBottom: 28 }}>
          <div style={{ fontSize: 20, fontWeight: 700, color: 'var(--text)' }}>AI Video Studio</div>
          <div style={{ fontSize: 12, color: 'var(--muted)', marginTop: 6 }}>{subtitle}</div>
        </div>

        {(step === 'signin' || step === 'details' || step === 'forgot') && (
          <div style={{ marginBottom: step === 'forgot' ? 16 : 12 }}>
            <label style={gateLabel}>Email</label>
            <input type="email" value={email} onChange={e => setEmail(e.target.value)} onKeyDown={onEnter} placeholder="you@example.com" style={gateInput} />
          </div>
        )}

        {step === 'signin' && (
          <div style={{ marginBottom: 16 }}>
            <label style={gateLabel}>Password</label>
            <input type="password" value={password} onChange={e => setPassword(e.target.value)} onKeyDown={onEnter} placeholder="Your password" style={gateInput} />
          </div>
        )}

        {step === 'details' && (
          <div style={{ marginBottom: 16 }}>
            <label style={gateLabel}>Access Code</label>
            <input type="text" value={accessCode} onChange={e => setAccessCode(e.target.value)} onKeyDown={onEnter} placeholder="Enter access code" style={{ ...gateInput, fontFamily: 'monospace' }} />
          </div>
        )}

        {step === 'code' && (
          <div style={{ marginBottom: 16 }}>
            <div style={{ fontSize: 12, color: 'var(--faint)', marginBottom: 14, lineHeight: 1.5, textAlign: 'center' }}>
              {purpose === 'signup'
                ? <>We sent a 6-digit code to <span style={{ color: 'var(--muted)' }}>{cleanEmail}</span>.</>
                : <>If <span style={{ color: 'var(--muted)' }}>{cleanEmail}</span> has an account, we sent it a 6-digit code.</>}
            </div>
            <label style={gateLabel}>Verification Code</label>
            <input
              type="text" inputMode="numeric" maxLength={6} autoFocus
              value={emailCode} onChange={e => setEmailCode(e.target.value.replace(/\D/g, ''))} onKeyDown={onEnter}
              placeholder="000000"
              style={{ ...gateInput, fontFamily: 'monospace', letterSpacing: '0.3em', textAlign: 'center', fontSize: 16 }}
            />
          </div>
        )}

        {step === 'password' && (
          <>
            <div style={{ marginBottom: 12 }}>
              <label style={gateLabel}>Password</label>
              <input type="password" value={password} onChange={e => setPassword(e.target.value)} onKeyDown={onEnter} placeholder="At least 8 characters" autoFocus style={gateInput} />
            </div>
            <div style={{ marginBottom: 16 }}>
              <label style={gateLabel}>Confirm Password</label>
              <input type="password" value={confirmPassword} onChange={e => setConfirmPassword(e.target.value)} onKeyDown={onEnter} placeholder="Repeat password" style={gateInput} />
            </div>
          </>
        )}

        {(error || stdb.failed) && (
          <div style={{ fontSize: 12, color: '#f87171', marginBottom: 12 }}>
            {error || 'Cannot reach the server.'}
          </div>
        )}
        {notice && <div style={{ fontSize: 12, color: 'var(--muted)', marginBottom: 12 }}>{notice}</div>}

        <button onClick={submit} disabled={busy} style={{ ...gatePrimary, opacity: busy ? 0.5 : 1 }}>{submitLabel}</button>

        {step === 'signin' && (
          <>
            <button onClick={() => go('forgot')} style={gateLink}>Forgot password?</button>
            <button onClick={() => go('details')} style={{ ...gateLink, marginTop: 0 }}>New here? Create an account</button>
          </>
        )}
        {step === 'code' && (
          <button onClick={() => sendCode(purpose, () => setNotice('A new code is on its way.'))} style={gateLink}>Resend code</button>
        )}
        {step !== 'signin' && (
          <button onClick={() => go('signin')} style={{ ...gateLink, marginTop: step === 'code' ? 0 : 8 }}>Back to sign in</button>
        )}
      </div>
    </div>
  )
}

type LocalRow = Record<string, any>
const localBridge = () => (window as any).gaussianStudio as {request: (path: string, body?: LocalRow) => Promise<any>} | undefined
const localRows = (value: unknown): LocalRow[] => Array.isArray(value) ? value.filter(v => v && typeof v === 'object') : []
const safeMedia = (url: unknown) => typeof url === 'string' && /^gaussian-media:\/\/artifact\/[a-zA-Z0-9_/?=&.-]+$/.test(url) ? url : undefined

// Contents describes the assigned task. Audio and reviews do not operate on
// Gaussian IDs; this display classification never grants visual edit access.
export function partContentState(part: LocalRow, jobs: LocalRow[] = [], media: LocalRow[] = []) {
  const matching = jobs.filter(job=>job.partId===part.id || localRows(job.workers).some(worker=>worker.partId===part.id))
  const job = latestRuns(matching).current
  const worker = localRows(job?.workers).find(row=>row.partId===part.id) || (job?.partId===part.id ? job : undefined)
  const role = worker?.roleId || worker?.agentId || part.ownerRoleId || part.roleId || part.agentId
  const status = worker?.status
  const workflowRoles = ['idea-model','video-controller','crawler','decision','task-assigner','surroundings-list','kind-checkers','vision-sensor']
  const visual = part.protected === true || part.bindingVerified === true || Number.isInteger(part.gaussianCount)
  if (!visual && role==='sound-agents') {
    const generated = status==='completed' && (worker?.execution==='elevenlabs_sound' || worker?.outputKind==='audio') && typeof worker?.output==='string' && !!worker.output
    const audioId = worker?.mediaId ? 'audio_'+worker.mediaId : 'audio_'+worker?.id
    const available = generated && media.some(item=>item.id===audioId && item.kind==='audio')
    return {kind:'audio',requiresGeometryBinding:false,label:available ? 'Audio ready · Separate generated sound layer' : generated ? 'Audio generated · Preview unavailable' : status==='failed' ? 'Sound task failed · No audio generated' : status==='blocked' ? 'Sound task needs attention · No audio generated' : status==='running' ? 'Generating sound' : 'Sound task · No audio generated'}
  }
  if (!visual && (role==='verifiers' || workflowRoles.includes(role))) {
    const review = role==='verifiers'
    const prefix = review ? 'Review' : 'Workflow'
    const completed = status==='completed' && worker?.execution==='specialist_brief'
    return {kind:review?'review':'workflow',requiresGeometryBinding:false,
      label:completed ? (review ? 'Review brief ready · Geometry quality unverified' : 'Brief ready · No visual edit executed') : `${prefix} task · ${status==='failed'?'Failed':status==='blocked'?'Needs attention':status==='running'?'In progress':'No brief produced'}`}
  }
  const bound = part.bindingVerified===true && Number.isInteger(part.gaussianCount) && part.gaussianCount>0
  return {kind:'visual',requiresGeometryBinding:true,label:part.protected===true ? 'Protected scene part' : bound ? `Canonical IDs bound · ${part.gaussianCount} Gaussian IDs` : 'Unbound · Visual edits require verified Gaussian IDs'}
}

export function latestRuns(jobs: LocalRow[]) {
  const submitted = (job: LocalRow) => {
    const value = typeof job.createdAt==='number' ? job.createdAt : Date.parse(job.createdAt)
    return Number.isFinite(value) ? value : 0
  }
  const ordered = jobs.map((job,index)=>({job,index})).sort((a,b)=>submitted(b.job)-submitted(a.job)||b.index-a.index).map(item=>item.job)
  return {current:ordered[0],history:ordered.slice(1)}
}

export function runTimestamp(value: unknown) {
  if (typeof value!=='string' && typeof value!=='number') return 'Time not recorded'
  const date = new Date(value)
  return Number.isFinite(date.getTime()) ? date.toLocaleString() : 'Time not recorded'
}

export function runProblem(job: LocalRow) {
  if (!job.error && !/blocked|fail|error|reject/i.test(String(job.status))) return null
  const detail = typeof job.error==='string' && job.error.trim() ? job.error.trim() : typeof job.stage==='string' && job.stage.trim() ? job.stage.trim() : `Run ${job.status}.`
  const summary = detail.split(/\r?\n/)[0]
  const short = summary.length>150 ? `${summary.slice(0,147)}…` : summary
  if (/401|403|auth|credential|api.?key|permission/i.test(detail)) return {summary:short,action:'Check the provider credentials in Environment, then submit the task again.',environment:true}
  if (job.kind==='apple_experiment') return {summary:short,action:'Open Environment to review the missing scene-replacement prerequisites.',environment:true}
  if (/connect|unavailable|timeout|timed out|network|remote|gpu/i.test(detail)) return {summary:short,action:'Check the remote connection in Environment before retrying.',environment:true}
  return {summary:short,action:job.kind==='agent_swarm' && job.status==='blocked' ? 'Review the unfinished tasks below, resolve their prerequisites, then resume.' : 'Review the run details and resolve the reported prerequisite before retrying.',environment:false}
}

const runLabel = (job: LocalRow, artwork?: Artifact) => job.kind==='agent_swarm' ? 'Scene team' : artwork?.agent || String(job.kind||'Task').replace(/_/g,' ')

function RunDetails({job,artwork}: {job:LocalRow;artwork?:Artifact}) {
  return <details style={{fontSize:11,lineHeight:1.6,marginTop:8}}>
    <summary style={{cursor:'pointer',color:'var(--muted)'}}>Run details</summary>
    {artwork&&<img src={artwork.gif} alt={`${artwork.agent} role illustration`} style={{width:44,height:25,objectFit:'cover',borderRadius:3,marginTop:8}}/>}
    <div style={{color:'var(--faint)',marginTop:6}}>Run ID: {job.id}</div>
    {job.stage&&<div style={{color:'var(--muted)',marginTop:6}}>{job.stage}</div>}
    {job.partId&&<div style={{color:'var(--faint)'}}>Assigned part: {job.partId}</div>}
    {job.prompt&&<div style={{marginTop:6,color:'var(--muted)'}}>Request: {job.prompt}</div>}
    {job.reply&&<div style={{whiteSpace:'pre-wrap',marginTop:8}}>{job.reply}</div>}
    {job.error&&<div style={{color:'#ef9999',whiteSpace:'pre-wrap',overflowWrap:'anywhere',marginTop:8}}>{String(job.error)}</div>}
    <div style={{color:'var(--faint)',marginTop:6}}>Cost: {typeof job.cost_usd==='number'?`$${job.cost_usd.toFixed(4)}`:'Not measured'} · Quality: {typeof job.review==='string'?job.review:'Not accepted'}</div>
    <div style={{color:'var(--faint)',marginTop:4}}>Submitted: {runTimestamp(job.createdAt)}{job.finishedAt?` · Finished: ${runTimestamp(job.finishedAt)}`:''}</div>
  </details>
}

function RunProblem({job,onEnvironment}: {job:LocalRow;onEnvironment:()=>void}) {
  const problem=runProblem(job)
  if(!problem)return null
  return <div style={{fontSize:12,lineHeight:1.5,marginTop:8}}>
    <div style={{color:'#ef9999'}}>{problem.summary}</div>
    <div style={{color:'var(--muted)',fontSize:11,marginTop:4}}>{problem.action}</div>
    {problem.environment&&<button onClick={onEnvironment} style={{...gateLink,width:'auto',padding:'4px 0',marginTop:3,textDecoration:'underline'}}>Open Environment</button>}
  </div>
}

export function RunHistory({jobs,artwork,onEnvironment,onAssign}: {jobs:LocalRow[];artwork:(id:unknown)=>Artifact|undefined;onEnvironment:()=>void;onAssign:(role:string,part:string)=>void}) {
  if(!jobs.length)return null
  return <details data-run-history="true" style={{borderTop:'1px solid var(--line)',paddingTop:14,marginTop:22,fontSize:12}}>
    <summary style={{cursor:'pointer',color:'var(--muted)'}}>History · {jobs.length} earlier {jobs.length===1?'run':'runs'}</summary>
    <div style={{fontSize:10,color:'var(--faint)',margin:'8px 0'}}>Previous submissions · newest first</div>
    {jobs.map(job=><details key={job.id} data-history-run={job.id} style={{border:'1px solid var(--line)',padding:10,borderRadius:4,marginBottom:8,lineHeight:1.6}}>
      <summary style={{cursor:'pointer'}}>{runLabel(job,artwork(job.agentId))} · {job.status}<span style={{display:'block',fontSize:10,color:'var(--faint)',marginTop:3}}>{runTimestamp(job.createdAt)}</span></summary>
      {job.stage&&<div style={{color:'var(--muted)',marginTop:7}}>{job.stage}</div>}
      <RunProblem job={job} onEnvironment={onEnvironment}/>
      {job.kind==='agent_swarm'&&<LiveSwarm job={job} artwork={artwork} onAssign={onAssign}/>}
      <RunDetails job={job} artwork={artwork(job.agentId)}/>
    </details>)}
  </details>
}

export function appleReplacementReadiness(value: unknown) {
  const capability = value && typeof value==='object' && !Array.isArray(value) ? value as LocalRow : {}
  const revision = typeof capability.revision==='string' && /^[A-Za-z0-9._-]{1,120}$/.test(capability.revision) ? capability.revision : null
  const missing = Array.isArray(capability.missing) ? capability.missing.filter((item: unknown): item is string => typeof item==='string' && item.length>0).slice(0,16) : []
  const validMissing = Array.isArray(capability.missing) && capability.missing.length<=16 && capability.missing.every((item: unknown)=>typeof item==='string' && item.length>0)
  const ready = capability.ready===true && revision!==null && revision!=='semantic-static-v1-rejected' && validMissing && missing.length===0
  return {ready, revision, missing, reason:typeof capability.reason==='string' && capability.reason.trim()
    ? capability.reason
    : ready ? 'A versioned replacement pipeline is available.' : 'A repaired material/contact pipeline has not been made available. The rejected static-apple experiment will not be repeated.'}
}

export function remoteFailure(remote: unknown) {
  if (!remote || typeof remote!=='object' || Array.isArray(remote)) return null
  const state = remote as LocalRow
  const explanations: Record<string,string> = {
    auth_failed:'RunPod rejected the saved owner authorization. A successful connection has not been verified.',
    permission_denied:'The saved owner authorization cannot access this RunPod resource.',
    unavailable:'RunPod is unavailable. No successful connection has been verified.',
  }
  return typeof state.error==='string' && state.error.trim() ? state.error : explanations[state.status] || null
}

function DesktopAgentCard({agent, state, parts, onTask}: {
  agent: Artifact; state?: LocalRow; parts: LocalRow[]; onTask: () => void
}) {
  const status = typeof state?.status === 'string' ? state.status : 'No task submitted'
  const assigned = parts.filter(p => p.ownerRoleId===AGENT_ROLE_IDS[agent.id] || p.agentId===AGENT_ROLE_IDS[agent.id] ||
    (Array.isArray(state?.partIds) && state.partIds.includes(p.id)))
  return <div data-agent-role={AGENT_ROLE_IDS[agent.id]} style={{padding:'10px 12px',borderBottom:'1px solid var(--line-soft)'}}>
    <div style={{display:'flex',alignItems:'center',gap:9}}>
      <img src={agent.gif} alt={`${agent.agent} role illustration`} title="Role illustration — not project output" style={{width:112,height:68,objectFit:'contain',borderRadius:6,flexShrink:0}}/>
      <div style={{flex:1,minWidth:0}}>
        <div style={{fontSize:12,fontWeight:600}}>{agent.agent}</div>
        <div style={{fontSize:10,color:'var(--faint)',marginTop:3}}>{state?.model || 'Owner-configured backend'}</div>
      </div>
      <button aria-label={`Task ${agent.agent}`} onClick={onTask} style={{background:'var(--panel)',border:'1px solid var(--line)',borderRadius:3,padding:'5px 7px',color:'var(--text)',cursor:'pointer'}}><Send size={12}/></button>
    </div>
    <div style={{fontSize:11,marginTop:7,color:/blocked|fail|error/i.test(status)?'#ef9999':'var(--muted)'}}>{status}</div>
    <div style={{fontSize:11,color:'var(--faint)',marginTop:4,lineHeight:1.5}}>{typeof state?.task==='string' && state.task ? state.task : agent.desc}</div>
    {assigned.length>0 && <div style={{fontSize:10,color:'var(--muted)',marginTop:6}}>Parts: {assigned.map(p=>p.label||p.id).join(', ')}</div>}
  </div>
}

function BrowserApp() { const stdb = useSpacetime(); return <StudioApp stdb={stdb} /> }
export default function App() {
  const sharedProject=new URLSearchParams(window.location.search).get('collaborate')
  if(!localBridge() && sharedProject && /^[A-Za-z0-9_-]{1,80}$/.test(sharedProject)) return <div style={{height:'100vh',overflowY:'auto',background:'var(--canvas)'}}>
    <header style={{padding:'20px 28px',borderBottom:'1px solid var(--line)'}}><h1 style={{fontSize:20}}>AI Video Studio · Shared editor</h1><p style={{fontSize:12,color:'var(--muted)',marginTop:6}}>Local collaboration for {sharedProject}. Ask the project owner to grant your public device identity access. API keys and remote execution remain in the owner’s desktop app.</p></header>
    <div style={{maxWidth:860,margin:'0 auto'}}><AgentTimeline projectId={sharedProject} jobs={[]} agents={AGENTS.map(a=>({id:AGENT_ROLE_IDS[a.id],name:a.agent}))} parts={[]} dark={true}/></div>
  </div>
  if (!localBridge()) return <BrowserApp />
  const empty = {sessionEmail:'Local workspace', collaborators:[], agentRuns:[], agentChanges:[], projects:[], studioUsers:[],ready:false,failed:false,reducers:null,procedures:null} as unknown as ReturnType<typeof useSpacetime>
  return <StudioApp stdb={empty} />
}

function StudioApp({stdb}: {stdb: ReturnType<typeof useSpacetime>}) {
  const { theme, toggleTheme } = useTheme()
  const [chats, setChats] = useState<Chat[]>([
    { id: 'default', title: 'New video', messages: [WELCOME], createdAt: Date.now() },
  ])
  const [activeChatId, setActiveChatId] = useState('default')
  const [input, setInput] = useState('')
  const [generating, setGenerating] = useState(false)
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const [editingChatId, setEditingChatId] = useState<string | null>(null)
  const [editTitle, setEditTitle] = useState('')
  const [panelOpen, setPanelOpen] = useState(true)
  const [panelTab, setPanelTab] = useState<PanelTab>('video')
  const [selectedAgent, setSelectedAgent] = useState<Artifact | null>(null)
  const [playing, setPlaying] = useState(false)
  const [revealedAgents, setRevealedAgents] = useState<AgentNote[]>([])
  const [activeAgentIdx, setActiveAgentIdx] = useState(-1)
  const [expandedContentId, setExpandedContentId] = useState<string | null>(null)
  const [targetAgent, setTargetAgent] = useState<Artifact | null>(null)
  const [targetPartId, setTargetPartId] = useState<string | null>(null)
  const revealTimerRef = useRef<number | null>(null)
  const videoRef = useRef<HTMLVideoElement>(null)
  const bottomRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLTextAreaElement>(null)
  const desktop = !!localBridge()
  const [backend, setBackend] = useState<LocalRow>({})
  const [backendError, setBackendError] = useState('')
  const [actionError, setActionError] = useState('')
  const [remotePod, setRemotePod] = useState('')
  const [remoteUrl, setRemoteUrl] = useState('')
  const [localBusy, setLocalBusy] = useState(false)
  const refreshBackend = async () => {
    if (!desktop) return
    try { const next = await localBridge()!.request('/api/state'); setBackend(next); setBackendError('');setChats(prev=>[...prev,...localRows(next.projects).filter(p=>!prev.some(c=>c.id===p.id)).map(p=>({id:String(p.id),title:String(p.title||'New video'),createdAt:Number(p.createdAt)||Date.now(),messages:[WELCOME]}))]) }
    catch(e) {setBackendError(e instanceof Error ? e.message : 'Backend unavailable')}
  }
  useEffect(() => { if(!desktop)return; void refreshBackend(); const timer=window.setInterval(()=>{if(!document.hidden)void refreshBackend()},3000);return()=>window.clearInterval(timer) }, [desktop])
  const localAction = async(path:string,body:LocalRow) => {
    if(localBusy)return;setLocalBusy(true);setActionError('')
    try {await localBridge()!.request(path,body);await refreshBackend()}
    catch(e){setActionError(e instanceof Error?e.message:'Request failed');await refreshBackend()}
    finally{setLocalBusy(false)}
  }
  const projectJobs = localRows(backend.jobs).filter(j=>j.projectId===activeChatId)
  const {current:currentRun,history:pastRuns} = latestRuns(projectJobs)
  const latestSwarm = latestRuns(projectJobs.filter(j=>j.kind==='agent_swarm')).current
  const newestTeam = latestRuns(localRows(backend.jobs).filter(j=>j.kind==='agent_swarm')).current
  const newestTeamSession = chats.find(c=>c.id===newestTeam?.projectId)
  const visibleJobs = projectJobs.flatMap(j=>j.kind==='agent_swarm' && Array.isArray(j.workers)?[...j.workers.map((w:LocalRow)=>({...w,id:`${j.id}_${w.id}`,parentRunId:j.id,projectId:j.projectId,agentId:w.roleId,kind:'agent_worker',stage:w.summary||w.task})),j]:[j])
  const referencePrompt = input.trim() || [...projectJobs].reverse().find(j=>j.kind==='plan' && typeof j.prompt==='string')?.prompt || ''
  const projectMedia = localRows(backend.media).filter(m=>m.projectId===activeChatId).filter((m,i,rows)=>!m.assetId||rows.findIndex(other=>other.assetId===m.assetId)===i)
  const projectParts = localRows(backend.parts).filter(p=>p.projectId===activeChatId)
  const projectAgents = localRows(backend.agents).filter(a=>a.projectId===activeChatId)
  const appleReplacement = appleReplacementReadiness(backend.capabilities?.appleReplacement)
  const remoteError = remoteFailure(backend.remote)
  const roleState = (agent: Artifact) => projectAgents.find(a => a.roleId===AGENT_ROLE_IDS[agent.id] || a.id===AGENT_ROLE_IDS[agent.id])
  const roleArtwork = (id: unknown) => AGENTS.find(a=>a.id===id || AGENT_ROLE_IDS[a.id]===id)
  const openEnvironment = () => {setPanelOpen(true);setPanelTab('environment')}
  const assignRunTask = (role:string,part:string) => {const agent=roleArtwork(role);if(agent)taskAgent(agent,{id:part})}
  const partArtwork = (part: LocalRow) => roleArtwork(part.ownerRoleId || part.roleId || part.agentId) || AGENTS.find(a=>a.id==='59')!
  const authedUser = stdb.sessionEmail
  const onlineCollaborators = stdb.collaborators.filter(c => c.online)

  useEffect(() => {
    if (authedUser) {
      stdb.reducers?.setNickname({ nickname: authedUser.split('@')[0], color: pickColor() })
    }
  }, [authedUser])

  const activeChat = chats.find(c => c.id === activeChatId)!

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [activeChat.messages])

  useEffect(() => {
    setTargetAgent(null)
    setTargetPartId(null)
    setActionError('')
    inputRef.current?.focus()
  }, [activeChatId])

  const taskAgent = (agent: Artifact, part?: LocalRow) => {
    setTargetAgent(agent)
    setTargetPartId(part ? String(part.id) : null)
    setInput('')
    inputRef.current?.focus()
  }

  const handleSend = async () => {
    if(!input.trim() || generating)return
    const prompt=input.trim()
    const assignment=targetAgent?{agentId:AGENT_ROLE_IDS[targetAgent.id],...(targetPartId?{partId:targetPartId}:{})}:null
    const visiblePrompt=targetAgent?`@${targetAgent.agent}${targetPartId?` / ${targetPartId}`:''}: ${prompt}`:prompt
    const chatId=activeChatId
    setChats(prev=>prev.map(c=>c.id===chatId?{...c,title:c.messages.length<=1?prompt.slice(0,40):c.title,messages:[...c.messages,{id:makeId(),role:'user',content:visiblePrompt}]}:c))
    setInput('');setTargetAgent(null);setTargetPartId(null);setGenerating(true);setPanelOpen(true)
    try {
      if(!desktop)throw new Error('Connect the desktop Gaussian backend to submit a real generation job.')
      await localBridge()!.request('/api/jobs',assignment?{kind:'agent_task',prompt,projectId:chatId,...assignment}:{kind:'plan',prompt,projectId:chatId})
      await refreshBackend()
    }catch(e){setChats(prev=>prev.map(c=>c.id===chatId?{...c,messages:[...c.messages,{id:makeId(),role:'assistant',content:e instanceof Error?e.message:'Submission failed'}]}:c))}
    finally{setGenerating(false)}
  }


  const newChat = async () => {
    const id = makeId()
    if(desktop){try{await localBridge()!.request('/api/projects',{id,title:'New video'})}catch(e){setBackendError(e instanceof Error?e.message:'Could not create session');return}}
    setChats(prev => [
      { id, title: 'New video', messages: [WELCOME], createdAt: Date.now() },
      ...prev,
    ])
    setActiveChatId(id)
  }

  const deleteChat = async (id: string) => {
    if(desktop){try{await localBridge()!.request('/api/projects',{id,title:chats.find(c=>c.id===id)?.title||'New video',archived:true})}catch(e){setBackendError(e instanceof Error?e.message:'Could not archive session');return}}
    setChats(prev => {
      const next = prev.filter(c => c.id !== id)
      if (next.length === 0) {
        const fresh: Chat = { id: makeId(), title: 'New video', messages: [WELCOME], createdAt: Date.now() }
        setActiveChatId(fresh.id)
        return [fresh]
      }
      if (activeChatId === id) setActiveChatId(next[0].id)
      return next
    })
  }

  const startRename = (id: string, currentTitle: string) => {
    setEditingChatId(id)
    setEditTitle(currentTitle)
  }

  const confirmRename = async () => {
    if (!editingChatId) return
    const nextTitle=editTitle.trim()||chats.find(c=>c.id===editingChatId)?.title||'New video'
    if(desktop){try{await localBridge()!.request('/api/projects',{id:editingChatId,title:nextTitle})}catch(e){setBackendError(e instanceof Error?e.message:'Could not rename session');return}}
    setChats(prev =>
      prev.map(c => (c.id === editingChatId ? { ...c, title: editTitle.trim() || c.title } : c))
    )
    setEditingChatId(null)
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  const isDark = theme === 'dark'

  const selectAgent = (a: Artifact) => {
    setSelectedAgent(a)
    setPlaying(false)
  }

  const togglePlay = () => {
    if (!videoRef.current) return
    if (playing) { videoRef.current.pause() } else { videoRef.current.play() }
    setPlaying(!playing)
  }

  if (!authedUser) {
    return (
      <div className={isDark ? 'dark' : 'light'} style={{ width: '100vw', height: '100vh', background: 'var(--canvas)' }}>
        {(stdb.ready || stdb.failed) && <SignInGate stdb={stdb} />}
      </div>
    )
  }

  return (
    <div className={isDark ? 'dark' : 'light'}>
      <div style={{
        display: 'flex',
        width: '100vw',
        height: '100vh',
        background: 'var(--canvas)',
        color: 'var(--text)',
      }}>

        {/* Sidebar */}
        {sidebarOpen && (
          <aside style={{
            width: 258,
            height: '100%',
            display: 'flex',
            flexDirection: 'column',
            background: 'var(--sidebar)',
            borderRight: '1px solid var(--line)',
            flexShrink: 0,
          }}>
            <div className="drag-region" style={{ height: 38, display: 'flex', alignItems: 'center', padding: '0 12px', gap: 8 }}>
              <div style={{ width: 68 }} />
            </div>

            <div style={{ padding: '0 10px 8px' }}>
              <button
                onClick={newChat}
                className="no-drag"
                style={{
                  width: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center',
                  gap: 6, padding: '6px 12px', border: '1px solid #464646', borderRadius: 3,
                  background: '#252525', color: 'var(--text)', fontSize: 12, fontWeight: 500,
                  letterSpacing: '0.02em', cursor: 'pointer', minHeight: 30,
                }}
                onMouseEnter={e => { e.currentTarget.style.background = 'var(--hover)' }}
                onMouseLeave={e => { e.currentTarget.style.background = '#252525' }}
              >
                <Plus size={14} />
                New session
              </button>
            </div>

            <div style={{ padding: '8px 14px 4px', fontSize: 10, fontWeight: 600, letterSpacing: '0.06em', textTransform: 'uppercase', color: 'var(--faint)' }}>
              Sessions
            </div>

            <nav style={{ flex: 1, overflowY: 'auto', padding: '2px 6px' }}>
              {chats.map(chat => (
                <div
                  key={chat.id}
                  onClick={() => setActiveChatId(chat.id)}
                  style={{
                    display: 'flex', alignItems: 'center', gap: 8, padding: '6px 8px',
                    borderRadius: 3, cursor: 'pointer',
                    background: activeChatId === chat.id ? 'var(--selected)' : 'transparent',
                    marginBottom: 1,
                  }}
                  onMouseEnter={e => { if (activeChatId !== chat.id) e.currentTarget.style.background = 'var(--hover)' }}
                  onMouseLeave={e => { if (activeChatId !== chat.id) e.currentTarget.style.background = 'transparent' }}
                >
                  {editingChatId === chat.id ? (
                    <div style={{ flex: 1, display: 'flex', alignItems: 'center', gap: 4 }}>
                      <input
                        value={editTitle}
                        onChange={e => setEditTitle(e.target.value)}
                        onKeyDown={e => e.key === 'Enter' && confirmRename()}
                        autoFocus
                        style={{ flex: 1, background: 'transparent', border: 'none', borderBottom: '1px solid var(--line-strong)', outline: 'none', color: 'var(--text)', fontSize: 12, padding: '0 0 2px' }}
                        onClick={e => e.stopPropagation()}
                      />
                      <button onClick={e => { e.stopPropagation(); confirmRename() }} style={{ padding: 2, color: 'var(--muted)', background: 'none', border: 'none', cursor: 'pointer', minHeight: 'auto' }}>
                        <Check size={12} />
                      </button>
                      <button onClick={e => { e.stopPropagation(); setEditingChatId(null) }} style={{ padding: 2, color: 'var(--muted)', background: 'none', border: 'none', cursor: 'pointer', minHeight: 'auto' }}>
                        <X size={12} />
                      </button>
                    </div>
                  ) : (
                    <>
                      <span style={{ flex: 1, fontSize: 12, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{chat.title}{desktop&&newestTeam?.projectId===chat.id&&<span style={{display:'block',fontSize:10,color:'var(--faint)',marginTop:3}}>Latest team · {newestTeam.status}</span>}</span>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 2, opacity: 0 }} className="chat-actions">
                        <button onClick={e => { e.stopPropagation(); startRename(chat.id, chat.title) }} style={{ padding: 3, color: 'var(--faint)', background: 'none', border: 'none', cursor: 'pointer', minHeight: 'auto', borderRadius: 2 }}>
                          <Pencil size={11} />
                        </button>
                        <button title={desktop?'Archive session (preserves artifacts)':'Delete session'} onClick={e => { e.stopPropagation(); void deleteChat(chat.id) }} style={{ padding: 3, color: 'var(--faint)', background: 'none', border: 'none', cursor:'pointer', minHeight: 'auto', borderRadius: 2 }}>
                          <Trash2 size={11} />
                        </button>
                      </div>
                    </>
                  )}
                </div>
              ))}
            </nav>

            <div style={{ padding: '8px 10px', borderTop: '1px solid var(--line)' }}>
              {authedUser && (
                <div style={{ display: 'flex', alignItems: 'center', gap: 6, padding: '4px 8px', marginTop: 4 }}>
                  <span style={{ fontSize: 10, color: 'var(--faint)', flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{authedUser}</span>
                  {!desktop && <button
                    onClick={() => stdb.reducers?.signOut({})}
                    className="no-drag"
                    style={{ fontSize: 10, color: 'var(--faint)', background: 'none', border: 'none', cursor: 'pointer', padding: 0, minHeight: 'auto', textDecoration: 'underline' }}
                  >
                    Sign out
                  </button>}
                </div>
              )}
              <button
                onClick={toggleTheme}
                className="no-drag"
                style={{ width: '100%', display: 'flex', alignItems: 'center', gap: 8, padding: '6px 8px', border: 'none', borderRadius: 3, background: 'transparent', color: 'var(--muted)', fontSize: 12, cursor: 'pointer', minHeight: 'auto', marginTop: 2 }}
                onMouseEnter={e => { e.currentTarget.style.background = 'var(--hover)' }}
                onMouseLeave={e => { e.currentTarget.style.background = 'transparent' }}
              >
                {isDark ? <Sun size={13} /> : <Moon size={13} />}
                {isDark ? 'Light mode' : 'Dark mode'}
              </button>
            </div>
          </aside>
        )}

        {/* Main chat area */}
        <main style={{ flex: 1, display: 'flex', flexDirection: 'column', height: '100%', minWidth: 0 }}>
          {/* Top bar */}
          <div className="drag-region" style={{
            height: 38, display: 'flex', alignItems: 'center', padding: '0 16px',
            borderBottom: '1px solid var(--line)', flexShrink: 0, background: 'var(--chrome)',
          }}>
            {!sidebarOpen && (
              <div className="no-drag" style={{ display: 'flex', alignItems: 'center', gap: 8, marginRight: 12 }}>
                <div style={{ width: 68 }} />
                <button
                  onClick={() => setSidebarOpen(true)}
                  style={{ padding: '4px 8px', border: 'none', borderRadius: 3, background: 'transparent', color: 'var(--muted)', cursor: 'pointer', fontSize: 12, minHeight: 'auto' }}
                  onMouseEnter={e => { e.currentTarget.style.background = 'var(--hover)' }}
                  onMouseLeave={e => { e.currentTarget.style.background = 'transparent' }}
                >
                  Sidebar
                </button>
                <button
                  onClick={newChat}
                  style={{ padding: '4px 8px', border: '1px solid #464646', borderRadius: 3, background: '#252525', color: 'var(--text)', cursor: 'pointer', fontSize: 12, minHeight: 'auto' }}
                  onMouseEnter={e => { e.currentTarget.style.background = 'var(--hover)' }}
                  onMouseLeave={e => { e.currentTarget.style.background = '#252525' }}
                >
                  <Plus size={12} />
                </button>
              </div>
            )}
            <div style={{ flex: 1, textAlign: 'center' }}>
              <span className="no-drag" style={{ fontSize: 12, fontWeight: 500, color: 'var(--muted)', letterSpacing: '0.02em' }}>
                {desktop?'Session: ':''}{activeChat.title}
              </span>
            </div>
            <div className="no-drag" style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              {onlineCollaborators.length > 0 && (
                <div style={{ display: 'flex', alignItems: 'center' }}>
                  {onlineCollaborators.map((c, i) => (
                    <div
                      key={i}
                      title={c.nickname}
                      style={{
                        width: 22, height: 22, borderRadius: '50%',
                        background: c.color, border: '2px solid var(--chrome)',
                        display: 'flex', alignItems: 'center', justifyContent: 'center',
                        fontSize: 10, fontWeight: 700, color: '#000',
                        marginLeft: i > 0 ? -6 : 0,
                      }}
                    >
                      {c.nickname.charAt(0).toUpperCase()}
                    </div>
                  ))}
                </div>
              )}
              <button
                onClick={() => setPanelOpen(!panelOpen)}
                style={{
                  padding: '4px 8px', border: 'none', borderRadius: 3,
                  background: panelOpen ? 'var(--selected)' : 'transparent',
                  color: 'var(--muted)', cursor: 'pointer', fontSize: 12, minHeight: 'auto',
                  display: 'flex', alignItems: 'center', gap: 5,
                }}
                onMouseEnter={e => { if (!panelOpen) e.currentTarget.style.background = 'var(--hover)' }}
                onMouseLeave={e => { if (!panelOpen) e.currentTarget.style.background = 'transparent' }}
              >
                {panelOpen ? <PanelRightClose size={14} /> : <PanelRight size={14} />}
                Workspace
              </button>
            </div>
          </div>

          {desktop && <AgentSpotlight agents={AGENTS.map(a=>({...a,desc:roleState(a)?.task||a.desc}))} roleIds={AGENT_ROLE_IDS} jobs={visibleJobs} selected={targetAgent || selectedAgent} onSelect={agent=>{const artwork=AGENTS.find(a=>a.id===agent.id);if(artwork)taskAgent(artwork)}} stale={!!backendError} onHistory={()=>{setPanelOpen(true);setPanelTab('timeline')}}/>}

          {/* Content row: messages + optional panel */}
          <div style={{ flex: 1, display: 'flex', overflow: 'hidden' }}>

            {/* Messages column */}
            <div style={{ flex: 1, display: 'flex', flexDirection: 'column', minWidth: 0 }}>
              <div style={{ flex: 1, overflowY: 'auto', background: 'var(--canvas)' }}>
                <div style={{ maxWidth: 680, margin: '0 auto', padding: '24px 20px' }}>
                  {desktop&&newestTeamSession&&newestTeamSession.id!==activeChatId&&<div style={{display:'flex',alignItems:'center',gap:12,padding:'10px 12px',border:'1px solid var(--line)',borderRadius:6,marginBottom:18,fontSize:11}}>
                    <div style={{flex:1,minWidth:0,color:'var(--muted)'}}>Latest team is in {newestTeamSession.title}<div style={{fontSize:10,color:'var(--faint)',marginTop:3}}>{newestTeamSession.id} · {newestTeam.status} · {runTimestamp(newestTeam.createdAt)}</div></div>
                    <button onClick={()=>setActiveChatId(newestTeamSession.id)} style={{...gateLink,width:'auto',marginTop:0,padding:'6px 8px',border:'1px solid var(--line)',color:'var(--text)'}}>Open latest team session</button>
                  </div>}
                  {desktop&&<div style={{marginBottom:18}}><button disabled={localBusy||projectJobs.some(j=>['queued','running'].includes(j.status))||!backend.credentials?.gemini?.configured||!backend.credentials?.elevenlabs?.configured} onClick={()=>void localAction('/api/jobs',{kind:'agent_swarm',projectId:activeChatId,prompt:input.trim()||'Create a polished sound-design and Gaussian-edit preparation team for the existing five-second donkey eating an orange clip. Protect the donkey and environment. Plan orange-to-apple replacement including slices, peel and mouth contact, without claiming it executed. Allocate at least three separate sound specialists for quiet outdoor ambience, donkey chewing/foley and delicate fruit/bowl handling. Keep the layers isolated, natural and without speech or music.'})} style={{border:'1px solid var(--line)',background:'var(--panel)',color:'var(--text)',padding:'10px 14px',borderRadius:7,cursor:'pointer'}}>Start scene team</button><div style={{fontSize:10,color:'var(--muted)',marginTop:5}}>Donkey source · Gemini + 3–4 sound effects · uses API credits</div></div>}
                  {desktop&&currentRun&&<section aria-label="Current run" data-current-run={currentRun.id} style={{marginBottom:22,border:'1px solid var(--line-strong)',borderRadius:8,padding:14}}>
                    <div style={{display:'flex',alignItems:'center',gap:10,marginBottom:6}}><span style={{fontSize:10,fontWeight:600,letterSpacing:'0.06em',textTransform:'uppercase',color:'var(--faint)',flex:1}}>Current run</span><span style={{fontSize:11,color:/blocked|fail|error|reject/i.test(String(currentRun.status))?'#ef9999':'var(--muted)'}}>{currentRun.status}</span></div>
                    <div style={{fontSize:14,fontWeight:600}}>{runLabel(currentRun,roleArtwork(currentRun.agentId))}</div>
                    <div style={{fontSize:10,color:'var(--faint)',marginTop:5,marginBottom:10}}>Submitted {runTimestamp(currentRun.createdAt)}</div>
                    {currentRun.kind!=='agent_swarm'&&currentRun.stage&&!runProblem(currentRun)&&<div style={{fontSize:12,lineHeight:1.5,color:'var(--muted)'}}>{currentRun.stage}</div>}
                    <RunProblem job={currentRun} onEnvironment={openEnvironment}/>
                    {currentRun.kind==='agent_swarm'&&currentRun.status==='blocked'&&<button disabled={localBusy} onClick={()=>void localAction('/api/jobs',{kind:'agent_swarm',projectId:activeChatId,prompt:currentRun.prompt,resumeJobId:currentRun.id})} style={{padding:'8px 12px',marginTop:10,marginBottom:14,color:'var(--text)',background:'var(--panel)',border:'1px solid var(--line)',borderRadius:6}}>Resume unfinished tasks</button>}
                    {currentRun.kind==='agent_swarm'&&<LiveSwarm key={currentRun.id} job={currentRun} artwork={roleArtwork} onAssign={assignRunTask}/>}
                    <RunDetails job={currentRun} artwork={roleArtwork(currentRun.agentId)}/>
                  </section>}
                  {activeChat.messages.filter(msg=>!currentRun||msg.id!==WELCOME.id).map(msg => (
                    <div key={msg.id} style={{ display: 'flex', gap: 12, marginBottom: 20 }}>
                      <div style={{
                        width: 28, height: 28, borderRadius: '50%', display: 'flex',
                        alignItems: 'center', justifyContent: 'center', flexShrink: 0,
                        background: msg.role === 'user' ? 'var(--selected)' : 'var(--panel)',
                        border: '1px solid var(--line)', fontSize: 11, fontWeight: 600, color: 'var(--muted)',
                      }}>
                        {msg.role === 'user' ? 'U' : 'A'}
                      </div>
                      <div style={{ flex: 1, minWidth: 0, paddingTop: 2 }}>
                        <span style={{ display: 'block', fontSize: 11, fontWeight: 600, color: 'var(--faint)', marginBottom: 4, letterSpacing: '0.02em' }}>
                          {msg.role === 'user' ? 'You' : 'Agent'}
                        </span>
                        <div style={{ fontSize: 13, lineHeight: 1.6, whiteSpace: 'pre-wrap', color: 'var(--text)' }}>
                          {msg.content}
                        </div>
                      </div>
                    </div>
                  ))}

                  {desktop && backendError && <div role="alert" style={{fontSize:12,color:'#f87171',marginBottom:16}}>{backendError}</div>}
                  {desktop && actionError && <div role="alert" style={{display:'flex',alignItems:'start',gap:10,fontSize:12,color:'#f87171',marginBottom:16}}><span style={{flex:1}}>{actionError}</span><button aria-label="Dismiss action error" onClick={()=>setActionError('')} style={{padding:0,background:'none',border:0,color:'inherit',cursor:'pointer'}}><X size={14}/></button></div>}
                  {desktop&&<RunHistory key={activeChatId} jobs={pastRuns} artwork={roleArtwork} onEnvironment={openEnvironment} onAssign={assignRunTask}/>}
                  {!desktop && revealedAgents.length > 0 && (
                    <div style={{ display: 'flex', gap: 12, marginBottom: 20 }}>
                      <div style={{
                        width: 28, height: 28, borderRadius: '50%', display: 'flex',
                        alignItems: 'center', justifyContent: 'center', flexShrink: 0,
                        background: 'var(--panel)', border: '1px solid var(--line)',
                        fontSize: 11, fontWeight: 600, color: 'var(--muted)',
                      }}>A</div>
                      <div style={{ flex: 1, minWidth: 0, paddingTop: 2 }}>
                        {revealedAgents.slice(0, activeAgentIdx >= 0 ? activeAgentIdx : revealedAgents.length).map((an, i) => (
                          <DoneAgentRow key={an.agentId + i} agentId={an.agentId} isDark={isDark} />
                        ))}
                        {activeAgentIdx >= 0 && (
                          <div style={{ marginTop: activeAgentIdx > 0 ? 8 : 0 }}>
                            <ActiveAgentBanner
                              key={revealedAgents[activeAgentIdx].agentId}
                              agentId={revealedAgents[activeAgentIdx].agentId}
                              note={revealedAgents[activeAgentIdx].note}
                              isDark={isDark}
                            />
                          </div>
                        )}
                      </div>
                    </div>
                  )}

                  {generating && revealedAgents.length === 0 && (
                    <div style={{ display: 'flex', gap: 12, marginBottom: 20 }}>
                      <div style={{
                        width: 28, height: 28, borderRadius: '50%', display: 'flex',
                        alignItems: 'center', justifyContent: 'center', flexShrink: 0,
                        background: 'var(--panel)', border: '1px solid var(--line)',
                        fontSize: 11, fontWeight: 600, color: 'var(--muted)',
                      }}>A</div>
                      <div style={{ paddingTop: 8, display: 'flex', gap: 4 }}>
                        <span style={{ width: 6, height: 6, borderRadius: '50%', background: 'var(--faint)', animation: 'pulse 1.2s infinite', animationDelay: '0ms' }} />
                        <span style={{ width: 6, height: 6, borderRadius: '50%', background: 'var(--faint)', animation: 'pulse 1.2s infinite', animationDelay: '200ms' }} />
                        <span style={{ width: 6, height: 6, borderRadius: '50%', background: 'var(--faint)', animation: 'pulse 1.2s infinite', animationDelay: '400ms' }} />
                      </div>
                    </div>
                  )}

                  <div ref={bottomRef} />
                </div>
              </div>

              {/* Input area */}
              <div style={{ flexShrink: 0, borderTop: '1px solid var(--line)', padding: '12px 20px', background: 'var(--chrome)' }}>
                <div style={{ maxWidth: 680, margin: '0 auto' }}>
                  {targetAgent && (
                    <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 6, padding: '4px 8px', borderRadius: 3, background: 'var(--panel)', border: '1px solid var(--line)', width: 'fit-content' }}>
                      <img src={targetAgent.gif} alt="" style={{ width: 20, height: 12, borderRadius: 2, objectFit: 'cover' }} />
                      <span style={{ fontSize: 11, fontWeight: 600, color: 'var(--text)' }}>@{targetAgent.agent}{targetPartId?` / ${targetPartId}`:''}</span>
                      <button
                        onClick={() => {setTargetAgent(null);setTargetPartId(null)}}
                        style={{ background: 'none', border: 'none', padding: 0, cursor: 'pointer', display: 'flex', minHeight: 'auto' }}
                      >
                        <X size={12} style={{ color: 'var(--faint)' }} />
                      </button>
                    </div>
                  )}
                  <div style={{ display: 'flex', alignItems: 'flex-end', gap: 8, border: '1px solid var(--line)', borderRadius: 3, background: 'var(--panel)', padding: '8px 12px' }}>
                    <textarea
                      ref={inputRef}
                      value={input}
                      onChange={e => setInput(e.target.value)}
                      onKeyDown={handleKeyDown}
                      placeholder={targetAgent ? `Task ${targetAgent.agent}...` : 'Describe the video you want to create...'}
                      rows={1}
                      style={{ flex: 1, background: 'transparent', border: 'none', outline: 'none', color: 'var(--text)', fontSize: 13, lineHeight: 1.5, resize: 'none', maxHeight: 128, minHeight: 20 }}
                      onInput={e => {
                        const t = e.currentTarget
                        t.style.height = 'auto'
                        t.style.height = Math.min(t.scrollHeight, 128) + 'px'
                      }}
                    />
                    <button
                      onClick={handleSend}
                      disabled={!input.trim() || generating}
                      style={{
                        padding: '5px 14px', border: '1px solid #dedede', borderRadius: 3,
                        background: '#dedede', color: '#181818', fontSize: 12, fontWeight: 600,
                        cursor: input.trim() && !generating ? 'pointer' : 'not-allowed',
                        opacity: !input.trim() || generating ? 0.35 : 1,
                        display: 'flex', alignItems: 'center', gap: 5, flexShrink: 0, minHeight: 28,
                      }}
                    >
                      {generating ? <Loader2 size={13} className="animate-spin" /> : <Send size={13} />}
                      Send
                    </button>
                  </div>
                  <p style={{ fontSize: 10, color: 'var(--faint)', textAlign: 'center', marginTop: 6 }}>
                    {targetAgent?'Send to '+targetAgent.agent:'Enter to send · Shift + Enter for a new line'}
                  </p>
                </div>
              </div>
            </div>

            {/* Right panel (browser) */}
            {panelOpen && (
              <div style={{
                width: panelTab==='timeline'?500:440, maxWidth:'48%', flexShrink: 0, display: 'flex', flexDirection: 'column',
                borderLeft: '1px solid var(--line)', background: 'var(--chrome)',
              }}>
                {/* Panel tabs */}
                <div style={{
                  display: 'flex', borderBottom: '1px solid var(--line)', flexShrink: 0,
                }}>
                  {(['video', 'contents', 'timeline', 'environment'] as PanelTab[]).map(tab => (
                    <button
                      key={tab}
                      onClick={() => setPanelTab(tab)}
                      style={{
                        flex: 1, padding: '8px 0', border: 'none', borderRadius: 0,
                        background: 'transparent', color: panelTab === tab ? 'var(--text)' : 'var(--faint)',
                        fontSize: 11, fontWeight: 600, letterSpacing: '0.04em', textTransform: 'uppercase',
                        cursor: 'pointer', minHeight: 'auto',
                        borderBottom: panelTab === tab ? '2px solid var(--text)' : '2px solid transparent',
                      }}
                    >
                      {tab==='timeline'?'Agent Timeline':tab.charAt(0).toUpperCase() + tab.slice(1)}
                    </button>
                  ))}
                </div>

                {/* Panel content */}
                <div style={{ flex: 1, overflowY: 'auto', padding: 0 }}>

                  {desktop && panelTab === 'video' && <div>
                    {latestSwarm&&localRows(latestSwarm.workers).filter(w=>w.roleId==='sound-agents'&&w.status==='completed'&&w.execution==='elevenlabs_sound'&&w.mediaId).length>=3&&<SceneAudioMixer key={latestSwarm.id} job={latestSwarm} busy={localBusy||projectJobs.some(j=>['queued','running'].includes(j.status))} onExport={gains=>void localAction('/api/jobs',{kind:'sound_mix',projectId:activeChatId,sourceJobId:latestSwarm.id,gains})}/>}
                    {!projectMedia.length && <div style={{aspectRatio:'16/9',background:'#000',display:'flex',alignItems:'center',justifyContent:'center',fontSize:12,color:'#777'}}>No video generated for this session</div>}
                    {projectMedia.filter(m=>m.kind!=='image').map(m=><div key={m.id} style={{borderBottom:'1px solid var(--line)',paddingBottom:10}}>
                      {safeMedia(m.url) ? m.kind==='video'?<video controls preload="metadata" src={safeMedia(m.url)} style={{width:'100%',background:'#000'}}/>:m.kind==='audio'?<audio controls preload="metadata" src={safeMedia(m.url)} style={{width:'100%',padding:10}} aria-label={String(m.title||'Generated sound effect')}/>:<img src={safeMedia(m.url)} alt={String(m.title||'Reference')} style={{width:'100%',objectFit:'contain',background:'#000'}}/>:<div>Artifact URL unavailable</div>}
                      <div style={{padding:'8px 12px',fontSize:12}}>{m.title}<div style={{fontSize:10,color:'var(--faint)',marginTop:4}}>{typeof m.review==='string'?m.review:'Not reviewed'}</div></div>
                    </div>)}
                    {projectMedia.filter(m=>m.kind==='image').map(m=><details key={m.id} style={{padding:12,borderBottom:'1px solid var(--line)',fontSize:12}}><summary style={{cursor:'pointer'}}>{m.title}</summary>{safeMedia(m.url)&&<img src={safeMedia(m.url)} alt={m.title} style={{width:'100%',marginTop:8}}/>}<small style={{color:'var(--muted)'}}>{m.review}</small></details>)}
                    <div style={{padding:'12px 12px 6px',fontSize:11,color:'var(--faint)'}}>Agent protocol · {AGENTS.length} specialist roles</div>
                    {AGENTS.map(a=><DesktopAgentCard key={a.id} agent={a} state={roleState(a)} parts={projectParts} onTask={()=>taskAgent(a)}/>)}
                  </div>}
                  {desktop && panelTab === 'contents' && <div style={{padding:12}}>
                    <div style={{fontSize:10,color:'var(--faint)',textTransform:'uppercase',marginBottom:10}}>Parts & tasks</div>
                    {!projectParts.length&&<p style={{fontSize:12,color:'var(--muted)'}}>No parts assigned.</p>}
                    {projectParts.map(p=><div key={p.id} style={{padding:'10px 8px',borderBottom:'1px solid var(--line-soft)',fontSize:12}}>
                      <strong>{p.label||p.id}</strong>
                      <div style={{fontSize:10,color:'var(--faint)',marginTop:4}}>{partContentState(p,projectJobs,projectMedia).label}</div>
                      {p.bindingVerified===true && partContentState(p,projectJobs,projectMedia).requiresGeometryBinding && <div style={{fontSize:11,color:'var(--muted)',lineHeight:1.5,marginTop:7}}>
                        <div>Binding scope: {p.bindingScope || 'Canonical asset only; scene placement and contact are unverified.'}</div>
                        <div style={{marginTop:4,color:/reject|coarse|fail/i.test(String(p.geometryQuality||''))?'#ef9999':'var(--faint)'}}>Geometry quality: {p.geometryQuality || 'Not accepted or visually verified.'}</div>
                      </div>}
                      <div style={{display:'flex',alignItems:'center',gap:8,marginTop:8}}>
                        <img src={partArtwork(p).gif} alt="Owner role illustration" style={{width:40,height:23,objectFit:'cover',borderRadius:3}}/>
                        <div style={{fontSize:11,color:'var(--muted)'}}>{p.ownerRoleId||p.roleId ? partArtwork(p).agent : p.agentId || 'Owner not yet assigned'}<div style={{fontSize:10,color:'var(--faint)',marginTop:3}}>{p.task || 'No task yet'}</div></div>
                      </div>
                      <button onClick={()=>taskAgent(partArtwork(p),p)} style={{...gateLink,textAlign:'left',padding:0}}>Assign task for this part</button>
                    </div>)}
                  </div>}
                  {desktop && panelTab === 'timeline' && <AgentTimeline projectId={activeChatId} jobs={visibleJobs} recentRunIds={[currentRun?.id,latestSwarm?.id].filter(Boolean)} agents={projectAgents} parts={projectParts} dark={isDark} onRun={async task=>{await localBridge()!.request('/api/jobs',{kind:'agent_task',projectId:activeChatId,...task});await refreshBackend()}}/>}
                  {desktop && panelTab === 'environment' && <div style={{padding:12,fontSize:12}}>
                    <div style={{fontSize:10,color:'var(--faint)',textTransform:'uppercase',marginBottom:10}}>Models & remote environment</div>
                    <p style={{color:'var(--muted)',marginBottom:12}}>Shared owner credentials · encrypted locally</p>
                    <div style={{border:'1px solid var(--line)',borderRadius:3,marginBottom:14}}>
                      {['gemini','runway','runpod','elevenlabs'].map(provider=><div key={provider} style={{padding:'8px 10px',borderBottom:'1px solid var(--line-soft)'}}>
                        <span style={{textTransform:'capitalize',fontWeight:600}}>{provider}</span>
                        <span style={{float:'right',fontSize:11,color:backend.credentials?.[provider]?.configured?'var(--muted)':'var(--faint)'}}>{backend.credentials?.[provider]?.configured?'Key saved':'Not configured'}</span>
                        {backend.credentials?.[provider]?.error&&<div role="alert" style={{fontSize:10,color:'#ef9999',marginTop:4}}>Provider unavailable. Owner configuration needs attention.</div>}
                      </div>)}
                    </div>
                    <ElevenLabsSettings configured={!!backend.credentials?.elevenlabs?.configured} save={async key=>{await localBridge()!.request('/api/credentials',{provider:'elevenlabs',key});await refreshBackend()}}/>
                    <p style={{fontSize:10,color:'var(--faint)',marginBottom:14}}>Keys are checked when a task runs.</p>
                    <label style={gateLabel}>Existing pod<input value={remotePod} onChange={e=>setRemotePod(e.target.value)} placeholder={backend.remote?.podId||'Pod ID'} style={gateInput}/></label>
                    <input value={remoteUrl} onChange={e=>setRemoteUrl(e.target.value)} placeholder="https://…proxy.runpod.net" style={{...gateInput,marginBottom:8}}/>
                    <button disabled={localBusy||!remoteUrl||!(remotePod||backend.remote?.podId)} style={gatePrimary} onClick={()=>void localAction('/api/remote/connect',{podId:remotePod||backend.remote?.podId,baseUrl:remoteUrl})}>Connect existing remote</button>
                    <p style={{fontSize:11,color:'var(--faint)',margin:'8px 0'}}>GPU: {backend.remote?.status||'Not connected'}</p>
                    {remoteError&&<div role="alert" style={{fontSize:11,lineHeight:1.5,color:'#ef9999',marginBottom:10}}>{remoteError}{typeof backend.remote?.errorCode==='string'&&<div style={{fontSize:10,marginTop:3}}>Code: {backend.remote.errorCode}</div>}{typeof backend.remote?.checkedAt==='string'&&<div style={{fontSize:10,color:'var(--faint)',marginTop:3}}>Checked: {backend.remote.checkedAt}</div>}</div>}
                    <button disabled={localBusy} style={{...gateLink,border:'1px solid var(--line)'}} onClick={()=>void localAction('/api/jobs',{kind:'remote_preflight',projectId:activeChatId})}>Check remote GPU</button>
                    <button disabled={localBusy||!referencePrompt} style={{...gateLink,border:'1px solid var(--line)'}} onClick={()=>void localAction('/api/jobs',{kind:'gemini_reference',projectId:activeChatId,prompt:referencePrompt})}>Generate project reference (API credits)</button>
                    {activeChatId==='apple-experiment'&&<div style={{border:'1px solid var(--line)',borderRadius:3,padding:10,marginTop:10}}>
                      <div style={{fontSize:11,fontWeight:600}}>Orange → apple scene replacement</div>
                      <div style={{fontSize:11,lineHeight:1.5,color:appleReplacement.ready?'var(--muted)':'#ef9999',marginTop:6}}>{appleReplacement.reason}</div>
                      {appleReplacement.missing.length>0&&<ul style={{fontSize:10,color:'var(--muted)',paddingLeft:16,margin:'8px 0',lineHeight:1.6}}>{appleReplacement.missing.map((item,index)=><li key={index}>{item}</li>)}</ul>}
                      <div style={{fontSize:10,color:'var(--faint)',marginTop:6}}>Pipeline revision: {appleReplacement.revision || 'Not supplied by backend'}</div>
                      <button disabled={localBusy||!appleReplacement.ready} title={appleReplacement.ready?'Run the backend-verified replacement pipeline':'Blocked: do not repeat the rejected static-apple worker'} style={{...gateLink,border:'1px solid var(--line)',opacity:appleReplacement.ready?1:0.5,cursor:appleReplacement.ready?'pointer':'not-allowed'}} onClick={()=>{if(appleReplacement.ready)void localAction('/api/jobs',{kind:'apple_experiment',projectId:activeChatId})}}>{appleReplacement.ready?'Run repaired orange → apple experiment':'Replacement pipeline unavailable'}</button>
                    </div>}
                    <p style={{fontSize:10,color:'var(--faint)',marginTop:8}}>Reference = still image, using the current prompt.</p>
                    <button disabled={localBusy||!backend.remote?.podId} style={{...gateLink,color:'#ef9999'}} onClick={()=>{if(window.confirm('Stop the connected pod? Running work will be interrupted.'))void localAction('/api/remote/stop',{podId:backend.remote?.podId})}}>Stop remote pod</button>
                    <div style={{fontSize:10,color:'var(--faint)',textTransform:'uppercase',margin:'18px 0 8px'}}>Live protocol roles</div>
                    {AGENTS.map(a=><div key={a.id} style={{display:'flex',gap:8,alignItems:'center',padding:'7px 0',borderBottom:'1px solid var(--line-soft)'}}><img src={a.gif} alt="" style={{width:28,height:16,objectFit:'cover',borderRadius:2}}/><span style={{flex:1,fontSize:11}}>{a.agent}</span><span style={{fontSize:10,color:'var(--faint)',maxWidth:130,textAlign:'right'}}>{roleState(a)?.status||'No task submitted'}</span></div>)}
                  </div>}
                  {!desktop && panelTab === 'video' && (
                    <div style={{ padding: 0 }}>
                      {/* Video preview */}
                      <div style={{
                        aspectRatio: '16/9', background: '#000', position: 'relative',
                        display: 'flex', alignItems: 'center', justifyContent: 'center', overflow: 'hidden',
                      }}>
                        {selectedAgent ? (
                          <>
                            <video
                              ref={videoRef}
                              src={selectedAgent.video}
                              poster={selectedAgent.thumb}
                              loop
                              onEnded={() => setPlaying(false)}
                              style={{ width: '100%', height: '100%', objectFit: 'contain' }}
                            />
                            <div
                              onClick={togglePlay}
                              style={{
                                position: 'absolute', inset: 0, display: 'flex',
                                alignItems: 'center', justifyContent: 'center', cursor: 'pointer',
                                background: playing ? 'transparent' : 'rgba(0,0,0,0.3)',
                                transition: 'background 0.2s',
                              }}
                            >
                              {!playing && (
                                <div style={{
                                  width: 48, height: 48, borderRadius: '50%', background: 'rgba(255,255,255,0.15)',
                                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                                }}>
                                  <Play size={22} style={{ color: '#fff', marginLeft: 3 }} />
                                </div>
                              )}
                            </div>
                            <div style={{
                              position: 'absolute', bottom: 0, left: 0, right: 0,
                              padding: '16px 10px 6px', background: 'linear-gradient(transparent, rgba(0,0,0,0.7))',
                            }}>
                              <div style={{ fontSize: 12, color: '#eee', fontWeight: 500 }}>{selectedAgent.agent}</div>
                              <div style={{ fontSize: 10, color: '#999', marginTop: 1 }}>{selectedAgent.model} &middot; {selectedAgent.sizeKB} KB</div>
                            </div>
                          </>
                        ) : (
                          <div style={{ fontSize: 12, color: '#555' }}>Select an agent to preview</div>
                        )}
                      </div>

                      {/* Agent list */}
                      <div style={{ padding: '8px 0' }}>
                        <div style={{ padding: '6px 12px 4px', fontSize: 10, fontWeight: 600, letterSpacing: '0.06em', textTransform: 'uppercase', color: 'var(--faint)' }}>
                          Agents ({AGENTS.length})
                        </div>
                        {AGENTS.map(a => (
                          <div
                            key={a.id}
                            onClick={() => selectAgent(a)}
                            style={{
                              display: 'flex', alignItems: 'center', gap: 8,
                              padding: '5px 12px', cursor: 'pointer',
                              background: selectedAgent?.id === a.id ? 'var(--selected)' : 'transparent',
                            }}
                            onMouseEnter={e => { if (selectedAgent?.id !== a.id) e.currentTarget.style.background = 'var(--hover)' }}
                            onMouseLeave={e => { if (selectedAgent?.id !== a.id) e.currentTarget.style.background = 'transparent' }}
                          >
                            <img
                              src={a.gif}
                              alt=""
                              style={{ width: 32, height: 18, borderRadius: 3, objectFit: 'cover', flexShrink: 0 }}
                            />
                            <div style={{ flex: 1, minWidth: 0 }}>
                              <div style={{ fontSize: 12, fontWeight: 500, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                                {a.agent}
                              </div>
                              <div style={{ fontSize: 10, color: 'var(--faint)', marginTop: 1 }}>
                                {a.model} &middot; {AGENT_COUNTS[a.agent] || '1'} instances
                              </div>
                            </div>
                            <div style={{ fontSize: 10, color: 'var(--faint)', flexShrink: 0 }}>
                              #{a.id}
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {!desktop && panelTab === 'contents' && (
                    <div style={{ padding: '12px' }}>
                      <div style={{ padding: '0 0 8px', fontSize: 10, fontWeight: 600, letterSpacing: '0.06em', textTransform: 'uppercase', color: 'var(--faint)' }}>
                        Output Contents
                      </div>
                      {AGENTS.map(a => {
                        const isExpanded = expandedContentId === a.id
                        return (
                          <div
                            key={a.id}
                            style={{
                              borderBottom: '1px solid var(--line-soft)',
                              background: isExpanded ? 'var(--hover)' : 'transparent',
                            }}
                          >
                            <div
                              onClick={() => setExpandedContentId(isExpanded ? null : a.id)}
                              style={{ padding: '10px 8px', cursor: 'pointer' }}
                              onMouseEnter={e => { if (!isExpanded) e.currentTarget.style.background = 'var(--hover)' }}
                              onMouseLeave={e => { if (!isExpanded) e.currentTarget.style.background = 'transparent' }}
                            >
                              <div style={{ fontSize: 12, fontWeight: 500, color: 'var(--text)' }}>{a.output}</div>
                            </div>
                            {isExpanded && (
                              <div style={{ padding: '0 8px 12px 22px' }}>
                                <div style={{ display: 'flex', gap: 10, alignItems: 'center', marginBottom: 8 }}>
                                  <img src={a.gif} alt={a.agent} style={{ width: 64, height: 36, borderRadius: 4, objectFit: 'cover', border: '1px solid var(--line)' }} />
                                  <div>
                                    <div style={{ fontSize: 12, fontWeight: 700 }}>{a.agent}</div>
                                    <div style={{ fontSize: 10, color: 'var(--faint)', fontFamily: 'monospace' }}>
                                      {a.model}{AGENT_COUNTS[a.agent] !== '1' ? ` x${AGENT_COUNTS[a.agent]}` : ''}
                                    </div>
                                  </div>
                                </div>
                                <div style={{ fontSize: 11, color: 'var(--muted)', lineHeight: 1.5, marginBottom: 8 }}>
                                  {a.desc}
                                </div>
                                <button
                                  onClick={(e) => { e.stopPropagation(); taskAgent(a) }}
                                  style={{
                                    fontSize: 11, fontWeight: 600, padding: '4px 10px', borderRadius: 3,
                                    border: '1px solid var(--line-strong)', background: 'var(--panel)',
                                    color: 'var(--text)', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: 4,
                                  }}
                                >
                                  <Send size={10} /> Task this agent
                                </button>
                              </div>
                            )}
                          </div>
                        )
                      })}
                    </div>
                  )}

                  {!desktop && panelTab === 'environment' && (
                    <div style={{ padding: '12px' }}>
                      <div style={{ padding: '0 0 8px', fontSize: 10, fontWeight: 600, letterSpacing: '0.06em', textTransform: 'uppercase', color: 'var(--faint)' }}>
                        Models
                      </div>
                      <div style={{ border: '1px solid var(--line)', borderRadius: 3, overflow: 'hidden', marginBottom: 16 }}>
                        {[
                          { model: 'Qwen3-4B', used: 'Idea, Crawler, Decision, Section Leads, Field Agents, Surroundings List' },
                          { model: 'Qwen3.5-0.8B', used: 'Vision Sensor, Kind Checkers, Mini Objects, Task Assigner, Vectors, Sound' },
                          { model: 'none (code)', used: 'Video Controller' },
                          { model: 'none (laws)', used: 'Verifiers' },
                        ].map((m, i) => (
                          <div key={m.model} style={{
                            padding: '6px 10px', borderTop: i > 0 ? '1px solid var(--line-soft)' : 'none',
                            background: i % 2 === 0 ? 'var(--panel)' : 'transparent',
                          }}>
                            <div style={{ fontSize: 12, fontWeight: 600, fontFamily: 'monospace' }}>{m.model}</div>
                            <div style={{ fontSize: 10, color: 'var(--muted)', marginTop: 2 }}>{m.used}</div>
                          </div>
                        ))}
                      </div>

                      <div style={{ padding: '0 0 8px', fontSize: 10, fontWeight: 600, letterSpacing: '0.06em', textTransform: 'uppercase', color: 'var(--faint)' }}>
                        Agent Instances (last run)
                      </div>
                      <div style={{ border: '1px solid var(--line)', borderRadius: 3, overflow: 'hidden' }}>
                        {AGENTS.map((a, i) => (
                          <div key={a.id} style={{
                            display: 'flex', alignItems: 'center', gap: 8, padding: '5px 10px',
                            borderTop: i > 0 ? '1px solid var(--line-soft)' : 'none',
                            background: i % 2 === 0 ? 'var(--panel)' : 'transparent',
                          }}>
                            <img src={a.gif} alt="" style={{ width: 18, height: 18, borderRadius: 2, objectFit: 'cover', background: '#111' }} />
                            <span style={{ fontSize: 12, flex: 1 }}>{a.agent}</span>
                            <span style={{ fontSize: 11, color: 'var(--muted)', fontFamily: 'monospace', minWidth: 36, textAlign: 'right' }}>
                              {AGENT_COUNTS[a.agent] || '1'}
                            </span>
                            <span style={{ fontSize: 10, color: 'var(--faint)', width: 50, textAlign: 'right' }}>idle</span>
                          </div>
                        ))}
                      </div>

                      <div style={{ padding: '16px 0 8px', fontSize: 10, fontWeight: 600, letterSpacing: '0.06em', textTransform: 'uppercase', color: 'var(--faint)' }}>
                        Total Agent Count
                      </div>
                      <div style={{
                        padding: '12px', border: '1px solid var(--line)', borderRadius: 3,
                        background: 'var(--panel)', textAlign: 'center', marginBottom: 16,
                      }}>
                        <span style={{ fontSize: 28, fontWeight: 700, fontFamily: 'monospace' }}>~175</span>
                        <div style={{ fontSize: 10, color: 'var(--muted)', marginTop: 4 }}>agents spawned per generation</div>
                      </div>

                      {stdb.ready && (
                        <>
                          <div style={{ padding: '0 0 8px', fontSize: 10, fontWeight: 600, letterSpacing: '0.06em', textTransform: 'uppercase', color: 'var(--faint)' }}>Collaborators</div>
                          <div style={{ border: '1px solid var(--line)', borderRadius: 3, overflow: 'hidden', marginBottom: 16 }}>
                            {stdb.collaborators.length === 0 && <div style={{ padding: '8px 10px', fontSize: 11, color: 'var(--faint)' }}>No collaborators yet</div>}
                            {stdb.collaborators.map((c, i) => (
                              <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '6px 10px', borderTop: i > 0 ? '1px solid var(--line-soft)' : 'none' }}>
                                <div style={{ width: 8, height: 8, borderRadius: '50%', background: c.online ? '#4ade80' : 'var(--faint)', flexShrink: 0 }} />
                                <span style={{ fontSize: 12, flex: 1 }}>{c.nickname}</span>
                                <span style={{ fontSize: 10, color: 'var(--faint)' }}>{c.online ? 'online' : 'offline'}</span>
                              </div>
                            ))}
                          </div>

                          <div style={{ padding: '0 0 8px', fontSize: 10, fontWeight: 600, letterSpacing: '0.06em', textTransform: 'uppercase', color: 'var(--faint)' }}>Agent Run Log</div>
                          <div style={{ border: '1px solid var(--line)', borderRadius: 3, overflow: 'hidden', marginBottom: 16 }}>
                            {stdb.agentRuns.length === 0 && <div style={{ padding: '8px 10px', fontSize: 11, color: 'var(--faint)' }}>No runs recorded yet</div>}
                            {[...stdb.agentRuns].sort((a, b) => Number(b.id - a.id)).slice(0, 20).map((run, i) => (
                              <div key={i} style={{ padding: '6px 10px', borderTop: i > 0 ? '1px solid var(--line-soft)' : 'none' }}>
                                <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                                  <span style={{ fontSize: 11, fontWeight: 600 }}>{run.agentName}</span>
                                  <span style={{ fontSize: 10, color: 'var(--faint)', marginLeft: 'auto' }}>#{run.executionOrder + 1}</span>
                                </div>
                                <div style={{ fontSize: 10, color: 'var(--muted)', marginTop: 2, display: 'flex', gap: 8 }}>
                                  <span>{run.status}</span>
                                  {run.runtimeMs > 0n && <span>{Number(run.runtimeMs)}ms</span>}
                                  <span>{new Date(Number(run.startedAt)).toLocaleTimeString()}</span>
                                </div>
                              </div>
                            ))}
                          </div>

                          <div style={{ padding: '0 0 8px', fontSize: 10, fontWeight: 600, letterSpacing: '0.06em', textTransform: 'uppercase', color: 'var(--faint)' }}>Change History</div>
                          <div style={{ border: '1px solid var(--line)', borderRadius: 3, overflow: 'hidden', marginBottom: 16 }}>
                            {stdb.agentChanges.length === 0 && <div style={{ padding: '8px 10px', fontSize: 11, color: 'var(--faint)' }}>No changes recorded yet</div>}
                            {[...stdb.agentChanges].sort((a, b) => Number(b.id - a.id)).slice(0, 20).map((change, i) => (
                              <div key={i} style={{ padding: '6px 10px', borderTop: i > 0 ? '1px solid var(--line-soft)' : 'none' }}>
                                <div style={{ fontSize: 11 }}>
                                  <span style={{ fontWeight: 600 }}>{change.agentName}</span>
                                  <span style={{ color: 'var(--muted)' }}> . {change.field}</span>
                                </div>
                                <div style={{ fontSize: 10, color: 'var(--faint)', marginTop: 2 }}>{new Date(Number(change.changedAt)).toLocaleTimeString()}</div>
                              </div>
                            ))}
                          </div>

                          <div style={{ padding: '0 0 8px', fontSize: 10, fontWeight: 600, letterSpacing: '0.06em', textTransform: 'uppercase', color: 'var(--faint)' }}>Users ({stdb.studioUsers.length})</div>
                          <div style={{ border: '1px solid var(--line)', borderRadius: 3, overflow: 'hidden', marginBottom: 16 }}>
                            {stdb.studioUsers.length === 0 && <div style={{ padding: '8px 10px', fontSize: 11, color: 'var(--faint)' }}>No users yet</div>}
                            {[...stdb.studioUsers].sort((a, b) => Number(b.lastLoginAt - a.lastLoginAt)).map((u, i) => (
                              <div key={i} style={{ padding: '6px 10px', borderTop: i > 0 ? '1px solid var(--line-soft)' : 'none' }}>
                                <div style={{ fontSize: 11, fontWeight: 600 }}>{u.email}</div>
                                <div style={{ fontSize: 10, color: 'var(--faint)', marginTop: 2 }}>Last login {new Date(Number(u.lastLoginAt)).toLocaleString()}</div>
                              </div>
                            ))}
                          </div>
                        </>
                      )}
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        </main>

        <style>{`
          @keyframes pulse {
            0%, 80%, 100% { opacity: 0.3; transform: scale(0.8); }
            40% { opacity: 1; transform: scale(1); }
          }
          @keyframes agentSlideIn {
            from { opacity: 0; transform: translateY(12px); }
            to   { opacity: 1; transform: translateY(0); }
          }
          .chat-actions { transition: opacity 0.15s; }
          div:hover > .chat-actions { opacity: 1 !important; }
          ::placeholder { color: var(--faint) !important; }
          textarea::placeholder { color: var(--faint) !important; }
        `}</style>
      </div>
    </div>
  )
}

