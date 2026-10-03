import { useState, useRef, useEffect } from 'react'
import { Plus, Send, Loader2, Sun, Moon, Trash2, Pencil, Check, X, PanelRightClose, PanelRight, Play } from 'lucide-react'
import { useTheme } from '@/contexts/ThemeContext'
import { useSpacetime } from './useSpacetime'
import { pickColor } from './spacetime'

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

const WELCOME: Message = {
  id: 'welcome',
  role: 'assistant',
  content:
    "Hey! Tell me what video you want to create and I'll handle scripting, visuals, voiceover, and editing.\n\nTry something like:\n- \"A 30-second product launch teaser with cinematic transitions\"\n- \"YouTube explainer about how solar panels work\"\n- \"Instagram reel for a coffee shop grand opening\"",
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

type PanelTab = 'video' | 'contents' | 'environment'

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

export default function App() {
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
  const [panelOpen, setPanelOpen] = useState(false)
  const [panelTab, setPanelTab] = useState<PanelTab>('video')
  const [selectedAgent, setSelectedAgent] = useState<Artifact | null>(null)
  const [playing, setPlaying] = useState(false)
  const [revealedAgents, setRevealedAgents] = useState<AgentNote[]>([])
  const [activeAgentIdx, setActiveAgentIdx] = useState(-1)
  const [expandedContentId, setExpandedContentId] = useState<string | null>(null)
  const [targetAgent, setTargetAgent] = useState<Artifact | null>(null)
  const revealTimerRef = useRef<number | null>(null)
  const videoRef = useRef<HTMLVideoElement>(null)
  const bottomRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLTextAreaElement>(null)
  const stdb = useSpacetime()
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
    inputRef.current?.focus()
  }, [activeChatId])

  const taskAgent = (agent: Artifact) => {
    setTargetAgent(agent)
    setInput('')
    inputRef.current?.focus()
  }

  const handleSend = () => {
    if (!input.trim() || generating) return
    const displayContent = targetAgent ? `@${targetAgent.agent} ${input.trim()}` : input.trim()
    const userMsg: Message = { id: makeId(), role: 'user', content: displayContent }

    setChats(prev =>
      prev.map(c =>
        c.id === activeChatId
          ? {
              ...c,
              messages: [...c.messages, userMsg],
              title: c.messages.length <= 1 ? input.trim().slice(0, 40) : c.title,
            }
          : c
      )
    )
    const chatId = activeChatId
    const currentTarget = targetAgent
    setInput('')
    setTargetAgent(null)

    if (currentTarget) {
      setGenerating(true)
      window.setTimeout(() => {
        const reply: Message = {
          id: makeId(),
          role: 'assistant',
          content: `[${currentTarget.agent}] Working on it. ${currentTarget.output}`,
          agents: [{ agentId: currentTarget.id, note: currentTarget.desc }],
        }
        setChats(prev =>
          prev.map(c =>
            c.id === chatId ? { ...c, messages: [...c.messages, reply] } : c
          )
        )
        setGenerating(false)
        if (!panelOpen) setPanelOpen(true)
      }, 1200)
      return
    }

    setGenerating(true)
    setRevealedAgents([])
    setActiveAgentIdx(-1)

    if (revealTimerRef.current) {
      clearTimeout(revealTimerRef.current)
      revealTimerRef.current = null
    }

    const { content, agents } = getAgentReply(input.trim())

    const introMsg: Message = { id: makeId(), role: 'assistant', content }
    setChats(prev =>
      prev.map(c =>
        c.id === chatId ? { ...c, messages: [...c.messages, introMsg] } : c
      )
    )
    if (!panelOpen) setPanelOpen(true)

    stdb.reducers?.createProject({ name: displayContent.slice(0, 40), prompt: displayContent })

    const agentsCopy = [...agents]
    let idx = 0
    const reveal = () => {
      if (idx < agentsCopy.length) {
        const current = agentsCopy[idx]
        setRevealedAgents(agentsCopy.slice(0, idx + 1))
        setActiveAgentIdx(idx)

        const agent = AGENTS.find(a => a.id === current.agentId)
        if (agent) {
          stdb.reducers?.startAgentRun({ projectId: 0n, agentId: agent.id, agentName: agent.agent, executionOrder: idx })
        }

        idx++
        revealTimerRef.current = window.setTimeout(reveal, 2200)
      } else {
        setGenerating(false)
        setActiveAgentIdx(-1)
        const doneMsg: Message = {
          id: makeId(),
          role: 'assistant',
          content: 'All 14 agent types active. ~175 instances running. Check the Browser panel for details.',
        }
        setChats(prev =>
          prev.map(c =>
            c.id === chatId ? { ...c, messages: [...c.messages, doneMsg] } : c
          )
        )
      }
    }
    revealTimerRef.current = window.setTimeout(reveal, 800)
  }

  const newChat = () => {
    const id = makeId()
    setChats(prev => [
      { id, title: 'New video', messages: [WELCOME], createdAt: Date.now() },
      ...prev,
    ])
    setActiveChatId(id)
  }

  const deleteChat = (id: string) => {
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

  const confirmRename = () => {
    if (!editingChatId) return
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
                      <span style={{ flex: 1, fontSize: 12, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{chat.title}</span>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 2, opacity: 0 }} className="chat-actions">
                        <button onClick={e => { e.stopPropagation(); startRename(chat.id, chat.title) }} style={{ padding: 3, color: 'var(--faint)', background: 'none', border: 'none', cursor: 'pointer', minHeight: 'auto', borderRadius: 2 }}>
                          <Pencil size={11} />
                        </button>
                        <button onClick={e => { e.stopPropagation(); deleteChat(chat.id) }} style={{ padding: 3, color: 'var(--faint)', background: 'none', border: 'none', cursor: 'pointer', minHeight: 'auto', borderRadius: 2 }}>
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
                  <button
                    onClick={() => stdb.reducers?.signOut({})}
                    className="no-drag"
                    style={{ fontSize: 10, color: 'var(--faint)', background: 'none', border: 'none', cursor: 'pointer', padding: 0, minHeight: 'auto', textDecoration: 'underline' }}
                  >
                    Sign out
                  </button>
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
                {activeChat.title}
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
                Browser
              </button>
            </div>
          </div>

          {/* Content row: messages + optional panel */}
          <div style={{ flex: 1, display: 'flex', overflow: 'hidden' }}>

            {/* Messages column */}
            <div style={{ flex: 1, display: 'flex', flexDirection: 'column', minWidth: 0 }}>
              <div style={{ flex: 1, overflowY: 'auto', background: 'var(--canvas)' }}>
                <div style={{ maxWidth: 680, margin: '0 auto', padding: '24px 20px' }}>
                  {activeChat.messages.map(msg => (
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

                  {revealedAgents.length > 0 && (
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
                      <span style={{ fontSize: 11, fontWeight: 600, color: 'var(--text)' }}>@{targetAgent.agent}</span>
                      <button
                        onClick={() => setTargetAgent(null)}
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
                    Agents generate scripts, visuals, voiceover &amp; edits from your prompt.
                  </p>
                </div>
              </div>
            </div>

            {/* Right panel (browser) */}
            {panelOpen && (
              <div style={{
                width: 380, flexShrink: 0, display: 'flex', flexDirection: 'column',
                borderLeft: '1px solid var(--line)', background: 'var(--chrome)',
              }}>
                {/* Panel tabs */}
                <div style={{
                  display: 'flex', borderBottom: '1px solid var(--line)', flexShrink: 0,
                }}>
                  {(['video', 'contents', 'environment'] as PanelTab[]).map(tab => (
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
                      {tab.charAt(0).toUpperCase() + tab.slice(1)}
                    </button>
                  ))}
                </div>

                {/* Panel content */}
                <div style={{ flex: 1, overflowY: 'auto', padding: 0 }}>

                  {panelTab === 'video' && (
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

                  {panelTab === 'contents' && (
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

                  {panelTab === 'environment' && (
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

function getAgentReply(_input: string): { content: string; agents: AgentNote[] } {
  return {
    content: 'Spawning ~175 agents for your video. Each one is working on its piece:',
    agents: [
      { agentId: '50', note: "I'm the controller. I own the 60-second budget and I'll time every stage so nothing runs over." },
      { agentId: '49', note: "Writing your script right now. Once I'm done I'll hand off to Crawler and Decision." },
      { agentId: '51', note: "Building image search queries from the script. I figure out what visuals we need." },
      { agentId: '52', note: "I pick the best photos from crawl results and answer the fur, ground, bone-name forms." },
      { agentId: '53', note: "One yes/no question per photo. I'm fast: just tell me what to look for." },
      { agentId: '54', note: "Four of us, one per section: look, material, gait, ground texture. We lead the detail work." },
      { agentId: '55', note: "~35 of us, one tiny agent per number or word inside a section. We fill in every attribute." },
      { agentId: '56', note: "I list every kind of surrounding object and their proportions in the scene." },
      { agentId: '57', note: "Five of us checking each kind: 'is this actually a physical object?' Yes or no." },
      { agentId: '58', note: "72 of us, one per surrounding object. We figure out exactly where it sits and how big it is." },
      { agentId: '59', note: "I decide which Vector Agents to spawn based on what body parts need to move." },
      { agentId: '60', note: "V1 through V7. Each of us drives one body part movement in the animation." },
      { agentId: '61', note: "I pick which contacts make sound. Footsteps, impacts, surfaces." },
      { agentId: '62', note: "44 of us. Every single decision gets checked against physics or maths law." },
    ],
  }
}
