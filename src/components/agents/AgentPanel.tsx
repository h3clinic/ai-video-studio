import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  X, Sparkles, FileText, ImageIcon, Mic, Music, Wand2,
  Send, Bot, User, ChevronRight, Loader2, RefreshCcw,
  Palette, Film, Scissors, Zap, BrainCircuit,
} from 'lucide-react'

interface Props {
  onClose: () => void
}

type AgentId = 'chat' | 'script' | 'storyboard' | 'voice' | 'music' | 'style' | 'edit'

interface Agent {
  id: AgentId
  label: string
  description: string
  icon: React.ComponentType<{ className?: string }>
  color: string
}

const AGENTS: Agent[] = [
  { id: 'chat', label: 'AI Assistant', description: 'General help and ideas', icon: BrainCircuit, color: 'text-brand-500' },
  { id: 'script', label: 'Script Writer', description: 'Generate and refine scripts', icon: FileText, color: 'text-emerald-500' },
  { id: 'storyboard', label: 'Storyboard', description: 'Visual scene planning', icon: ImageIcon, color: 'text-amber-500' },
  { id: 'voice', label: 'Voice Agent', description: 'AI voiceover generation', icon: Mic, color: 'text-pink-500' },
  { id: 'music', label: 'Music Agent', description: 'Background music & SFX', icon: Music, color: 'text-orange-500' },
  { id: 'style', label: 'Style Agent', description: 'Visual style & branding', icon: Palette, color: 'text-violet-500' },
  { id: 'edit', label: 'Auto Editor', description: 'Smart cuts & transitions', icon: Scissors, color: 'text-sky-500' },
]

interface ChatMessage {
  id: string
  role: 'user' | 'agent'
  content: string
  agent?: string
}

const DEMO_MESSAGES: ChatMessage[] = [
  {
    id: '1',
    role: 'agent',
    content: "Hi! I'm your AI video assistant. I can help you with scripts, storyboards, voiceovers, music selection, and more. What would you like to work on?",
    agent: 'AI Assistant',
  },
]

export default function AgentPanel({ onClose }: Props) {
  const [activeAgent, setActiveAgent] = useState<AgentId>('chat')
  const [messages, setMessages] = useState<ChatMessage[]>(DEMO_MESSAGES)
  const [input, setInput] = useState('')
  const [isProcessing, setIsProcessing] = useState(false)

  const handleSend = () => {
    if (!input.trim() || isProcessing) return
    const userMsg: ChatMessage = {
      id: Date.now().toString(),
      role: 'user',
      content: input,
    }
    setMessages(m => [...m, userMsg])
    setInput('')
    setIsProcessing(true)

    setTimeout(() => {
      const agentMsg: ChatMessage = {
        id: (Date.now() + 1).toString(),
        role: 'agent',
        content: getAgentResponse(activeAgent, input),
        agent: AGENTS.find(a => a.id === activeAgent)?.label,
      }
      setMessages(m => [...m, agentMsg])
      setIsProcessing(false)
    }, 1500)
  }

  return (
    <aside className="w-80 h-full border-l border-zinc-200 dark:border-border-dark bg-zinc-50 dark:bg-surface-dark-alt shrink-0 flex flex-col">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-zinc-200 dark:border-border-dark">
        <div className="flex items-center gap-2">
          <Sparkles className="w-4 h-4 text-brand-500" />
          <span className="text-xs font-semibold uppercase tracking-wider text-zinc-500 dark:text-zinc-400">AI Agents</span>
        </div>
        <button onClick={onClose} className="p-1 rounded-md text-zinc-400 hover:text-zinc-600 dark:hover:text-zinc-200 hover:bg-zinc-100 dark:hover:bg-surface-dark-hover transition-colors">
          <X className="w-4 h-4" />
        </button>
      </div>

      {/* Agent selector */}
      <div className="px-3 py-2 border-b border-zinc-200 dark:border-border-dark overflow-x-auto">
        <div className="flex gap-1.5">
          {AGENTS.map(agent => (
            <button
              key={agent.id}
              onClick={() => setActiveAgent(agent.id)}
              className={`flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-[11px] font-medium whitespace-nowrap transition-colors ${
                activeAgent === agent.id
                  ? 'bg-brand-500/10 text-brand-500 border border-brand-500/20'
                  : 'text-zinc-400 hover:bg-zinc-100 dark:hover:bg-surface-dark-hover border border-transparent'
              }`}
            >
              <agent.icon className={`w-3.5 h-3.5 ${activeAgent === agent.id ? agent.color : ''}`} />
              {agent.label}
            </button>
          ))}
        </div>
      </div>

      {/* Agent content area */}
      {activeAgent === 'chat' ? (
        <ChatView
          messages={messages}
          input={input}
          onInputChange={setInput}
          onSend={handleSend}
          isProcessing={isProcessing}
        />
      ) : (
        <AgentWorkspace agent={AGENTS.find(a => a.id === activeAgent)!} />
      )}
    </aside>
  )
}

function ChatView({
  messages,
  input,
  onInputChange,
  onSend,
  isProcessing,
}: {
  messages: ChatMessage[]
  input: string
  onInputChange: (v: string) => void
  onSend: () => void
  isProcessing: boolean
}) {
  return (
    <>
      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-3 space-y-3">
        {messages.map(msg => (
          <motion.div
            key={msg.id}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className={`flex gap-2 ${msg.role === 'user' ? 'flex-row-reverse' : ''}`}
          >
            <div className={`w-6 h-6 rounded-full flex items-center justify-center shrink-0 ${
              msg.role === 'user'
                ? 'bg-brand-600'
                : 'bg-gradient-to-br from-brand-500 to-purple-600'
            }`}>
              {msg.role === 'user'
                ? <User className="w-3 h-3 text-white" />
                : <Bot className="w-3 h-3 text-white" />
              }
            </div>
            <div className={`max-w-[85%] ${msg.role === 'user' ? 'text-right' : ''}`}>
              {msg.agent && (
                <span className="text-[10px] text-zinc-400 mb-0.5 block">{msg.agent}</span>
              )}
              <div className={`rounded-lg px-3 py-2 text-xs leading-relaxed ${
                msg.role === 'user'
                  ? 'bg-brand-600 text-white'
                  : 'bg-white dark:bg-surface-dark-elevated border border-zinc-200 dark:border-border-dark'
              }`}>
                {msg.content}
              </div>
            </div>
          </motion.div>
        ))}

        {isProcessing && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="flex gap-2"
          >
            <div className="w-6 h-6 rounded-full bg-gradient-to-br from-brand-500 to-purple-600 flex items-center justify-center">
              <Bot className="w-3 h-3 text-white" />
            </div>
            <div className="bg-white dark:bg-surface-dark-elevated border border-zinc-200 dark:border-border-dark rounded-lg px-3 py-2">
              <div className="flex items-center gap-1.5">
                <Loader2 className="w-3 h-3 text-brand-500 animate-spin" />
                <span className="text-xs text-zinc-400">Thinking...</span>
              </div>
            </div>
          </motion.div>
        )}
      </div>

      {/* Quick actions */}
      <div className="px-3 pb-1">
        <div className="flex gap-1.5 flex-wrap">
          {['Write a script', 'Generate storyboard', 'Add voiceover', 'Suggest music'].map(action => (
            <button
              key={action}
              onClick={() => onInputChange(action)}
              className="text-[10px] px-2 py-1 rounded-md border border-zinc-200 dark:border-border-dark text-zinc-400 hover:border-brand-500/30 hover:text-brand-500 transition-colors"
            >
              {action}
            </button>
          ))}
        </div>
      </div>

      {/* Input */}
      <div className="p-3 border-t border-zinc-200 dark:border-border-dark">
        <div className="flex items-center gap-2 bg-white dark:bg-surface-dark-elevated border border-zinc-200 dark:border-border-dark rounded-lg px-3 py-2 focus-within:border-brand-500 transition-colors">
          <input
            type="text"
            value={input}
            onChange={e => onInputChange(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && onSend()}
            placeholder="Ask the AI agent..."
            className="flex-1 bg-transparent outline-none text-xs placeholder:text-zinc-400"
          />
          <button
            onClick={onSend}
            disabled={!input.trim() || isProcessing}
            className="p-1 rounded-md text-brand-500 hover:bg-brand-500/10 disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
          >
            <Send className="w-4 h-4" />
          </button>
        </div>
      </div>
    </>
  )
}

function AgentWorkspace({ agent }: { agent: Agent }) {
  return (
    <div className="flex-1 overflow-y-auto p-4 space-y-4">
      {/* Agent header */}
      <div className="flex items-center gap-3 p-3 rounded-lg bg-white dark:bg-surface-dark-elevated border border-zinc-200 dark:border-border-dark">
        <div className={`w-10 h-10 rounded-xl bg-zinc-100 dark:bg-surface-dark flex items-center justify-center ${agent.color}`}>
          <agent.icon className="w-5 h-5" />
        </div>
        <div>
          <h3 className="text-sm font-semibold">{agent.label}</h3>
          <p className="text-[11px] text-zinc-400">{agent.description}</p>
        </div>
      </div>

      {/* Agent-specific controls */}
      {agent.id === 'script' && <ScriptAgentControls />}
      {agent.id === 'storyboard' && <StoryboardAgentControls />}
      {agent.id === 'voice' && <VoiceAgentControls />}
      {agent.id === 'music' && <MusicAgentControls />}
      {agent.id === 'style' && <StyleAgentControls />}
      {agent.id === 'edit' && <EditAgentControls />}
    </div>
  )
}

function ScriptAgentControls() {
  return (
    <div className="space-y-3">
      <div>
        <label className="text-xs text-zinc-400 mb-1 block">Topic / Brief</label>
        <textarea
          placeholder="Describe what the video is about..."
          className="w-full h-20 bg-white dark:bg-surface-dark-elevated border border-zinc-200 dark:border-border-dark rounded-lg p-2.5 text-xs outline-none resize-none focus:border-brand-500 transition-colors"
        />
      </div>
      <div className="grid grid-cols-2 gap-2">
        <div>
          <label className="text-xs text-zinc-400 mb-1 block">Tone</label>
          <select className="w-full bg-white dark:bg-surface-dark-elevated border border-zinc-200 dark:border-border-dark rounded-lg px-2.5 py-1.5 text-xs outline-none">
            <option>Professional</option>
            <option>Casual</option>
            <option>Humorous</option>
            <option>Dramatic</option>
          </select>
        </div>
        <div>
          <label className="text-xs text-zinc-400 mb-1 block">Length</label>
          <select className="w-full bg-white dark:bg-surface-dark-elevated border border-zinc-200 dark:border-border-dark rounded-lg px-2.5 py-1.5 text-xs outline-none">
            <option>30 seconds</option>
            <option>1 minute</option>
            <option>3 minutes</option>
            <option>5+ minutes</option>
          </select>
        </div>
      </div>
      <button className="w-full flex items-center justify-center gap-2 bg-emerald-600 hover:bg-emerald-700 text-white rounded-lg py-2 text-xs font-medium transition-colors">
        <Wand2 className="w-3.5 h-3.5" />
        Generate Script
      </button>
    </div>
  )
}

function StoryboardAgentControls() {
  const scenes = [
    { id: 1, label: 'Scene 1', description: 'Opening aerial shot' },
    { id: 2, label: 'Scene 2', description: 'Product close-up' },
    { id: 3, label: 'Scene 3', description: 'Customer testimonial' },
    { id: 4, label: 'Scene 4', description: 'Call to action' },
  ]

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <span className="text-xs font-medium text-zinc-500 dark:text-zinc-400">Scenes</span>
        <button className="text-[10px] text-brand-500 hover:text-brand-400 flex items-center gap-0.5">
          <RefreshCcw className="w-3 h-3" /> Regenerate
        </button>
      </div>

      <div className="space-y-2">
        {scenes.map(scene => (
          <div
            key={scene.id}
            className="flex gap-3 p-2 rounded-lg border border-zinc-200 dark:border-border-dark hover:border-brand-500/30 transition-colors cursor-pointer"
          >
            <div className="w-16 h-12 rounded-md bg-zinc-200 dark:bg-surface-dark-elevated shrink-0 flex items-center justify-center">
              <Film className="w-4 h-4 text-zinc-400" />
            </div>
            <div className="flex-1">
              <p className="text-xs font-medium">{scene.label}</p>
              <p className="text-[10px] text-zinc-400">{scene.description}</p>
            </div>
          </div>
        ))}
      </div>

      <button className="w-full flex items-center justify-center gap-2 bg-amber-600 hover:bg-amber-700 text-white rounded-lg py-2 text-xs font-medium transition-colors">
        <ImageIcon className="w-3.5 h-3.5" />
        Generate Frames
      </button>
    </div>
  )
}

function VoiceAgentControls() {
  const voices = [
    { id: 1, name: 'Alex', accent: 'American', gender: 'Male' },
    { id: 2, name: 'Sophie', accent: 'British', gender: 'Female' },
    { id: 3, name: 'Raj', accent: 'Indian', gender: 'Male' },
    { id: 4, name: 'Yuki', accent: 'Japanese', gender: 'Female' },
  ]

  return (
    <div className="space-y-3">
      <div>
        <label className="text-xs text-zinc-400 mb-1 block">Select Voice</label>
        <div className="space-y-1.5">
          {voices.map(voice => (
            <button
              key={voice.id}
              className={`w-full flex items-center gap-3 p-2 rounded-lg border transition-colors ${
                voice.id === 1
                  ? 'border-pink-500/30 bg-pink-500/5'
                  : 'border-zinc-200 dark:border-border-dark hover:border-pink-500/20'
              }`}
            >
              <div className="w-8 h-8 rounded-full bg-zinc-200 dark:bg-surface-dark flex items-center justify-center">
                <Mic className="w-3.5 h-3.5 text-zinc-400" />
              </div>
              <div className="text-left flex-1">
                <p className="text-xs font-medium">{voice.name}</p>
                <p className="text-[10px] text-zinc-400">{voice.accent} - {voice.gender}</p>
              </div>
              <button className="p-1 rounded-md text-zinc-400 hover:text-zinc-200">
                <Zap className="w-3 h-3" />
              </button>
            </button>
          ))}
        </div>
      </div>

      <div>
        <label className="text-xs text-zinc-400 mb-1 block">Speed</label>
        <input
          type="range"
          min="50"
          max="200"
          defaultValue="100"
          className="w-full h-1 bg-zinc-200 dark:bg-zinc-700 rounded-full appearance-none [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-3 [&::-webkit-slider-thumb]:h-3 [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-pink-500 [&::-webkit-slider-thumb]:cursor-pointer"
        />
        <div className="flex justify-between text-[10px] text-zinc-400 mt-0.5">
          <span>0.5x</span>
          <span>1.0x</span>
          <span>2.0x</span>
        </div>
      </div>

      <button className="w-full flex items-center justify-center gap-2 bg-pink-600 hover:bg-pink-700 text-white rounded-lg py-2 text-xs font-medium transition-colors">
        <Mic className="w-3.5 h-3.5" />
        Generate Voiceover
      </button>
    </div>
  )
}

function MusicAgentControls() {
  return (
    <div className="space-y-3">
      <div>
        <label className="text-xs text-zinc-400 mb-1 block">Mood</label>
        <div className="flex gap-1.5 flex-wrap">
          {['Energetic', 'Calm', 'Dramatic', 'Happy', 'Mysterious', 'Corporate'].map(mood => (
            <button
              key={mood}
              className={`text-[10px] px-2.5 py-1 rounded-full border transition-colors ${
                mood === 'Energetic'
                  ? 'border-orange-500/30 bg-orange-500/10 text-orange-500'
                  : 'border-zinc-200 dark:border-border-dark text-zinc-400 hover:border-orange-500/20'
              }`}
            >
              {mood}
            </button>
          ))}
        </div>
      </div>

      <div>
        <label className="text-xs text-zinc-400 mb-1 block">Genre</label>
        <select className="w-full bg-white dark:bg-surface-dark-elevated border border-zinc-200 dark:border-border-dark rounded-lg px-2.5 py-1.5 text-xs outline-none">
          <option>Electronic</option>
          <option>Acoustic</option>
          <option>Orchestral</option>
          <option>Lo-fi</option>
          <option>Pop</option>
        </select>
      </div>

      <div>
        <label className="text-xs text-zinc-400 mb-1 block">Duration</label>
        <span className="text-xs text-zinc-600 dark:text-zinc-300">Match to video length (2:34)</span>
      </div>

      <button className="w-full flex items-center justify-center gap-2 bg-orange-600 hover:bg-orange-700 text-white rounded-lg py-2 text-xs font-medium transition-colors">
        <Music className="w-3.5 h-3.5" />
        Generate Music
      </button>
    </div>
  )
}

function StyleAgentControls() {
  const styles = [
    { id: 1, name: 'Minimal', preview: 'bg-zinc-800' },
    { id: 2, name: 'Neon', preview: 'bg-gradient-to-r from-purple-900 to-pink-900' },
    { id: 3, name: 'Warm', preview: 'bg-gradient-to-r from-amber-900 to-orange-900' },
    { id: 4, name: 'Corporate', preview: 'bg-gradient-to-r from-blue-900 to-indigo-900' },
  ]

  return (
    <div className="space-y-3">
      <div>
        <label className="text-xs text-zinc-400 mb-1 block">Visual Style</label>
        <div className="grid grid-cols-2 gap-2">
          {styles.map(style => (
            <button
              key={style.id}
              className={`flex flex-col items-center gap-1.5 p-3 rounded-lg border transition-colors ${
                style.id === 1
                  ? 'border-violet-500/30 bg-violet-500/5'
                  : 'border-zinc-200 dark:border-border-dark hover:border-violet-500/20'
              }`}
            >
              <div className={`w-full h-10 rounded-md ${style.preview}`} />
              <span className="text-[10px]">{style.name}</span>
            </button>
          ))}
        </div>
      </div>

      <button className="w-full flex items-center justify-center gap-2 bg-violet-600 hover:bg-violet-700 text-white rounded-lg py-2 text-xs font-medium transition-colors">
        <Palette className="w-3.5 h-3.5" />
        Apply Style
      </button>
    </div>
  )
}

function EditAgentControls() {
  return (
    <div className="space-y-3">
      <div>
        <label className="text-xs text-zinc-400 mb-1 block">Editing Style</label>
        <div className="space-y-1.5">
          {[
            { name: 'Fast Cuts', desc: 'Quick transitions, high energy' },
            { name: 'Smooth', desc: 'Gradual transitions, cinematic' },
            { name: 'Documentary', desc: 'Natural pacing, storytelling' },
            { name: 'Social Media', desc: 'Punchy, attention-grabbing' },
          ].map(style => (
            <button
              key={style.name}
              className={`w-full text-left p-2.5 rounded-lg border transition-colors ${
                style.name === 'Fast Cuts'
                  ? 'border-sky-500/30 bg-sky-500/5'
                  : 'border-zinc-200 dark:border-border-dark hover:border-sky-500/20'
              }`}
            >
              <p className="text-xs font-medium">{style.name}</p>
              <p className="text-[10px] text-zinc-400">{style.desc}</p>
            </button>
          ))}
        </div>
      </div>

      <button className="w-full flex items-center justify-center gap-2 bg-sky-600 hover:bg-sky-700 text-white rounded-lg py-2 text-xs font-medium transition-colors">
        <Scissors className="w-3.5 h-3.5" />
        Auto-Edit Timeline
      </button>
    </div>
  )
}

function getAgentResponse(agent: AgentId, input: string): string {
  const responses: Record<string, string> = {
    chat: "Great idea! I can help you with that. Would you like me to start with a script outline, or should we jump straight into visual planning? I'd recommend starting with a clear script structure, then we can break it into scenes for the storyboard agent.",
    script: "I've drafted a script outline based on your input. The video will have 4 main scenes: Hook (0-5s), Problem (5-20s), Solution (20-45s), and CTA (45-60s). Want me to flesh out the narration for each scene?",
    storyboard: "Based on the current script, I've planned 6 visual scenes. Each scene has suggested camera angles, transitions, and overlay text. Click on any scene card to preview or edit the visual direction.",
    voice: "I've analyzed your script and recommend the 'Alex' voice for this content type. The estimated voiceover duration is 2:15. Should I generate a preview with the first paragraph?",
    music: "For a professional product demo, I'd suggest an upbeat electronic track at 120 BPM. I've queued 3 options that match your video's energy arc. Press play on any track to preview.",
    style: "I've analyzed your brand colors and suggest a clean, minimal style with your primary blue as the accent color. Transitions will use smooth fades to maintain professionalism.",
    edit: "I'll analyze your clips and apply smart cuts. Based on the 'Fast Cuts' style, I'll add beat-synced transitions, remove dead air, and optimize pacing. This typically takes about 30 seconds.",
  }
  return responses[agent] || responses.chat
}
