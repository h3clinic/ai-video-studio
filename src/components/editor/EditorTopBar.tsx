import { useNavigate } from 'react-router-dom'
import {
  ArrowLeft, Undo2, Redo2, ZoomIn, ZoomOut, Maximize2,
  Play, Download, Share2, Sparkles, Moon, Sun,
  MonitorSmartphone, Smartphone, Monitor,
} from 'lucide-react'
import { useState } from 'react'
import { useTheme } from '@/contexts/ThemeContext'

interface Props {
  onToggleAgents: () => void
  agentsOpen: boolean
}

export default function EditorTopBar({ onToggleAgents, agentsOpen }: Props) {
  const navigate = useNavigate()
  const { theme, toggleTheme } = useTheme()
  const [zoom, setZoom] = useState(100)

  return (
    <header className="drag-region h-12 flex items-center justify-between px-3 border-b border-zinc-200 dark:border-border-dark bg-white dark:bg-surface-dark-alt shrink-0">
      {/* Left section */}
      <div className="no-drag flex items-center gap-2">
        <button
          onClick={() => navigate('/')}
          className="p-2 rounded-lg text-zinc-400 hover:text-zinc-600 dark:hover:text-zinc-200 hover:bg-zinc-100 dark:hover:bg-surface-dark-hover transition-colors"
        >
          <ArrowLeft className="w-4 h-4" />
        </button>

        <div className="h-5 w-px bg-zinc-200 dark:bg-border-dark" />

        <div className="flex items-center gap-0.5">
          <button className="p-1.5 rounded-md text-zinc-400 hover:text-zinc-600 dark:hover:text-zinc-200 hover:bg-zinc-100 dark:hover:bg-surface-dark-hover transition-colors">
            <Undo2 className="w-4 h-4" />
          </button>
          <button className="p-1.5 rounded-md text-zinc-400 hover:text-zinc-600 dark:hover:text-zinc-200 hover:bg-zinc-100 dark:hover:bg-surface-dark-hover transition-colors">
            <Redo2 className="w-4 h-4" />
          </button>
        </div>

        <div className="h-5 w-px bg-zinc-200 dark:bg-border-dark" />

        <input
          type="text"
          defaultValue="Untitled Project"
          className="bg-transparent text-sm font-medium outline-none border-b border-transparent hover:border-zinc-300 dark:hover:border-zinc-600 focus:border-brand-500 px-1 py-0.5 w-44"
        />
      </div>

      {/* Center section - zoom & preview controls */}
      <div className="no-drag flex items-center gap-2">
        <div className="flex items-center gap-1 bg-zinc-100 dark:bg-surface-dark-elevated rounded-lg px-1 py-0.5">
          <button
            onClick={() => setZoom(z => Math.max(25, z - 25))}
            className="p-1 rounded text-zinc-400 hover:text-zinc-600 dark:hover:text-zinc-200 transition-colors"
          >
            <ZoomOut className="w-3.5 h-3.5" />
          </button>
          <span className="text-xs text-zinc-500 w-10 text-center font-mono">{zoom}%</span>
          <button
            onClick={() => setZoom(z => Math.min(200, z + 25))}
            className="p-1 rounded text-zinc-400 hover:text-zinc-600 dark:hover:text-zinc-200 transition-colors"
          >
            <ZoomIn className="w-3.5 h-3.5" />
          </button>
        </div>

        <div className="h-5 w-px bg-zinc-200 dark:bg-border-dark" />

        <div className="flex items-center gap-0.5 bg-zinc-100 dark:bg-surface-dark-elevated rounded-lg px-1 py-0.5">
          <button className="p-1 rounded text-brand-500 bg-brand-500/10">
            <Monitor className="w-3.5 h-3.5" />
          </button>
          <button className="p-1 rounded text-zinc-400 hover:text-zinc-600 dark:hover:text-zinc-200 transition-colors">
            <Smartphone className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>

      {/* Right section */}
      <div className="no-drag flex items-center gap-2">
        <button
          onClick={onToggleAgents}
          className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm font-medium transition-colors ${
            agentsOpen
              ? 'bg-brand-600 text-white'
              : 'text-zinc-500 dark:text-zinc-400 hover:bg-zinc-100 dark:hover:bg-surface-dark-hover'
          }`}
        >
          <Sparkles className="w-4 h-4" />
          AI Agents
        </button>

        <button
          onClick={toggleTheme}
          className="p-2 rounded-lg text-zinc-400 hover:text-zinc-600 dark:hover:text-zinc-200 hover:bg-zinc-100 dark:hover:bg-surface-dark-hover transition-colors"
        >
          {theme === 'dark' ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />}
        </button>

        <div className="h-5 w-px bg-zinc-200 dark:bg-border-dark" />

        <button className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm text-zinc-500 dark:text-zinc-400 hover:bg-zinc-100 dark:hover:bg-surface-dark-hover transition-colors">
          <Play className="w-4 h-4" />
          Preview
        </button>

        <button className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm text-zinc-500 dark:text-zinc-400 hover:bg-zinc-100 dark:hover:bg-surface-dark-hover transition-colors">
          <Share2 className="w-4 h-4" />
          Share
        </button>

        <button className="flex items-center gap-1.5 bg-brand-600 hover:bg-brand-700 text-white rounded-lg px-4 py-1.5 text-sm font-medium transition-colors">
          <Download className="w-4 h-4" />
          Export
        </button>
      </div>
    </header>
  )
}
