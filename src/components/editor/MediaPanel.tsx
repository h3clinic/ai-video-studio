import { useState } from 'react'
import {
  FileText, Image, Type, Shapes, Music, Upload, Search,
  Play, Clock, Star, Filter, Grid3X3, List,
} from 'lucide-react'
import type { PanelTab } from '@/pages/Editor'

interface Props {
  activeTab: PanelTab
  onTabChange: (tab: PanelTab) => void
}

const TABS = [
  { id: 'script' as PanelTab, label: 'Script', icon: FileText },
  { id: 'media' as PanelTab, label: 'Media', icon: Image },
  { id: 'text' as PanelTab, label: 'Text', icon: Type },
  { id: 'elements' as PanelTab, label: 'Elements', icon: Shapes },
  { id: 'audio' as PanelTab, label: 'Audio', icon: Music },
  { id: 'uploads' as PanelTab, label: 'Uploads', icon: Upload },
]

const STOCK_MEDIA = [
  { id: 1, type: 'video', label: 'City timelapse', duration: '0:12', color: 'bg-sky-900/40' },
  { id: 2, type: 'video', label: 'Ocean waves', duration: '0:08', color: 'bg-blue-900/40' },
  { id: 3, type: 'image', label: 'Mountain sunset', color: 'bg-amber-900/40' },
  { id: 4, type: 'video', label: 'Code typing', duration: '0:15', color: 'bg-emerald-900/40' },
  { id: 5, type: 'image', label: 'Team meeting', color: 'bg-violet-900/40' },
  { id: 6, type: 'video', label: 'Product unbox', duration: '0:20', color: 'bg-pink-900/40' },
  { id: 7, type: 'image', label: 'Abstract gradient', color: 'bg-indigo-900/40' },
  { id: 8, type: 'video', label: 'Coffee pour', duration: '0:06', color: 'bg-orange-900/40' },
  { id: 9, type: 'image', label: 'Workspace flat lay', color: 'bg-teal-900/40' },
  { id: 10, type: 'video', label: 'Drone flyover', duration: '0:18', color: 'bg-cyan-900/40' },
  { id: 11, type: 'image', label: 'Night cityscape', color: 'bg-purple-900/40' },
  { id: 12, type: 'video', label: 'Hands typing', duration: '0:10', color: 'bg-rose-900/40' },
]

const TEXT_PRESETS = [
  { id: 1, label: 'Title', preview: 'Big Bold Title', style: 'text-2xl font-bold' },
  { id: 2, label: 'Subtitle', preview: 'Supporting subtitle', style: 'text-lg font-medium text-zinc-400' },
  { id: 3, label: 'Body', preview: 'Body text paragraph', style: 'text-sm' },
  { id: 4, label: 'Caption', preview: 'Small caption text', style: 'text-xs text-zinc-500' },
  { id: 5, label: 'Lower Third', preview: 'Speaker Name', style: 'text-sm font-semibold tracking-wide' },
  { id: 6, label: 'Call to Action', preview: 'Subscribe Now', style: 'text-lg font-bold text-brand-500' },
]

const ELEMENTS = [
  { id: 1, label: 'Rectangle', icon: '▭' },
  { id: 2, label: 'Circle', icon: '●' },
  { id: 3, label: 'Arrow', icon: '→' },
  { id: 4, label: 'Line', icon: '━' },
  { id: 5, label: 'Star', icon: '★' },
  { id: 6, label: 'Speech Bubble', icon: '💬' },
  { id: 7, label: 'Progress Bar', icon: '▰▰▰▱▱' },
  { id: 8, label: 'Logo Placeholder', icon: '◆' },
]

const AUDIO_TRACKS = [
  { id: 1, title: 'Upbeat Corporate', artist: 'Studio AI', duration: '2:30', mood: 'Energetic' },
  { id: 2, title: 'Calm Piano', artist: 'Studio AI', duration: '3:15', mood: 'Relaxed' },
  { id: 3, title: 'Tech Future', artist: 'Studio AI', duration: '2:45', mood: 'Modern' },
  { id: 4, title: 'Cinematic Rise', artist: 'Studio AI', duration: '1:20', mood: 'Dramatic' },
  { id: 5, title: 'Lo-fi Chill', artist: 'Studio AI', duration: '4:00', mood: 'Chill' },
  { id: 6, title: 'Acoustic Morning', artist: 'Studio AI', duration: '3:30', mood: 'Warm' },
]

export default function MediaPanel({ activeTab, onTabChange }: Props) {
  const [searchQuery, setSearchQuery] = useState('')
  const [viewMode, setViewMode] = useState<'grid' | 'list'>('grid')

  return (
    <aside className="w-72 h-full flex flex-col border-r border-zinc-200 dark:border-border-dark bg-zinc-50 dark:bg-surface-dark-alt shrink-0">
      {/* Tabs */}
      <div className="flex border-b border-zinc-200 dark:border-border-dark overflow-x-auto">
        {TABS.map(tab => (
          <button
            key={tab.id}
            onClick={() => onTabChange(tab.id)}
            className={`flex flex-col items-center gap-0.5 px-3 py-2.5 text-[10px] font-medium shrink-0 border-b-2 transition-colors ${
              activeTab === tab.id
                ? 'border-brand-500 text-brand-500'
                : 'border-transparent text-zinc-400 hover:text-zinc-600 dark:hover:text-zinc-200'
            }`}
          >
            <tab.icon className="w-4 h-4" />
            {tab.label}
          </button>
        ))}
      </div>

      {/* Search */}
      <div className="px-3 py-2">
        <div className="flex items-center gap-2 bg-white dark:bg-surface-dark-elevated rounded-lg px-2.5 py-1.5 border border-zinc-200 dark:border-border-dark">
          <Search className="w-3.5 h-3.5 text-zinc-400" />
          <input
            type="text"
            value={searchQuery}
            onChange={e => setSearchQuery(e.target.value)}
            placeholder={`Search ${activeTab}...`}
            className="flex-1 bg-transparent outline-none text-xs placeholder:text-zinc-400"
          />
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto px-3 pb-3">
        {activeTab === 'script' && <ScriptPanel />}
        {activeTab === 'media' && <MediaGrid viewMode={viewMode} />}
        {activeTab === 'text' && <TextPanel />}
        {activeTab === 'elements' && <ElementsPanel />}
        {activeTab === 'audio' && <AudioPanel />}
        {activeTab === 'uploads' && <UploadsPanel />}
      </div>
    </aside>
  )
}

function ScriptPanel() {
  const [script, setScript] = useState(
    'Scene 1: Open with a dramatic aerial shot of the city skyline at golden hour.\n\nNarrator: "In a world where technology moves faster than ever..."\n\nScene 2: Cut to close-up of hands working on a laptop.\n\nNarrator: "...one platform is changing the game."'
  )

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <span className="text-xs font-medium text-zinc-500 dark:text-zinc-400 uppercase tracking-wider">Script</span>
        <button className="flex items-center gap-1 text-xs text-brand-500 hover:text-brand-400">
          <Star className="w-3 h-3" />
          AI Rewrite
        </button>
      </div>
      <textarea
        value={script}
        onChange={e => setScript(e.target.value)}
        className="w-full h-64 bg-white dark:bg-surface-dark-elevated border border-zinc-200 dark:border-border-dark rounded-lg p-3 text-xs leading-relaxed outline-none resize-none focus:border-brand-500 transition-colors"
      />
      <div className="flex gap-2">
        <button className="flex-1 text-xs py-2 rounded-lg bg-brand-600 hover:bg-brand-700 text-white font-medium transition-colors">
          Generate Scenes
        </button>
        <button className="text-xs py-2 px-3 rounded-lg border border-zinc-200 dark:border-border-dark text-zinc-500 hover:bg-zinc-100 dark:hover:bg-surface-dark-hover transition-colors">
          Import
        </button>
      </div>
    </div>
  )
}

function MediaGrid({ viewMode }: { viewMode: string }) {
  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <span className="text-xs font-medium text-zinc-500 dark:text-zinc-400 uppercase tracking-wider">Stock Media</span>
        <div className="flex gap-1">
          <button className="p-1 rounded text-brand-500">
            <Grid3X3 className="w-3.5 h-3.5" />
          </button>
          <button className="p-1 rounded text-zinc-400 hover:text-zinc-600 dark:hover:text-zinc-200">
            <List className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>

      <div className="flex gap-1.5 flex-wrap">
        {['All', 'Videos', 'Images', 'AI Generated'].map(f => (
          <button
            key={f}
            className={`text-[10px] px-2 py-1 rounded-md transition-colors ${
              f === 'All'
                ? 'bg-brand-500/10 text-brand-500'
                : 'text-zinc-400 hover:bg-zinc-100 dark:hover:bg-surface-dark-hover'
            }`}
          >
            {f}
          </button>
        ))}
      </div>

      <div className="grid grid-cols-2 gap-2">
        {STOCK_MEDIA.map(item => (
          <div
            key={item.id}
            className={`group relative aspect-video rounded-lg ${item.color} cursor-grab hover:ring-2 hover:ring-brand-500/50 transition-all overflow-hidden`}
          >
            <div className="absolute inset-0 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity bg-black/30">
              <Play className="w-6 h-6 text-white" />
            </div>
            <span className="absolute bottom-1 left-1.5 text-[9px] text-white/80 font-medium">
              {item.label}
            </span>
            {item.duration && (
              <span className="absolute bottom-1 right-1.5 text-[9px] text-white/60 font-mono">
                {item.duration}
              </span>
            )}
          </div>
        ))}
      </div>
    </div>
  )
}

function TextPanel() {
  return (
    <div className="space-y-3">
      <span className="text-xs font-medium text-zinc-500 dark:text-zinc-400 uppercase tracking-wider">Text Presets</span>
      <div className="space-y-2">
        {TEXT_PRESETS.map(preset => (
          <button
            key={preset.id}
            className="w-full text-left p-3 rounded-lg border border-zinc-200 dark:border-border-dark bg-white dark:bg-surface-dark-elevated hover:border-brand-500/30 transition-colors"
          >
            <span className="text-[10px] uppercase tracking-wider text-zinc-400 mb-1 block">{preset.label}</span>
            <span className={preset.style}>{preset.preview}</span>
          </button>
        ))}
      </div>
    </div>
  )
}

function ElementsPanel() {
  return (
    <div className="space-y-3">
      <span className="text-xs font-medium text-zinc-500 dark:text-zinc-400 uppercase tracking-wider">Shapes & Elements</span>
      <div className="grid grid-cols-2 gap-2">
        {ELEMENTS.map(el => (
          <button
            key={el.id}
            className="flex flex-col items-center gap-2 p-4 rounded-lg border border-zinc-200 dark:border-border-dark bg-white dark:bg-surface-dark-elevated hover:border-brand-500/30 transition-colors"
          >
            <span className="text-xl">{el.icon}</span>
            <span className="text-[10px] text-zinc-400">{el.label}</span>
          </button>
        ))}
      </div>
    </div>
  )
}

function AudioPanel() {
  return (
    <div className="space-y-3">
      <span className="text-xs font-medium text-zinc-500 dark:text-zinc-400 uppercase tracking-wider">Music Library</span>
      <div className="space-y-1.5">
        {AUDIO_TRACKS.map(track => (
          <button
            key={track.id}
            className="w-full flex items-center gap-3 p-2.5 rounded-lg hover:bg-zinc-100 dark:hover:bg-surface-dark-hover transition-colors group"
          >
            <div className="w-8 h-8 rounded-md bg-brand-500/10 flex items-center justify-center shrink-0">
              <Play className="w-3.5 h-3.5 text-brand-500" />
            </div>
            <div className="flex-1 min-w-0 text-left">
              <p className="text-xs font-medium truncate">{track.title}</p>
              <p className="text-[10px] text-zinc-400">{track.artist} - {track.mood}</p>
            </div>
            <span className="text-[10px] text-zinc-400 font-mono shrink-0">{track.duration}</span>
          </button>
        ))}
      </div>
    </div>
  )
}

function UploadsPanel() {
  return (
    <div className="flex flex-col items-center justify-center h-48 gap-3">
      <div className="w-16 h-16 rounded-xl border-2 border-dashed border-zinc-300 dark:border-border-dark flex items-center justify-center">
        <Upload className="w-6 h-6 text-zinc-400" />
      </div>
      <p className="text-xs text-zinc-400 text-center">
        Drag and drop files here<br />or click to browse
      </p>
      <button className="text-xs px-4 py-2 rounded-lg border border-zinc-200 dark:border-border-dark text-zinc-500 hover:bg-zinc-100 dark:hover:bg-surface-dark-hover transition-colors">
        Browse Files
      </button>
    </div>
  )
}
