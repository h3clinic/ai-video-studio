import { useState, useRef, useCallback } from 'react'
import {
  Plus, Trash2, Eye, EyeOff, Lock, Unlock, Volume2, VolumeX,
  Scissors, Copy, ChevronUp, GripHorizontal, Magnet,
  SkipBack, SkipForward,
} from 'lucide-react'

interface Props {
  height: number
  onResize: (h: number) => void
}

interface Track {
  id: string
  label: string
  type: 'video' | 'audio' | 'text' | 'effect'
  visible: boolean
  locked: boolean
  muted: boolean
  clips: Clip[]
}

interface Clip {
  id: string
  label: string
  start: number
  duration: number
  color: string
}

const INITIAL_TRACKS: Track[] = [
  {
    id: 'v1',
    label: 'Video 1',
    type: 'video',
    visible: true,
    locked: false,
    muted: false,
    clips: [
      { id: 'c1', label: 'City Timelapse', start: 0, duration: 4, color: 'bg-sky-500/60' },
      { id: 'c2', label: 'Product Shot', start: 4.5, duration: 3, color: 'bg-emerald-500/60' },
      { id: 'c3', label: 'Team Interview', start: 8, duration: 5, color: 'bg-violet-500/60' },
    ],
  },
  {
    id: 'v2',
    label: 'Video 2',
    type: 'video',
    visible: true,
    locked: false,
    muted: false,
    clips: [
      { id: 'c4', label: 'B-Roll Overlay', start: 2, duration: 3, color: 'bg-amber-500/60' },
    ],
  },
  {
    id: 't1',
    label: 'Text',
    type: 'text',
    visible: true,
    locked: false,
    muted: false,
    clips: [
      { id: 'c5', label: 'Title Card', start: 0, duration: 3, color: 'bg-pink-500/60' },
      { id: 'c6', label: 'Lower Third', start: 4, duration: 4, color: 'bg-pink-400/60' },
      { id: 'c7', label: 'CTA Text', start: 10, duration: 2.5, color: 'bg-pink-500/60' },
    ],
  },
  {
    id: 'a1',
    label: 'Music',
    type: 'audio',
    visible: true,
    locked: false,
    muted: false,
    clips: [
      { id: 'c8', label: 'Upbeat Corporate', start: 0, duration: 13, color: 'bg-orange-500/40' },
    ],
  },
  {
    id: 'a2',
    label: 'Voiceover',
    type: 'audio',
    visible: true,
    locked: false,
    muted: false,
    clips: [
      { id: 'c9', label: 'VO Take 3', start: 0.5, duration: 5, color: 'bg-teal-500/40' },
      { id: 'c10', label: 'VO Take 4', start: 7, duration: 5.5, color: 'bg-teal-400/40' },
    ],
  },
]

const TOTAL_DURATION = 15
const PX_PER_SECOND = 80

export default function Timeline({ height, onResize }: Props) {
  const [tracks, setTracks] = useState(INITIAL_TRACKS)
  const [playhead, setPlayhead] = useState(2.3)
  const [snapping, setSnapping] = useState(true)
  const [selectedClip, setSelectedClip] = useState<string | null>(null)
  const resizeRef = useRef<HTMLDivElement>(null)
  const isDragging = useRef(false)

  const handleResizeStart = useCallback((e: React.MouseEvent) => {
    e.preventDefault()
    isDragging.current = true
    const startY = e.clientY
    const startHeight = height

    const handleMove = (e: MouseEvent) => {
      if (!isDragging.current) return
      const delta = startY - e.clientY
      onResize(Math.max(150, Math.min(400, startHeight + delta)))
    }

    const handleUp = () => {
      isDragging.current = false
      document.removeEventListener('mousemove', handleMove)
      document.removeEventListener('mouseup', handleUp)
    }

    document.addEventListener('mousemove', handleMove)
    document.addEventListener('mouseup', handleUp)
  }, [height, onResize])

  const toggleTrackProp = (trackId: string, prop: 'visible' | 'locked' | 'muted') => {
    setTracks(ts => ts.map(t => t.id === trackId ? { ...t, [prop]: !t[prop] } : t))
  }

  return (
    <div
      className="border-t border-zinc-200 dark:border-border-dark bg-zinc-50 dark:bg-surface-dark-alt flex flex-col shrink-0"
      style={{ height }}
    >
      {/* Resize handle */}
      <div
        ref={resizeRef}
        onMouseDown={handleResizeStart}
        className="h-1.5 cursor-ns-resize flex items-center justify-center hover:bg-brand-500/10 transition-colors group"
      >
        <div className="w-8 h-0.5 rounded-full bg-zinc-300 dark:bg-zinc-600 group-hover:bg-brand-500 transition-colors" />
      </div>

      {/* Timeline toolbar */}
      <div className="flex items-center gap-2 px-3 py-1.5 border-b border-zinc-200 dark:border-border-dark">
        <button className="p-1 rounded text-zinc-400 hover:text-zinc-200 hover:bg-zinc-100 dark:hover:bg-surface-dark-hover transition-colors">
          <Plus className="w-3.5 h-3.5" />
        </button>
        <button className="p-1 rounded text-zinc-400 hover:text-zinc-200 hover:bg-zinc-100 dark:hover:bg-surface-dark-hover transition-colors">
          <Scissors className="w-3.5 h-3.5" />
        </button>
        <button className="p-1 rounded text-zinc-400 hover:text-zinc-200 hover:bg-zinc-100 dark:hover:bg-surface-dark-hover transition-colors">
          <Copy className="w-3.5 h-3.5" />
        </button>
        <button className="p-1 rounded text-zinc-400 hover:text-zinc-200 hover:bg-zinc-100 dark:hover:bg-surface-dark-hover transition-colors">
          <Trash2 className="w-3.5 h-3.5" />
        </button>

        <div className="h-4 w-px bg-zinc-200 dark:bg-border-dark" />

        <button
          onClick={() => setSnapping(s => !s)}
          className={`p-1 rounded transition-colors ${snapping ? 'text-brand-500 bg-brand-500/10' : 'text-zinc-400 hover:text-zinc-200'}`}
        >
          <Magnet className="w-3.5 h-3.5" />
        </button>

        <div className="flex-1" />

        <span className="text-[10px] font-mono text-zinc-400">
          {Math.floor(playhead / 60)}:{Math.floor(playhead % 60).toString().padStart(2, '0')}.{Math.floor((playhead % 1) * 30).toString().padStart(2, '0')}
          <span className="text-zinc-500 mx-1">/</span>
          {Math.floor(TOTAL_DURATION / 60)}:{Math.floor(TOTAL_DURATION % 60).toString().padStart(2, '0')}.00
        </span>
      </div>

      {/* Timeline body */}
      <div className="flex flex-1 overflow-hidden">
        {/* Track labels */}
        <div className="w-40 shrink-0 border-r border-zinc-200 dark:border-border-dark overflow-y-auto">
          {tracks.map(track => (
            <div
              key={track.id}
              className="flex items-center gap-1 px-2 h-8 border-b border-zinc-100 dark:border-border-dark-subtle text-xs hover:bg-zinc-100 dark:hover:bg-surface-dark-hover transition-colors group"
            >
              <GripHorizontal className="w-3 h-3 text-zinc-300 dark:text-zinc-600 cursor-grab opacity-0 group-hover:opacity-100 transition-opacity" />
              <span className={`w-1.5 h-1.5 rounded-full shrink-0 ${
                track.type === 'video' ? 'bg-sky-500' :
                track.type === 'audio' ? 'bg-orange-500' :
                track.type === 'text' ? 'bg-pink-500' : 'bg-emerald-500'
              }`} />
              <span className="flex-1 text-zinc-600 dark:text-zinc-300 truncate text-[11px] font-medium">{track.label}</span>
              <div className="flex gap-0.5 opacity-0 group-hover:opacity-100 transition-opacity">
                <button
                  onClick={() => toggleTrackProp(track.id, 'visible')}
                  className="p-0.5 rounded text-zinc-400 hover:text-zinc-200"
                >
                  {track.visible ? <Eye className="w-3 h-3" /> : <EyeOff className="w-3 h-3" />}
                </button>
                <button
                  onClick={() => toggleTrackProp(track.id, 'locked')}
                  className="p-0.5 rounded text-zinc-400 hover:text-zinc-200"
                >
                  {track.locked ? <Lock className="w-3 h-3" /> : <Unlock className="w-3 h-3" />}
                </button>
                {(track.type === 'audio') && (
                  <button
                    onClick={() => toggleTrackProp(track.id, 'muted')}
                    className="p-0.5 rounded text-zinc-400 hover:text-zinc-200"
                  >
                    {track.muted ? <VolumeX className="w-3 h-3" /> : <Volume2 className="w-3 h-3" />}
                  </button>
                )}
              </div>
            </div>
          ))}
        </div>

        {/* Timeline tracks */}
        <div className="flex-1 overflow-x-auto overflow-y-auto relative">
          {/* Time ruler */}
          <div className="sticky top-0 z-10 h-5 bg-zinc-100 dark:bg-surface-dark-alt border-b border-zinc-200 dark:border-border-dark flex">
            {Array.from({ length: TOTAL_DURATION + 1 }, (_, i) => (
              <div
                key={i}
                className="shrink-0 relative"
                style={{ width: PX_PER_SECOND }}
              >
                <span className="absolute left-1 top-0.5 text-[9px] font-mono text-zinc-400">
                  {Math.floor(i / 60)}:{(i % 60).toString().padStart(2, '0')}
                </span>
                <div className="absolute left-0 top-3 w-px h-2 bg-zinc-300 dark:bg-zinc-600" />
                <div className="absolute left-[50%] top-3.5 w-px h-1.5 bg-zinc-200 dark:bg-zinc-700" />
              </div>
            ))}
          </div>

          {/* Tracks */}
          {tracks.map(track => (
            <div
              key={track.id}
              className="relative h-8 border-b border-zinc-100 dark:border-border-dark-subtle"
              style={{ width: (TOTAL_DURATION + 1) * PX_PER_SECOND }}
            >
              {track.clips.map(clip => (
                <div
                  key={clip.id}
                  onClick={() => setSelectedClip(clip.id)}
                  className={`absolute top-1 h-6 rounded-md ${clip.color} border cursor-pointer flex items-center px-2 overflow-hidden transition-all ${
                    selectedClip === clip.id
                      ? 'border-brand-500 ring-1 ring-brand-500/30'
                      : 'border-white/10 dark:border-white/5 hover:border-white/30'
                  }`}
                  style={{
                    left: clip.start * PX_PER_SECOND,
                    width: clip.duration * PX_PER_SECOND,
                  }}
                >
                  <span className="text-[10px] text-white font-medium truncate drop-shadow-sm">
                    {clip.label}
                  </span>
                </div>
              ))}
            </div>
          ))}

          {/* Playhead */}
          <div
            className="absolute top-0 bottom-0 z-20 pointer-events-none"
            style={{ left: playhead * PX_PER_SECOND }}
          >
            <div className="w-3 h-3 bg-brand-500 rounded-sm rotate-45 -translate-x-1/2 -translate-y-0.5 border border-brand-400" />
            <div className="w-px bg-brand-500 absolute top-2.5 bottom-0 left-1/2 -translate-x-1/2" />
          </div>
        </div>
      </div>
    </div>
  )
}
