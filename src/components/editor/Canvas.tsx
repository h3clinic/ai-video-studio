import { useState } from 'react'
import { Play, Pause, SkipBack, SkipForward, Volume2, Maximize2 } from 'lucide-react'

export default function Canvas() {
  const [playing, setPlaying] = useState(false)
  const [currentTime, setCurrentTime] = useState(0)
  const totalTime = 154

  const formatTime = (s: number) => {
    const m = Math.floor(s / 60)
    const sec = Math.floor(s % 60)
    return `${m}:${sec.toString().padStart(2, '0')}`
  }

  return (
    <div className="flex-1 flex flex-col items-center justify-center bg-zinc-100 dark:bg-surface-dark p-6 gap-4">
      {/* Video preview area */}
      <div className="relative w-full max-w-[720px] aspect-video bg-black rounded-lg overflow-hidden shadow-2xl shadow-black/20">
        {/* Placeholder content */}
        <div className="absolute inset-0 bg-gradient-to-br from-zinc-900 via-zinc-800 to-zinc-900 flex flex-col items-center justify-center gap-3">
          <div className="w-16 h-16 rounded-full bg-white/5 flex items-center justify-center border border-white/10">
            <Play className="w-7 h-7 text-white/40 ml-1" />
          </div>
          <p className="text-white/30 text-sm">16:9 Preview</p>
          <p className="text-white/15 text-xs">1920 x 1080</p>
        </div>

        {/* Overlay elements preview */}
        <div className="absolute top-8 left-8 right-8 pointer-events-none">
          <div className="inline-block bg-white/10 backdrop-blur-sm border border-white/10 rounded-md px-4 py-2">
            <p className="text-white text-lg font-semibold">Your Title Here</p>
          </div>
        </div>

        <div className="absolute bottom-8 left-8 pointer-events-none">
          <div className="bg-white/10 backdrop-blur-sm border border-white/10 rounded-md px-3 py-1.5">
            <p className="text-white/80 text-xs font-medium">Speaker Name</p>
            <p className="text-white/50 text-[10px]">Title / Role</p>
          </div>
        </div>
      </div>

      {/* Playback controls */}
      <div className="flex items-center gap-4 w-full max-w-[720px]">
        <div className="flex items-center gap-1">
          <button className="p-1.5 rounded-md text-zinc-400 hover:text-zinc-600 dark:hover:text-zinc-200 transition-colors">
            <SkipBack className="w-4 h-4" />
          </button>
          <button
            onClick={() => setPlaying(p => !p)}
            className="p-2 rounded-full bg-brand-600 hover:bg-brand-700 text-white transition-colors"
          >
            {playing ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4 ml-0.5" />}
          </button>
          <button className="p-1.5 rounded-md text-zinc-400 hover:text-zinc-600 dark:hover:text-zinc-200 transition-colors">
            <SkipForward className="w-4 h-4" />
          </button>
        </div>

        <span className="text-xs text-zinc-400 font-mono w-10 text-right">
          {formatTime(currentTime)}
        </span>

        {/* Progress bar */}
        <div className="flex-1 group relative h-1.5 bg-zinc-200 dark:bg-zinc-700 rounded-full cursor-pointer">
          <div
            className="absolute inset-y-0 left-0 bg-brand-500 rounded-full transition-all"
            style={{ width: `${(currentTime / totalTime) * 100}%` }}
          />
          <div
            className="absolute top-1/2 -translate-y-1/2 w-3 h-3 bg-white border-2 border-brand-500 rounded-full opacity-0 group-hover:opacity-100 transition-opacity shadow-sm"
            style={{ left: `${(currentTime / totalTime) * 100}%`, marginLeft: -6 }}
          />
        </div>

        <span className="text-xs text-zinc-400 font-mono w-10">
          {formatTime(totalTime)}
        </span>

        <div className="flex items-center gap-1">
          <button className="p-1.5 rounded-md text-zinc-400 hover:text-zinc-600 dark:hover:text-zinc-200 transition-colors">
            <Volume2 className="w-4 h-4" />
          </button>
          <button className="p-1.5 rounded-md text-zinc-400 hover:text-zinc-600 dark:hover:text-zinc-200 transition-colors">
            <Maximize2 className="w-4 h-4" />
          </button>
        </div>
      </div>
    </div>
  )
}
