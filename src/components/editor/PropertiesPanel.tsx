import { useState } from 'react'
import { X, RotateCcw, Lock, Unlock, AlignLeft, AlignCenter, AlignRight, ChevronDown, Palette } from 'lucide-react'

interface Props {
  onClose: () => void
}

export default function PropertiesPanel({ onClose }: Props) {
  const [locked, setLocked] = useState(false)

  return (
    <aside className="w-64 h-full border-l border-zinc-200 dark:border-border-dark bg-zinc-50 dark:bg-surface-dark-alt shrink-0 flex flex-col">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-zinc-200 dark:border-border-dark">
        <span className="text-xs font-semibold uppercase tracking-wider text-zinc-500 dark:text-zinc-400">Properties</span>
        <button onClick={onClose} className="p-1 rounded-md text-zinc-400 hover:text-zinc-600 dark:hover:text-zinc-200 hover:bg-zinc-100 dark:hover:bg-surface-dark-hover transition-colors">
          <X className="w-4 h-4" />
        </button>
      </div>

      <div className="flex-1 overflow-y-auto p-4 space-y-5">
        {/* Transform */}
        <PropertySection title="Transform">
          <div className="grid grid-cols-2 gap-2">
            <PropertyInput label="X" value="120" suffix="px" />
            <PropertyInput label="Y" value="80" suffix="px" />
            <PropertyInput label="W" value="480" suffix="px" />
            <PropertyInput label="H" value="270" suffix="px" />
          </div>
          <div className="flex items-center gap-2 mt-2">
            <PropertyInput label="Rotate" value="0" suffix="deg" />
            <button
              onClick={() => setLocked(l => !l)}
              className="p-1.5 rounded-md text-zinc-400 hover:text-zinc-600 dark:hover:text-zinc-200 hover:bg-zinc-100 dark:hover:bg-surface-dark-hover transition-colors mt-4"
            >
              {locked ? <Lock className="w-3.5 h-3.5" /> : <Unlock className="w-3.5 h-3.5" />}
            </button>
          </div>
        </PropertySection>

        {/* Appearance */}
        <PropertySection title="Appearance">
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-xs text-zinc-400">Opacity</span>
              <span className="text-xs font-mono text-zinc-500">100%</span>
            </div>
            <input
              type="range"
              min="0"
              max="100"
              defaultValue="100"
              className="w-full h-1 bg-zinc-200 dark:bg-zinc-700 rounded-full appearance-none [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-3 [&::-webkit-slider-thumb]:h-3 [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-brand-500 [&::-webkit-slider-thumb]:cursor-pointer"
            />
          </div>

          <div className="flex items-center justify-between mt-3">
            <span className="text-xs text-zinc-400">Blend Mode</span>
            <button className="flex items-center gap-1 text-xs text-zinc-500 bg-white dark:bg-surface-dark-elevated border border-zinc-200 dark:border-border-dark rounded-md px-2 py-1">
              Normal <ChevronDown className="w-3 h-3" />
            </button>
          </div>

          <div className="flex items-center justify-between mt-3">
            <span className="text-xs text-zinc-400">Fill Color</span>
            <div className="flex items-center gap-1.5">
              <div className="w-5 h-5 rounded-md bg-brand-500 border border-zinc-200 dark:border-border-dark cursor-pointer" />
              <span className="text-xs font-mono text-zinc-500">#6366F1</span>
            </div>
          </div>

          <div className="flex items-center justify-between mt-3">
            <span className="text-xs text-zinc-400">Border</span>
            <button className="flex items-center gap-1 text-xs text-zinc-500 bg-white dark:bg-surface-dark-elevated border border-zinc-200 dark:border-border-dark rounded-md px-2 py-1">
              None <ChevronDown className="w-3 h-3" />
            </button>
          </div>

          <div className="flex items-center justify-between mt-3">
            <span className="text-xs text-zinc-400">Corner Radius</span>
            <PropertyInput label="" value="8" suffix="px" inline />
          </div>
        </PropertySection>

        {/* Text */}
        <PropertySection title="Text">
          <div className="flex items-center justify-between">
            <span className="text-xs text-zinc-400">Font</span>
            <button className="flex items-center gap-1 text-xs text-zinc-500 bg-white dark:bg-surface-dark-elevated border border-zinc-200 dark:border-border-dark rounded-md px-2 py-1">
              Inter <ChevronDown className="w-3 h-3" />
            </button>
          </div>
          <div className="grid grid-cols-2 gap-2 mt-2">
            <PropertyInput label="Size" value="24" suffix="px" />
            <PropertyInput label="Line H" value="1.4" suffix="" />
          </div>
          <div className="flex gap-1 mt-2">
            <button className="p-1.5 rounded-md bg-brand-500/10 text-brand-500">
              <AlignLeft className="w-3.5 h-3.5" />
            </button>
            <button className="p-1.5 rounded-md text-zinc-400 hover:text-zinc-600 dark:hover:text-zinc-200 hover:bg-zinc-100 dark:hover:bg-surface-dark-hover transition-colors">
              <AlignCenter className="w-3.5 h-3.5" />
            </button>
            <button className="p-1.5 rounded-md text-zinc-400 hover:text-zinc-600 dark:hover:text-zinc-200 hover:bg-zinc-100 dark:hover:bg-surface-dark-hover transition-colors">
              <AlignRight className="w-3.5 h-3.5" />
            </button>
          </div>
        </PropertySection>

        {/* Animation */}
        <PropertySection title="Animation">
          <div className="flex items-center justify-between">
            <span className="text-xs text-zinc-400">Enter</span>
            <button className="flex items-center gap-1 text-xs text-zinc-500 bg-white dark:bg-surface-dark-elevated border border-zinc-200 dark:border-border-dark rounded-md px-2 py-1">
              Fade In <ChevronDown className="w-3 h-3" />
            </button>
          </div>
          <div className="flex items-center justify-between mt-2">
            <span className="text-xs text-zinc-400">Exit</span>
            <button className="flex items-center gap-1 text-xs text-zinc-500 bg-white dark:bg-surface-dark-elevated border border-zinc-200 dark:border-border-dark rounded-md px-2 py-1">
              Fade Out <ChevronDown className="w-3 h-3" />
            </button>
          </div>
          <div className="grid grid-cols-2 gap-2 mt-2">
            <PropertyInput label="Duration" value="0.5" suffix="s" />
            <PropertyInput label="Delay" value="0" suffix="s" />
          </div>
        </PropertySection>
      </div>
    </aside>
  )
}

function PropertySection({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div>
      <h3 className="text-xs font-semibold text-zinc-600 dark:text-zinc-300 mb-2">{title}</h3>
      {children}
    </div>
  )
}

function PropertyInput({
  label,
  value,
  suffix,
  inline,
}: {
  label: string
  value: string
  suffix: string
  inline?: boolean
}) {
  if (inline) {
    return (
      <div className="flex items-center gap-1">
        <input
          type="text"
          defaultValue={value}
          className="w-12 bg-white dark:bg-surface-dark-elevated border border-zinc-200 dark:border-border-dark rounded-md px-2 py-1 text-xs text-right outline-none focus:border-brand-500 transition-colors"
        />
        {suffix && <span className="text-[10px] text-zinc-400">{suffix}</span>}
      </div>
    )
  }

  return (
    <div>
      {label && <span className="text-[10px] text-zinc-400 mb-0.5 block">{label}</span>}
      <div className="flex items-center gap-1">
        <input
          type="text"
          defaultValue={value}
          className="w-full bg-white dark:bg-surface-dark-elevated border border-zinc-200 dark:border-border-dark rounded-md px-2 py-1 text-xs outline-none focus:border-brand-500 transition-colors"
        />
        {suffix && <span className="text-[10px] text-zinc-400">{suffix}</span>}
      </div>
    </div>
  )
}
