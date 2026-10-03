import { Search, Bell, Moon, Sun } from 'lucide-react'
import { useTheme } from '@/contexts/ThemeContext'

export default function TopBar() {
  const { theme, toggleTheme } = useTheme()

  return (
    <header className="drag-region h-12 flex items-center justify-between px-6 border-b border-zinc-200 dark:border-border-dark bg-white dark:bg-surface-dark-alt shrink-0">
      {/* Left spacer for traffic lights on macOS */}
      <div className="w-20" />

      {/* Search */}
      <div className="no-drag flex items-center gap-2 bg-zinc-100 dark:bg-surface-dark-elevated rounded-lg px-3 py-1.5 w-80">
        <Search className="w-4 h-4 text-zinc-400" />
        <input
          type="text"
          placeholder="Search projects, templates..."
          className="flex-1 bg-transparent outline-none text-sm placeholder:text-zinc-400 dark:placeholder:text-zinc-500"
        />
        <kbd className="text-[10px] px-1.5 py-0.5 rounded bg-zinc-200 dark:bg-surface-dark text-zinc-400 font-mono">
          {'⌘'}K
        </kbd>
      </div>

      {/* Right actions */}
      <div className="no-drag flex items-center gap-2">
        <button
          onClick={toggleTheme}
          className="p-2 rounded-lg text-zinc-400 hover:text-zinc-600 dark:hover:text-zinc-200 hover:bg-zinc-100 dark:hover:bg-surface-dark-hover transition-colors"
        >
          {theme === 'dark' ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />}
        </button>
        <button className="p-2 rounded-lg text-zinc-400 hover:text-zinc-600 dark:hover:text-zinc-200 hover:bg-zinc-100 dark:hover:bg-surface-dark-hover transition-colors relative">
          <Bell className="w-4 h-4" />
          <span className="absolute top-1.5 right-1.5 w-2 h-2 rounded-full bg-brand-500" />
        </button>
        <div className="w-8 h-8 rounded-full bg-gradient-to-br from-brand-500 to-purple-600 flex items-center justify-center text-white text-xs font-bold ml-1">
          A
        </div>
      </div>
    </header>
  )
}
