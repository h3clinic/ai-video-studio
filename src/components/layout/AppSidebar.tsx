import { useNavigate } from 'react-router-dom'
import {
  Home, FolderOpen, Palette, LayoutTemplate, Settings,
  Sparkles, Video, HelpCircle,
} from 'lucide-react'

interface Props {
  activePage: string
}

const NAV_ITEMS = [
  { id: 'home', label: 'Home', icon: Home, path: '/' },
  { id: 'projects', label: 'Projects', icon: FolderOpen, path: '/' },
  { id: 'templates', label: 'Templates', icon: LayoutTemplate, path: '/' },
  { id: 'brand', label: 'Brand Kit', icon: Palette, path: '/' },
  { id: 'agents', label: 'AI Agents', icon: Sparkles, path: '/' },
]

const BOTTOM_ITEMS = [
  { id: 'help', label: 'Help', icon: HelpCircle },
  { id: 'settings', label: 'Settings', icon: Settings },
]

export default function AppSidebar({ activePage }: Props) {
  const navigate = useNavigate()

  return (
    <aside className="w-[68px] h-full flex flex-col items-center py-4 border-r border-zinc-200 dark:border-border-dark bg-zinc-50 dark:bg-surface-dark-alt shrink-0">
      {/* Logo */}
      <button
        onClick={() => navigate('/')}
        className="w-10 h-10 rounded-xl bg-brand-600 flex items-center justify-center mb-6"
      >
        <Video className="w-5 h-5 text-white" />
      </button>

      {/* Main nav */}
      <nav className="flex flex-col items-center gap-1 flex-1">
        {NAV_ITEMS.map(item => (
          <button
            key={item.id}
            onClick={() => navigate(item.path)}
            className={`group flex flex-col items-center gap-1 w-14 py-2 rounded-lg transition-colors ${
              activePage === item.id
                ? 'bg-brand-500/10 text-brand-500'
                : 'text-zinc-400 hover:text-zinc-600 dark:hover:text-zinc-200 hover:bg-zinc-100 dark:hover:bg-surface-dark-hover'
            }`}
          >
            <item.icon className="w-5 h-5" />
            <span className="text-[10px] font-medium">{item.label}</span>
          </button>
        ))}
      </nav>

      {/* Bottom nav */}
      <div className="flex flex-col items-center gap-1">
        {BOTTOM_ITEMS.map(item => (
          <button
            key={item.id}
            className="group flex flex-col items-center gap-1 w-14 py-2 rounded-lg text-zinc-400 hover:text-zinc-600 dark:hover:text-zinc-200 hover:bg-zinc-100 dark:hover:bg-surface-dark-hover transition-colors"
          >
            <item.icon className="w-5 h-5" />
            <span className="text-[10px] font-medium">{item.label}</span>
          </button>
        ))}
      </div>
    </aside>
  )
}
