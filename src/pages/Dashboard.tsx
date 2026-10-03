import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { motion } from 'framer-motion'
import {
  Play, Plus, Search, Sparkles, Video, Clock, Star, Folder,
  TrendingUp, Instagram, Youtube, Film, Tv, LayoutGrid,
  ChevronRight, MoreHorizontal, Trash2, Copy, Pencil,
} from 'lucide-react'
import AppSidebar from '@/components/layout/AppSidebar'
import TopBar from '@/components/layout/TopBar'

const TEMPLATES = [
  { id: 1, title: 'YouTube Explainer', category: 'YouTube', duration: '8-12 min', thumbnail: '🎬', color: 'from-red-100 to-red-50 dark:from-red-500/20 dark:to-red-900/20' },
  { id: 2, title: 'TikTok Promo', category: 'TikTok', duration: '15-60 sec', thumbnail: '📱', color: 'from-pink-100 to-purple-50 dark:from-pink-500/20 dark:to-purple-900/20' },
  { id: 3, title: 'Instagram Reel', category: 'Instagram', duration: '30-90 sec', thumbnail: '📸', color: 'from-amber-100 to-pink-50 dark:from-amber-500/20 dark:to-pink-900/20' },
  { id: 4, title: 'Product Demo', category: 'Business', duration: '2-5 min', thumbnail: '🎯', color: 'from-blue-100 to-indigo-50 dark:from-blue-500/20 dark:to-indigo-900/20' },
  { id: 5, title: 'Course Lecture', category: 'Education', duration: '10-30 min', thumbnail: '📚', color: 'from-emerald-100 to-teal-50 dark:from-emerald-500/20 dark:to-teal-900/20' },
  { id: 6, title: 'Podcast Clip', category: 'Podcast', duration: '1-3 min', thumbnail: '🎙️', color: 'from-violet-100 to-purple-50 dark:from-violet-500/20 dark:to-purple-900/20' },
]

const RECENT_PROJECTS = [
  { id: 'p1', title: 'Q4 Product Launch', duration: '2:34', lastEdited: '2 hours ago', thumbnail: '🚀', status: 'draft' },
  { id: 'p2', title: 'Team Onboarding Video', duration: '5:12', lastEdited: '1 day ago', thumbnail: '👋', status: 'published' },
  { id: 'p3', title: 'Customer Testimonial', duration: '1:45', lastEdited: '3 days ago', thumbnail: '⭐', status: 'draft' },
  { id: 'p4', title: 'Brand Story 2024', duration: '3:20', lastEdited: '1 week ago', thumbnail: '🎬', status: 'rendering' },
  { id: 'p5', title: 'Social Ad - Summer', duration: '0:30', lastEdited: '2 weeks ago', thumbnail: '☀️', status: 'published' },
  { id: 'p6', title: 'Tutorial: Getting Started', duration: '7:15', lastEdited: '2 weeks ago', thumbnail: '📖', status: 'draft' },
]

const CATEGORIES = [
  { label: 'All', icon: LayoutGrid },
  { label: 'YouTube', icon: Youtube },
  { label: 'Instagram', icon: Instagram },
  { label: 'TikTok', icon: Film },
  { label: 'Business', icon: TrendingUp },
  { label: 'Education', icon: Tv },
]

export default function Dashboard() {
  const navigate = useNavigate()
  const [prompt, setPrompt] = useState('')
  const [activeCategory, setActiveCategory] = useState('All')
  const [hoveredProject, setHoveredProject] = useState<string | null>(null)

  return (
    <div className="flex h-full">
      <AppSidebar activePage="home" />

      <div className="flex-1 flex flex-col overflow-hidden">
        <TopBar />

        <main className="flex-1 overflow-y-auto px-8 py-6">
          {/* Hero / Create Section */}
          <motion.section
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="mb-10"
          >
            <h1 className="text-3xl font-bold mb-2">
              Create a video with AI
            </h1>
            <p className="text-zinc-500 dark:text-zinc-400 mb-6 text-sm">
              Describe your video and our agents will handle the rest
            </p>

            <div className="relative max-w-2xl">
              <div className="flex items-center gap-3 rounded-xl border border-zinc-200 dark:border-border-dark bg-white dark:bg-surface-dark-elevated px-4 py-3 focus-within:border-brand-500 focus-within:ring-1 focus-within:ring-brand-500/30 transition-all">
                <Sparkles className="w-5 h-5 text-brand-500 shrink-0" />
                <input
                  type="text"
                  value={prompt}
                  onChange={e => setPrompt(e.target.value)}
                  placeholder="A 60-second explainer about our new feature..."
                  className="flex-1 bg-transparent outline-none text-sm placeholder:text-zinc-400 dark:placeholder:text-zinc-500"
                  onKeyDown={e => {
                    if (e.key === 'Enter' && prompt.trim()) {
                      navigate('/editor/new')
                    }
                  }}
                />
                <button
                  onClick={() => prompt.trim() && navigate('/editor/new')}
                  className="flex items-center gap-2 bg-brand-600 hover:bg-brand-700 text-white rounded-lg px-4 py-2 text-sm font-medium transition-colors"
                >
                  <Plus className="w-4 h-4" />
                  Generate
                </button>
              </div>

              <div className="flex gap-2 mt-3">
                {['Product demo in 30 seconds', 'YouTube tutorial with voiceover', 'Instagram ad for summer sale'].map(suggestion => (
                  <button
                    key={suggestion}
                    onClick={() => setPrompt(suggestion)}
                    className="text-xs px-3 py-1.5 rounded-full border border-zinc-200 dark:border-border-dark text-zinc-500 dark:text-zinc-400 hover:border-brand-500/40 hover:text-brand-500 transition-colors"
                  >
                    {suggestion}
                  </button>
                ))}
              </div>
            </div>
          </motion.section>

          {/* Templates */}
          <motion.section
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
            className="mb-10"
          >
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-lg font-semibold">Start from a template</h2>
              <button className="text-sm text-brand-500 hover:text-brand-400 flex items-center gap-1">
                View all <ChevronRight className="w-4 h-4" />
              </button>
            </div>

            <div className="flex gap-2 mb-4">
              {CATEGORIES.map(cat => (
                <button
                  key={cat.label}
                  onClick={() => setActiveCategory(cat.label)}
                  className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-colors ${
                    activeCategory === cat.label
                      ? 'bg-brand-600 text-white'
                      : 'text-zinc-500 dark:text-zinc-400 hover:bg-zinc-100 dark:hover:bg-surface-dark-hover'
                  }`}
                >
                  <cat.icon className="w-3.5 h-3.5" />
                  {cat.label}
                </button>
              ))}
            </div>

            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
              {TEMPLATES.map(tpl => (
                <motion.button
                  key={tpl.id}
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                  onClick={() => navigate('/editor/new')}
                  className={`group relative rounded-xl bg-gradient-to-br ${tpl.color} border border-zinc-200/50 dark:border-border-dark-subtle p-4 text-left transition-all hover:border-brand-500/30`}
                >
                  <div className="text-3xl mb-3">{tpl.thumbnail}</div>
                  <h3 className="text-sm font-medium mb-1">{tpl.title}</h3>
                  <p className="text-xs text-zinc-500 dark:text-zinc-400">{tpl.duration}</p>
                  <span className="absolute top-3 right-3 text-[10px] px-2 py-0.5 rounded-full bg-black/5 dark:bg-white/5 text-zinc-600 dark:text-zinc-400">
                    {tpl.category}
                  </span>
                </motion.button>
              ))}
            </div>
          </motion.section>

          {/* Recent Projects */}
          <motion.section
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
          >
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-lg font-semibold flex items-center gap-2">
                <Clock className="w-5 h-5 text-zinc-400" />
                Recent Projects
              </h2>
              <button className="text-sm text-brand-500 hover:text-brand-400 flex items-center gap-1">
                All projects <ChevronRight className="w-4 h-4" />
              </button>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
              {RECENT_PROJECTS.map(project => (
                <motion.div
                  key={project.id}
                  whileHover={{ scale: 1.01 }}
                  onClick={() => navigate(`/editor/${project.id}`)}
                  onMouseEnter={() => setHoveredProject(project.id)}
                  onMouseLeave={() => setHoveredProject(null)}
                  className="group relative flex items-center gap-4 rounded-xl border border-zinc-200 dark:border-border-dark bg-white dark:bg-surface-dark-elevated p-4 cursor-pointer hover:border-brand-500/30 transition-all"
                >
                  <div className="w-16 h-12 rounded-lg bg-zinc-100 dark:bg-surface-dark-elevated flex items-center justify-center text-2xl shrink-0">
                    {project.thumbnail}
                  </div>
                  <div className="flex-1 min-w-0">
                    <h3 className="text-sm font-medium truncate">{project.title}</h3>
                    <div className="flex items-center gap-2 mt-1">
                      <span className="text-xs text-zinc-400">{project.duration}</span>
                      <span className="text-zinc-300 dark:text-zinc-600">|</span>
                      <span className="text-xs text-zinc-400">{project.lastEdited}</span>
                    </div>
                  </div>
                  <div className="flex items-center gap-1">
                    <span className={`text-[10px] px-2 py-0.5 rounded-full font-medium ${
                      project.status === 'published'
                        ? 'bg-emerald-500/10 text-emerald-500'
                        : project.status === 'rendering'
                        ? 'bg-amber-500/10 text-amber-500'
                        : 'bg-zinc-500/10 text-zinc-400'
                    }`}>
                      {project.status}
                    </span>
                    {hoveredProject === project.id && (
                      <motion.button
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        onClick={e => e.stopPropagation()}
                        className="p-1 rounded-md hover:bg-zinc-100 dark:hover:bg-surface-dark-hover"
                      >
                        <MoreHorizontal className="w-4 h-4 text-zinc-400" />
                      </motion.button>
                    )}
                  </div>
                </motion.div>
              ))}
            </div>
          </motion.section>
        </main>
      </div>
    </div>
  )
}
