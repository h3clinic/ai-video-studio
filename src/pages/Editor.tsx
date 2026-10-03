import { useState } from 'react'
import EditorTopBar from '@/components/editor/EditorTopBar'
import MediaPanel from '@/components/editor/MediaPanel'
import Canvas from '@/components/editor/Canvas'
import PropertiesPanel from '@/components/editor/PropertiesPanel'
import Timeline from '@/components/editor/Timeline'
import AgentPanel from '@/components/agents/AgentPanel'

export type PanelTab = 'script' | 'media' | 'text' | 'elements' | 'audio' | 'uploads' | 'agents'

export default function Editor() {
  const [activeTab, setActiveTab] = useState<PanelTab>('media')
  const [showProperties, setShowProperties] = useState(true)
  const [showAgentPanel, setShowAgentPanel] = useState(false)
  const [timelineHeight, setTimelineHeight] = useState(220)

  return (
    <div className="flex flex-col h-full">
      <EditorTopBar
        onToggleAgents={() => setShowAgentPanel(p => !p)}
        agentsOpen={showAgentPanel}
      />

      <div className="flex flex-1 overflow-hidden">
        {/* Left Panel - Media/Assets */}
        <MediaPanel activeTab={activeTab} onTabChange={setActiveTab} />

        {/* Center - Canvas */}
        <div className="flex-1 flex flex-col overflow-hidden">
          <Canvas />
        </div>

        {/* Right Panel - Properties or Agent */}
        {showAgentPanel ? (
          <AgentPanel onClose={() => setShowAgentPanel(false)} />
        ) : showProperties ? (
          <PropertiesPanel onClose={() => setShowProperties(false)} />
        ) : null}
      </div>

      {/* Bottom - Timeline */}
      <Timeline height={timelineHeight} onResize={setTimelineHeight} />
    </div>
  )
}
