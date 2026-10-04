import GaussianWorkspace from './GaussianWorkspace'
export default function AgentPanel({onClose}: {onClose: () => void}) {
  return <aside className="w-96 h-full border-l border-zinc-700 flex flex-col"><button onClick={onClose} className="p-3 text-right">Close agents</button><GaussianWorkspace compact /></aside>
}
