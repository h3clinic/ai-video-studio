import { useState, useEffect } from 'react'
import { connect, getConnection, getSessionEmail, isReady, subscribe as stdbSubscribe, connectCollaboration, getCollaborationSnapshot, getCollaborationConnection, subscribeCollaboration } from './spacetime'

interface StdbSnapshot {
  ready: boolean
  sessionEmail: string | null
  collaborators: any[]
  agentRuns: any[]
  agentChanges: any[]
  projects: any[]
  studioUsers: any[]
}

const EMPTY: StdbSnapshot = {
  ready: false, sessionEmail: null,
  collaborators: [], agentRuns: [], agentChanges: [], projects: [], studioUsers: [],
}

function takeSnapshot(): StdbSnapshot {
  const c = getConnection()
  if (!c || !isReady()) return EMPTY
  try {
    return {
      ready: true,
      sessionEmail: getSessionEmail(),
      collaborators: Array.from(c.db.collaborator.iter()),
      agentRuns: Array.from(c.db.agentRun.iter()),
      agentChanges: Array.from(c.db.agentChange.iter()),
      projects: Array.from(c.db.project.iter()),
      studioUsers: Array.from(c.db.studioUser.iter()),
    }
  } catch {
    return EMPTY
  }
}

export function useSpacetime() {
  const [snap, setSnap] = useState<StdbSnapshot>(EMPTY)
  const [failed, setFailed] = useState(false)

  useEffect(() => {
    const unsub = stdbSubscribe(() => setSnap(takeSnapshot()))
    let timer: number | undefined
    // Keep retrying: the desktop app may launch before the database server is up.
    const tryConnect = () => {
      connect().then(() => setFailed(false)).catch(() => {
        setFailed(true)
        timer = window.setTimeout(tryConnect, 2000)
      })
    }
    tryConnect()
    return () => { unsub(); clearTimeout(timer) }
  }, [])

  return {
    ...snap,
    failed: failed && !snap.ready,
    reducers: snap.ready ? getConnection()?.reducers ?? null : null,
    procedures: snap.ready ? getConnection()?.procedures ?? null : null,
  }
}

export function useCollaboration() {
  const [snapshot,setSnapshot]=useState(getCollaborationSnapshot)
  useEffect(()=>{
    const unsubscribe=subscribeCollaboration(()=>setSnapshot(getCollaborationSnapshot()))
    connectCollaboration().catch(()=>{ /* reflected in the connection badge */ })
    return unsubscribe
  },[])
  return {...snapshot,reconnect:connectCollaboration,reducers:getCollaborationConnection()?.reducers ?? null}
}
