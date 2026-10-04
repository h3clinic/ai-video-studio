import { DbConnection, tables } from './module_bindings'
import { DbConnection as CollaborationConnection, tables as collaborationTables } from './module_bindings/collaboration'
import { installCspSafeCodecs } from './spacetimeCsp'

installCspSafeCodecs()

let conn: DbConnection | null = null
let ready = false
let myIdentity: { isEqual(other: unknown): boolean } | null = null
const TOKEN_KEY = 'aivs_stdb_token'
const listeners = new Set<() => void>()

const COLORS = ['#4ade80', '#60a5fa', '#f472b6', '#facc15', '#c084fc', '#fb923c', '#34d399', '#f87171']

export function pickColor(): string {
  return COLORS[Math.floor(Math.random() * COLORS.length)]
}

export function getConnection(): DbConnection | null {
  return conn
}

export function isReady(): boolean {
  return ready
}

export function getSessionEmail(): string | null {
  if (!conn || !ready || !myIdentity) return null
  for (const row of conn.db.session.iter()) {
    if (myIdentity.isEqual(row.identity)) return row.email
  }
  return null
}

export function subscribe(fn: () => void) {
  listeners.add(fn)
  return () => { listeners.delete(fn) }
}

function notify() {
  listeners.forEach(fn => fn())
}

export function connect(): Promise<DbConnection> {
  return new Promise((resolve, reject) => {
    if (conn) return resolve(conn)

    conn = DbConnection.builder()
      .withUri('ws://127.0.0.1:3000')
      .withDatabaseName('ai-video-studio')
      .withToken(localStorage.getItem(TOKEN_KEY) ?? undefined)
      .onConnect((connection, identity, token) => {
        // The token keeps this device's identity, so a signed-in session survives relaunches.
        localStorage.setItem(TOKEN_KEY, token)
        myIdentity = identity
        connection.subscriptionBuilder()
          .onApplied(() => { ready = true; notify() })
          .subscribe([
            tables.collaborator,
            tables.project,
            tables.agentRun,
            tables.agentChange,
            tables.studioUser,
            tables.session,
          ])

        connection.db.collaborator.onInsert(() => notify())
        connection.db.collaborator.onUpdate(() => notify())
        connection.db.collaborator.onDelete(() => notify())
        connection.db.agentRun.onInsert(() => notify())
        connection.db.agentRun.onUpdate(() => notify())
        connection.db.agentChange.onInsert(() => notify())
        connection.db.project.onInsert(() => notify())
        connection.db.studioUser.onInsert(() => notify())
        connection.db.studioUser.onUpdate(() => notify())
        connection.db.session.onInsert(() => notify())
        connection.db.session.onUpdate(() => notify())
        connection.db.session.onDelete(() => notify())
        resolve(connection)
      })
      .onDisconnect(() => {
        conn = null
        ready = false
        notify()
      })
      .onConnectError((_conn, err) => {
        console.error('[stdb] connect error', err)
        conn = null
        ready = false
        // A transport failure must not discard the user's identity.
        reject(err)
      })
      .build()
  })
}

// The Electron studio uses this isolated, private-schema database. The original
// public demo database remains untouched and is never copied into this one.
const COLLAB_TOKEN_KEY = 'aivs_collaboration_local_v1_identity'
let collaborationConnection: CollaborationConnection | null = null
let collaborationPending: Promise<CollaborationConnection> | null = null
let collaborationState: 'disconnected'|'connecting'|'connected'|'error' = 'disconnected'
let collaborationIdentity = ''
let collaborationError = ''
const collaborationListeners = new Set<() => void>()
const notifyCollaboration = () => collaborationListeners.forEach(fn=>fn())

export function subscribeCollaboration(fn:()=>void) {
  collaborationListeners.add(fn)
  return () => { collaborationListeners.delete(fn) }
}
export function getCollaborationConnection() { return collaborationState==='connected' ? collaborationConnection : null }
export function getCollaborationSnapshot() {
  const c=getCollaborationConnection()
  return {
    state:collaborationState, identity:collaborationIdentity, error:collaborationError,
    endpoint:'Local SpacetimeDB · 127.0.0.1:3000',
    projects:c ? Array.from(c.db.myProjects.iter()) : [],
    members:c ? Array.from(c.db.myMembers.iter()) : [],
    drafts:c ? Array.from(c.db.myDrafts.iter()) : [],
    edits:c ? Array.from(c.db.myEdits.iter()) : [],
    timings:c ? Array.from(c.db.myTimings.iter()) : [],
    activity:c ? Array.from(c.db.myActivity.iter()) : [],
  }
}
export function connectCollaboration(): Promise<CollaborationConnection> {
  if(collaborationState==='connected' && collaborationConnection) return Promise.resolve(collaborationConnection)
  if(collaborationPending) return collaborationPending
  collaborationState='connecting'; collaborationError=''; notifyCollaboration()
  // Defer construction one microtask so a synchronous builder error cannot
  // overwrite fail()'s null reset with a permanently rejected pending promise.
  collaborationPending=Promise.resolve().then(()=>new Promise<CollaborationConnection>((resolve,reject)=>{
    const fail=(message:string)=>{
      collaborationState='error'; collaborationError=message; collaborationPending=null
      collaborationConnection?.disconnect(); collaborationConnection=null
      notifyCollaboration(); reject(new Error(message))
    }
    const timeout=window.setTimeout(()=>fail('Local collaboration is unavailable. Your local jobs and artifacts are unchanged.'),8000)
    try {
      const token=localStorage.getItem(COLLAB_TOKEN_KEY) ?? undefined
      collaborationConnection=CollaborationConnection.builder()
        .withUri('ws://127.0.0.1:3000')
        .withDatabaseName('ai-video-studio-collaboration')
        .withToken(token)
        .onConnect((connection,identity,newToken)=>{
          localStorage.setItem(COLLAB_TOKEN_KEY,newToken)
          collaborationIdentity=identity.toHexString()
          for(const table of [connection.db.myProjects,connection.db.myMembers,connection.db.myDrafts,connection.db.myEdits,connection.db.myTimings,connection.db.myActivity]) {
            table.onInsert(()=>notifyCollaboration()); table.onDelete(()=>notifyCollaboration())
          }
          connection.subscriptionBuilder().onApplied(()=>{
            window.clearTimeout(timeout); collaborationState='connected'; collaborationError=''; collaborationPending=null
            notifyCollaboration(); resolve(connection)
          }).onError(()=>{window.clearTimeout(timeout);fail('Collaboration schema subscription failed. No shared changes were sent.')})
            .subscribe([collaborationTables.myProjects,collaborationTables.myMembers,collaborationTables.myDrafts,collaborationTables.myEdits,collaborationTables.myTimings,collaborationTables.myActivity])
        })
        .onConnectError(()=>{window.clearTimeout(timeout);fail('Local collaboration is unavailable. Start the local database and reconnect.')})
        .onDisconnect(()=>{
          window.clearTimeout(timeout)
          if(collaborationState!=='error') { collaborationState='disconnected'; collaborationError='Shared editing disconnected. Unsaved changes are not synchronized.' }
          collaborationConnection=null; collaborationPending=null; notifyCollaboration()
        }).build()
    } catch { window.clearTimeout(timeout);fail('Could not connect to the local collaboration database.') }
  }))
  return collaborationPending
}
