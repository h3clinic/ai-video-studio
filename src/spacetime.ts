import { DbConnection, tables } from './module_bindings'

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
        localStorage.removeItem(TOKEN_KEY)
        reject(err)
      })
      .build()
  })
}
