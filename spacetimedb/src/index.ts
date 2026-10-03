import { schema, table, t, SenderError } from 'spacetimedb/server'
import { sha256Hex } from './sha256'

const SIGNUP_ACCESS_CODE = 'MHACKS2026'

const collaborator = table(
  { name: 'collaborator', public: true },
  {
    identity: t.identity().primaryKey(),
    nickname: t.string(),
    color: t.string(),
    online: t.bool(),
    lastSeen: t.u64(),
  }
)

const project = table(
  { name: 'project', public: true },
  {
    id: t.u64().primaryKey().autoInc(),
    name: t.string(),
    prompt: t.string(),
    createdBy: t.identity(),
    createdAt: t.u64(),
  }
)

const agentRun = table(
  { name: 'agent_run', public: true },
  {
    id: t.u64().primaryKey().autoInc(),
    projectId: t.u64().index('btree'),
    agentId: t.string(),
    agentName: t.string(),
    executionOrder: t.u32(),
    startedAt: t.u64(),
    endedAt: t.u64(),
    runtimeMs: t.u64(),
    status: t.string(),
    startedBy: t.identity(),
  }
)

const agentChange = table(
  { name: 'agent_change', public: true },
  {
    id: t.u64().primaryKey().autoInc(),
    projectId: t.u64().index('btree'),
    agentId: t.string(),
    agentName: t.string(),
    changedBy: t.identity(),
    changedAt: t.u64(),
    field: t.string(),
    oldValue: t.string(),
    newValue: t.string(),
  }
)

const studioUser = table(
  { name: 'studio_user', public: true },
  {
    email: t.string().primaryKey(),
    registeredAt: t.u64(),
    lastLoginAt: t.u64(),
  }
)

// Private: never sent to clients.
const credential = table(
  { name: 'credential' },
  {
    email: t.string().primaryKey(),
    salt: t.string(),
    passwordHash: t.string(),
  }
)

const session = table(
  { name: 'session', public: true },
  {
    identity: t.identity().primaryKey(),
    email: t.string(),
  }
)

// Private: one pending verification code per email.
const emailCode = table(
  { name: 'email_code' },
  {
    email: t.string().primaryKey(),
    purpose: t.string(),
    code: t.string(),
    expiresAt: t.u64(),
    attempts: t.u32(),
    lastSentAt: t.u64(),
    requestedBy: t.identity(),
    verified: t.bool(),
  }
)

// Private: server settings such as the email API key. Set by the database owner.
const appConfig = table(
  { name: 'app_config' },
  {
    name: t.string().primaryKey(),
    value: t.string(),
  }
)

const spacetimedb = schema({ collaborator, project, agentRun, agentChange, studioUser, credential, session, emailCode, appConfig })
export default spacetimedb

export const setNickname = spacetimedb.reducer(
  { nickname: t.string(), color: t.string() },
  (ctx, { nickname, color }) => {
    const existing = ctx.db.collaborator.identity.find(ctx.sender)
    const row = { identity: ctx.sender, nickname, color, online: true, lastSeen: BigInt(Date.now()) }
    if (existing) ctx.db.collaborator.identity.update(row)
    else ctx.db.collaborator.insert(row)
  }
)

export const setOnline = spacetimedb.reducer(
  { online: t.bool() },
  (ctx, { online }) => {
    const existing = ctx.db.collaborator.identity.find(ctx.sender)
    if (existing) {
      ctx.db.collaborator.identity.update({ ...existing, online, lastSeen: BigInt(Date.now()) })
    }
  }
)

export const createProject = spacetimedb.reducer(
  { name: t.string(), prompt: t.string() },
  (ctx, { name, prompt }) => {
    ctx.db.project.insert({ id: 0n, name, prompt, createdBy: ctx.sender, createdAt: BigInt(Date.now()) })
  }
)

export const startAgentRun = spacetimedb.reducer(
  { projectId: t.u64(), agentId: t.string(), agentName: t.string(), executionOrder: t.u32() },
  (ctx, { projectId, agentId, agentName, executionOrder }) => {
    ctx.db.agentRun.insert({
      id: 0n,
      projectId,
      agentId,
      agentName,
      executionOrder,
      startedAt: BigInt(Date.now()),
      endedAt: 0n,
      runtimeMs: 0n,
      status: 'running',
      startedBy: ctx.sender,
    })
  }
)

export const finishAgentRun = spacetimedb.reducer(
  { id: t.u64(), status: t.string() },
  (ctx, { id, status }) => {
    const run = ctx.db.agentRun.id.find(id)
    if (run) {
      const now = BigInt(Date.now())
      ctx.db.agentRun.id.update({ ...run, endedAt: now, runtimeMs: now - run.startedAt, status })
    }
  }
)

export const logAgentChange = spacetimedb.reducer(
  {
    projectId: t.u64(),
    agentId: t.string(),
    agentName: t.string(),
    field: t.string(),
    oldValue: t.string(),
    newValue: t.string(),
  },
  (ctx, { projectId, agentId, agentName, field, oldValue, newValue }) => {
    ctx.db.agentChange.insert({
      id: 0n,
      projectId,
      agentId,
      agentName,
      changedBy: ctx.sender,
      changedAt: BigInt(Date.now()),
      field,
      oldValue,
      newValue,
    })
  }
)

const CODE_TTL_MS = 10n * 60n * 1000n
const RESEND_WAIT_MS = 60n * 1000n
const MAX_ATTEMPTS = 5
const EMAIL_RE = /^[^@\s]+@[^@\s]+\.[^@\s]+$/

const cleanEmail = (email: string) => email.trim().toLowerCase()

function hashWithNewSalt(ctx: any, clientHash: string) {
  const salt = Array.from(ctx.random.fill(new Uint8Array(16)), (b: number) => b.toString(16).padStart(2, '0')).join('')
  return { salt, passwordHash: sha256Hex(salt + clientHash) }
}

function startSession(ctx: any, email: string) {
  const row = { identity: ctx.sender, email }
  if (ctx.db.session.identity.find(ctx.sender)) ctx.db.session.identity.update(row)
  else ctx.db.session.insert(row)
}

// Returns the verified code row for this sender, or throws. clientHash is validated here too.
function requireVerified(ctx: any, email: string, purpose: string, clientHash: string) {
  if (!/^[0-9a-f]{64}$/.test(clientHash)) throw new SenderError('Invalid password data')
  const row = ctx.db.emailCode.email.find(email)
  const ok = row && row.purpose === purpose && row.verified
    && row.requestedBy.isEqual(ctx.sender) && row.expiresAt > BigInt(Date.now())
  if (!ok) throw new SenderError('Verification expired. Start again.')
}

// Returns '' on success, otherwise a message to show the user.
export const requestCode = spacetimedb.procedure(
  { email: t.string(), purpose: t.string(), accessCode: t.string() },
  t.string(),
  (ctx, { email, purpose, accessCode }) => {
    const e = cleanEmail(email)
    if (!EMAIL_RE.test(e)) return 'Enter a valid email address'
    if (purpose !== 'signup' && purpose !== 'reset') return 'Invalid request'

    const plan = ctx.withTx(tx => {
      const hasAccount = !!tx.db.credential.email.find(e)
      if (purpose === 'signup') {
        if (accessCode.trim().toUpperCase() !== SIGNUP_ACCESS_CODE) return { error: 'Invalid access code' }
        if (hasAccount) return { error: 'An account with this email already exists. Sign in instead.' }
      }
      const now = BigInt(Date.now())
      const existing = tx.db.emailCode.email.find(e)
      if (existing && now - existing.lastSentAt < RESEND_WAIT_MS) {
        return { error: 'Please wait a minute before requesting another code.' }
      }
      // Do not reveal whether an account exists: a reset for an unknown email succeeds silently.
      if (purpose === 'reset' && !hasAccount) return { error: '', send: false }

      const code = String(tx.random.integerInRange(0, 999999)).padStart(6, '0')
      const row = {
        email: e, purpose, code, expiresAt: now + CODE_TTL_MS, attempts: 0,
        lastSentAt: now, requestedBy: tx.sender, verified: false,
      }
      if (existing) tx.db.emailCode.email.update(row)
      else tx.db.emailCode.insert(row)
      return {
        error: '', send: true, code,
        apiKey: tx.db.appConfig.name.find('resend_api_key')?.value ?? '',
        from: tx.db.appConfig.name.find('email_from')?.value ?? '',
        devLog: tx.db.appConfig.name.find('dev_log_codes')?.value === '1',
      }
    })

    if (plan.error || !plan.send) return plan.error

    if (plan.apiKey && plan.from) {
      const res = ctx.http.fetch('https://api.resend.com/emails', {
        method: 'POST',
        headers: { Authorization: `Bearer ${plan.apiKey}`, 'Content-Type': 'application/json' },
        body: JSON.stringify({
          from: plan.from,
          to: [e],
          subject: `${plan.code} is your AI Video Studio code`,
          text: `Your AI Video Studio verification code is ${plan.code}. It expires in 10 minutes.\n\nIf you did not request this, you can ignore this email.`,
        }),
      })
      if (res.status >= 300) {
        console.error(`email send failed: ${res.status} ${res.text()}`)
        ctx.withTx(tx => { tx.db.emailCode.email.delete(e) })
        return 'Could not send the email. Try again later.'
      }
      return ''
    }
    if (plan.devLog) {
      console.log(`[dev] verification code for ${e}: ${plan.code}`)
      return ''
    }
    ctx.withTx(tx => { tx.db.emailCode.email.delete(e) })
    return 'Email sending is not set up yet.'
  }
)

// A procedure rather than a reducer so failed attempts are counted even when it reports an error.
export const verifyCode = spacetimedb.procedure(
  { email: t.string(), code: t.string() },
  t.string(),
  (ctx, { email, code }) => {
    const e = cleanEmail(email)
    return ctx.withTx(tx => {
      const row = tx.db.emailCode.email.find(e)
      const now = BigInt(Date.now())
      if (!row || !row.requestedBy.isEqual(tx.sender) || row.expiresAt <= now) {
        return 'This code has expired. Request a new one.'
      }
      if (row.attempts >= MAX_ATTEMPTS) return 'Too many attempts. Request a new code.'
      if (row.code !== code.trim()) {
        tx.db.emailCode.email.update({ ...row, attempts: row.attempts + 1 })
        return 'Incorrect code'
      }
      tx.db.emailCode.email.update({ ...row, verified: true, expiresAt: now + CODE_TTL_MS })
      return ''
    })
  }
)

// clientHash is the PBKDF2 hash computed in the app, so the raw password never reaches the database log.
export const signUp = spacetimedb.reducer(
  { email: t.string(), clientHash: t.string() },
  (ctx, { email, clientHash }) => {
    const e = cleanEmail(email)
    requireVerified(ctx, e, 'signup', clientHash)
    if (ctx.db.credential.email.find(e)) throw new SenderError('An account with this email already exists. Sign in instead.')
    const now = BigInt(Date.now())
    ctx.db.credential.insert({ email: e, ...hashWithNewSalt(ctx, clientHash) })
    ctx.db.studioUser.insert({ email: e, registeredAt: now, lastLoginAt: now })
    ctx.db.emailCode.email.delete(e)
    startSession(ctx, e)
  }
)

export const resetPassword = spacetimedb.reducer(
  { email: t.string(), clientHash: t.string() },
  (ctx, { email, clientHash }) => {
    const e = cleanEmail(email)
    requireVerified(ctx, e, 'reset', clientHash)
    if (!ctx.db.credential.email.find(e)) throw new SenderError('Verification expired. Start again.')
    ctx.db.credential.email.update({ email: e, ...hashWithNewSalt(ctx, clientHash) })
    ctx.db.emailCode.email.delete(e)
    // Sign out every device that was using the old password.
    for (const s of Array.from(ctx.db.session.iter())) {
      if (s.email === e) ctx.db.session.identity.delete(s.identity)
    }
    startSession(ctx, e)
  }
)

export const signIn = spacetimedb.reducer(
  { email: t.string(), clientHash: t.string() },
  (ctx, { email, clientHash }) => {
    const e = cleanEmail(email)
    const cred = ctx.db.credential.email.find(e)
    if (!cred || sha256Hex(cred.salt + clientHash) !== cred.passwordHash) {
      throw new SenderError('Incorrect email or password')
    }
    const user = ctx.db.studioUser.email.find(e)
    if (user) ctx.db.studioUser.email.update({ ...user, lastLoginAt: BigInt(Date.now()) })
    startSession(ctx, e)
  }
)

export const signOut = spacetimedb.reducer({}, (ctx) => {
  ctx.db.session.identity.delete(ctx.sender)
  const c = ctx.db.collaborator.identity.find(ctx.sender)
  if (c) ctx.db.collaborator.identity.update({ ...c, online: false, lastSeen: BigInt(Date.now()) })
})
