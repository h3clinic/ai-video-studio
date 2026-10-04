import { schema, table, t, SenderError } from 'spacetimedb/server'

// Separate private collaboration database. The original demo's public tables
// and data are not migrated, cleared, subscribed to, or republished here.
const project = table({ name: 'workspace_project' }, {
  id: t.string().primaryKey(), title: t.string(), owner: t.identity(), createdAt: t.u64(),
})
const member = table({ name: 'workspace_member' }, {
  key: t.string().primaryKey(), projectId: t.string().index('btree'), identity: t.identity(),
  role: t.string(), displayName: t.string(), joinedAt: t.u64(),
})
const partDraft = table({ name: 'part_draft' }, {
  key: t.string().primaryKey(), projectId: t.string().index('btree'), agentId: t.string(), partId: t.string(),
  field: t.string(), value: t.string(), revision: t.u64(), updatedAt: t.u64(), updatedBy: t.identity(),
})
const edit = table({ name: 'part_edit' }, {
  id: t.u64().primaryKey().autoInc(), projectId: t.string().index('btree'), agentId: t.string(), partId: t.string(),
  field: t.string(), oldValue: t.string(), newValue: t.string(), revision: t.u64(),
  editedAt: t.u64(), editedBy: t.identity(), operationId: t.string().unique(),
})
const timing = table({ name: 'agent_timing' }, {
  key: t.string().primaryKey(), projectId: t.string().index('btree'), jobId: t.string(), agentId: t.string(),
  partId: t.string(), sourceStatus: t.string(), sourceQueuedAt: t.string(), sourceStartedAt: t.string(),
  sourceFinishedAt: t.string(), sourceRuntimeMs: t.u64(), sourceHasRuntime: t.bool(),
  sourceSequence: t.u32(), observedAt: t.u64(), reportedBy: t.identity(),
})
const activity = table({ name: 'workspace_activity' }, {
  id: t.u64().primaryKey().autoInc(), projectId: t.string().index('btree'), kind: t.string(),
  target: t.string(), detail: t.string(), at: t.u64(), actor: t.identity(),
})
const db = schema({ project, member, partDraft, edit, timing, activity })
export default db

const ROLE_IDS = new Set(['idea-model','video-controller','crawler','decision','vision-sensor','section-leads','field-agents','surroundings-list','kind-checkers','mini-object-agents','task-assigner','vector-agents','sound-agents','verifiers'])
const FIELDS = new Set(['instruction','appearance','motion','review'])
const TERMINAL = new Set(['completed','failed','blocked','interrupted','cancelled','rejected'])
const memberKey = (projectId: string, identity: {toHexString(): string}) => `${projectId}:${identity.toHexString()}`
const now = (ctx: any): bigint => ctx.timestamp.microsSinceUnixEpoch / 1000n
function bounded(value: string, limit: number, label: string) {
  if (!value.trim() || value.length > limit || /[\u0000-\u0008\u000b\u000c\u000e-\u001f]/.test(value)) throw new SenderError(`Invalid ${label}`)
  // Credential values have no place in a replicated edit log. This is a guard,
  // not a claim that arbitrary prose can be exhaustively secret-detected.
  if (/(?:AIza[\w-]{25,}|sk[-_][A-Za-z0-9_-]{16,}|rpa_[A-Za-z0-9]{16,}|-----BEGIN.*PRIVATE KEY)/i.test(value)) throw new SenderError('Use owner credential settings, not shared history, for secrets')
  return value.trim()
}
function projectId(value: string) {
  if (!/^[A-Za-z0-9_-]{1,80}$/.test(value)) throw new SenderError('Invalid project ID')
}
function requireMember(ctx: any, id: string, write = false) {
  const row = ctx.db.member.key.find(memberKey(id, ctx.sender))
  if (!row || (write && !['owner','editor'].includes(row.role))) throw new SenderError('Project membership does not allow this action')
  return row
}
function requireOwner(ctx: any, id: string) {
  const row = ctx.db.project.id.find(id)
  if (!row || !row.owner.isEqual(ctx.sender)) throw new SenderError('Only the project owner may change members')
  return row
}
function visibleProjects(ctx: any): Set<string> {
  return new Set(Array.from(ctx.db.member.iter()).filter((m: any) => m.identity.isEqual(ctx.sender)).map((m: any) => m.projectId))
}
function log(ctx: any, id: string, kind: string, target: string, detail: string) {
  ctx.db.activity.insert({id:0n,projectId:id,kind,target,detail,at:now(ctx),actor:ctx.sender})
}

export const myProjects = db.view({name:'my_projects',public:true}, t.array(project.rowType), ctx => {
  const ids = visibleProjects(ctx); return Array.from(ctx.db.project.iter()).filter(p => ids.has(p.id))
})
export const myMembers = db.view({name:'my_members',public:true}, t.array(member.rowType), ctx => {
  const ids = visibleProjects(ctx); return Array.from(ctx.db.member.iter()).filter(p => ids.has(p.projectId))
})
export const myDrafts = db.view({name:'my_drafts',public:true}, t.array(partDraft.rowType), ctx => {
  const ids = visibleProjects(ctx); return Array.from(ctx.db.partDraft.iter()).filter(p => ids.has(p.projectId))
})
export const myEdits = db.view({name:'my_edits',public:true}, t.array(edit.rowType), ctx => {
  const ids = visibleProjects(ctx); return Array.from(ctx.db.edit.iter()).filter(p => ids.has(p.projectId))
})
export const myTimings = db.view({name:'my_timings',public:true}, t.array(timing.rowType), ctx => {
  const ids = visibleProjects(ctx); return Array.from(ctx.db.timing.iter()).filter(p => ids.has(p.projectId))
})
export const myActivity = db.view({name:'my_activity',public:true}, t.array(activity.rowType), ctx => {
  const ids = visibleProjects(ctx); return Array.from(ctx.db.activity.iter()).filter(p => ids.has(p.projectId))
})

export const createWorkspace = db.reducer({projectId:t.string(),title:t.string(),displayName:t.string()}, (ctx,args) => {
  projectId(args.projectId)
  if (ctx.db.project.id.find(args.projectId)) throw new SenderError('Project already exists; ask its owner for access')
  ctx.db.project.insert({id:args.projectId,title:bounded(args.title,200,'title'),owner:ctx.sender,createdAt:now(ctx)})
  ctx.db.member.insert({key:memberKey(args.projectId,ctx.sender),projectId:args.projectId,identity:ctx.sender,role:'owner',displayName:bounded(args.displayName,80,'name'),joinedAt:now(ctx)})
  log(ctx,args.projectId,'project_created',args.projectId,'Private shared workspace created')
})
export const setMember = db.reducer({projectId:t.string(),identity:t.identity(),role:t.string(),displayName:t.string()}, (ctx,args) => {
  const p = requireOwner(ctx,args.projectId)
  if (p.owner.isEqual(args.identity)) throw new SenderError('The owner membership cannot be removed or demoted')
  if (!['editor','viewer','removed'].includes(args.role)) throw new SenderError('Choose editor, viewer or removed')
  const key = memberKey(args.projectId,args.identity)
  const existing = ctx.db.member.key.find(key)
  if (args.role === 'removed') {
    if (existing) ctx.db.member.key.delete(key)
  } else {
    const row = {key,projectId:args.projectId,identity:args.identity,role:args.role,displayName:bounded(args.displayName,80,'name'),joinedAt:existing?.joinedAt ?? now(ctx)}
    if(existing) ctx.db.member.key.update(row); else ctx.db.member.insert(row)
  }
  log(ctx,args.projectId,'membership',args.identity.toHexString(),args.role)
})
export const editPart = db.reducer({projectId:t.string(),agentId:t.string(),partId:t.string(),field:t.string(),value:t.string(),expectedRevision:t.u64(),operationId:t.string()}, (ctx,args) => {
  requireMember(ctx,args.projectId,true)
  if(!ROLE_IDS.has(args.agentId) || !/^[a-z0-9_-]{1,80}$/.test(args.partId) || !FIELDS.has(args.field)) throw new SenderError('Unknown agent, part or editable field')
  const value = bounded(args.value,2000,'edit')
  if(!/^[a-f0-9-]{32,36}$/.test(args.operationId)) throw new SenderError('Invalid operation ID')
  const previousOp = ctx.db.edit.operationId.find(args.operationId)
  if(previousOp) {
    if (previousOp.editedBy.isEqual(ctx.sender) && previousOp.projectId===args.projectId && previousOp.agentId===args.agentId && previousOp.partId===args.partId && previousOp.field===args.field && previousOp.newValue===value) return
    throw new SenderError('Operation ID already used')
  }
  const key = `${args.projectId}:${args.agentId}:${args.partId}:${args.field}`
  const current = ctx.db.partDraft.key.find(key)
  if ((current?.revision ?? 0n) !== args.expectedRevision) throw new SenderError('Revision conflict: reload the latest draft before saving')
  const revision = (current?.revision ?? 0n)+1n
  const row = {key,projectId:args.projectId,agentId:args.agentId,partId:args.partId,field:args.field,value,revision,updatedAt:now(ctx),updatedBy:ctx.sender}
  if(current) ctx.db.partDraft.key.update(row); else ctx.db.partDraft.insert(row)
  ctx.db.edit.insert({id:0n,projectId:args.projectId,agentId:args.agentId,partId:args.partId,field:args.field,oldValue:current?.value ?? '',newValue:value,revision,editedAt:now(ctx),editedBy:ctx.sender,operationId:args.operationId})
})
// Job measurements are imported explicitly, not recomputed from browser clocks.
// observedAt is the DB clock; source fields remain labelled client-reported.
export const recordTiming = db.reducer({projectId:t.string(),jobId:t.string(),agentId:t.string(),partId:t.string(),status:t.string(),queuedAt:t.string(),startedAt:t.string(),finishedAt:t.string(),runtimeMs:t.u64(),hasRuntime:t.bool(),sequence:t.u32()}, (ctx,args) => {
  requireMember(ctx,args.projectId,true)
  if(!/^[A-Za-z0-9_-]{1,100}$/.test(args.jobId) || !ROLE_IDS.has(args.agentId) || !/^[a-z0-9_-]{0,80}$/.test(args.partId) || !['queued','running',...TERMINAL].includes(args.status)) throw new SenderError('Invalid timing event')
  for(const value of [args.queuedAt,args.startedAt,args.finishedAt]) {
    if(value && (!/^\d{4}-\d\d-\d\dT.*Z$/.test(value) || !Number.isFinite(Date.parse(value)) || value.length>32)) throw new SenderError('Invalid source timestamp')
  }
  if(args.runtimeMs>604800000n) throw new SenderError('Runtime outside measurement bounds')
  if(!args.hasRuntime && args.runtimeMs!==0n) throw new SenderError('Absent runtime must be zero')
  const key=`${args.projectId}:${args.jobId}`
  const current=ctx.db.timing.key.find(key)
  if(current) {
    if(!current.reportedBy.isEqual(ctx.sender)) throw new SenderError('Only the original reporter may update a timing')
    if(args.sequence<=current.sourceSequence) return
    if(TERMINAL.has(current.sourceStatus)) throw new SenderError('Completed timing records are immutable')
  }
  const row={key,projectId:args.projectId,jobId:args.jobId,agentId:args.agentId,partId:args.partId,sourceStatus:args.status,sourceQueuedAt:args.queuedAt,sourceStartedAt:args.startedAt,sourceFinishedAt:args.finishedAt,sourceRuntimeMs:args.runtimeMs,sourceHasRuntime:args.hasRuntime,sourceSequence:args.sequence,observedAt:now(ctx),reportedBy:ctx.sender}
  if(current) ctx.db.timing.key.update(row); else ctx.db.timing.insert(row)
  log(ctx,args.projectId,'timing_observed',args.jobId,`${args.agentId}: ${args.status}; source-reported measurement`)
})
