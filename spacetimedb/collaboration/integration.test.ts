// Real local server/SDK integration. No mocked ACL or synthetic success report.
import assert from 'node:assert/strict'
import { randomUUID } from 'node:crypto'
import { DbConnection, tables } from '../../src/module_bindings/collaboration'

const clients:DbConnection[]=[]
const checks:string[]=[]
const projectId=`test-${randomUUID()}`
async function until(fn:()=>boolean, label:string) {
  const deadline=Date.now()+5000
  while(!fn()) { if(Date.now()>deadline) throw new Error(`Timed out: ${label}`); await new Promise(r=>setTimeout(r,20)) }
}
async function connect():Promise<DbConnection> {
  return new Promise((resolve,reject)=>{
    const timeout=setTimeout(()=>reject(new Error('Local server connection timeout')),8000)
    const c=DbConnection.builder().withUri('ws://127.0.0.1:3000').withDatabaseName('ai-video-studio-collaboration').withToken(undefined)
      .onConnect(connection=>connection.subscriptionBuilder().onApplied(()=>{clearTimeout(timeout);resolve(connection)}).onError(ctx=>{clearTimeout(timeout);reject(new Error('Subscription failed'))})
        .subscribe([tables.myProjects,tables.myMembers,tables.myDrafts,tables.myEdits,tables.myTimings,tables.myActivity]))
      .onConnectError((_ctx,error)=>{clearTimeout(timeout);reject(error)}).build()
    clients.push(c)
  })
}
try {
  const owner=await connect(),editor=await connect(),viewer=await connect(),outsider=await connect()
  assert(!owner.identity!.isEqual(editor.identity!));checks.push('independent authenticated identities')
  const createdBefore=Date.now()
  await owner.reducers.createWorkspace({projectId,title:'Integration test (not a generated artifact)',displayName:'Test owner'})
  await until(()=>Array.from(owner.db.myProjects.iter()).some(p=>p.id===projectId),'owner project')
  assert.equal(Array.from(outsider.db.myProjects.iter()).length,0);checks.push('private workspace hidden from outsider')
  const initial={projectId,agentId:'vector-agents',partId:'apple_body',field:'instruction',value:'Increase surface detail; test fixture',expectedRevision:0n,operationId:randomUUID()}
  await assert.rejects(editor.reducers.editPart(initial),/membership/i);checks.push('non-member write rejected')
  await owner.reducers.setMember({projectId,identity:editor.identity!,role:'editor',displayName:'Test editor'})
  await until(()=>Array.from(editor.db.myProjects.iter()).some(p=>p.id===projectId),'editor grant')
  await editor.reducers.editPart(initial)
  await until(()=>Array.from(owner.db.myEdits.iter()).length===1,'first shared edit')
  const first=Array.from(owner.db.myEdits.iter())[0]
  assert(first.editedBy.isEqual(editor.identity!));assert(Number(first.editedAt)>=createdBefore);checks.push('server timestamp and exact author replicated')
  await editor.reducers.editPart(initial)
  assert.equal(Array.from(owner.db.myEdits.iter()).length,1);checks.push('idempotent edit retry')
  await assert.rejects(owner.reducers.editPart({...initial,operationId:randomUUID(),value:'Stale overwrite'}),/Revision conflict/i);checks.push('optimistic revision conflict rejected')
  await owner.reducers.editPart({...initial,operationId:randomUUID(),expectedRevision:1n,value:'Merge detail with contact constraint'})
  await until(()=>Array.from(editor.db.myEdits.iter()).length===2,'history append')
  assert.equal(Array.from(editor.db.myDrafts.iter())[0].revision,2n);checks.push('two authors append history without overwriting prior version')
  await owner.reducers.setMember({projectId,identity:viewer.identity!,role:'viewer',displayName:'Test viewer'})
  await until(()=>Array.from(viewer.db.myDrafts.iter()).length===1,'viewer subscription')
  await assert.rejects(viewer.reducers.editPart({...initial,operationId:randomUUID(),expectedRevision:2n}),/membership/i);checks.push('viewer reads but cannot edit')
  await assert.rejects(editor.reducers.setMember({projectId,identity:outsider.identity!,role:'editor',displayName:'Intruder'}),/owner/i);checks.push('only owner may grant access')
  await assert.rejects(owner.reducers.editPart({...initial,expectedRevision:2n,operationId:randomUUID(),value:'sk_test_12345678901234567890123456789'}),/secret|credential/i);checks.push('obvious credentials blocked from history')
  await owner.reducers.recordTiming({projectId,jobId:'test-job',agentId:'vector-agents',partId:'apple_body',status:'completed',queuedAt:'2026-10-04T01:00:00.000Z',startedAt:'2026-10-04T01:00:00.010Z',finishedAt:'2026-10-04T01:00:00.250Z',runtimeMs:240n,hasRuntime:true,sequence:3})
  await until(()=>Array.from(editor.db.myTimings.iter()).length===1,'timing replication')
  const timing=Array.from(editor.db.myTimings.iter())[0]
  assert.equal(timing.sourceRuntimeMs,240n);assert(Number(timing.observedAt)>=createdBefore);checks.push('reported source timing separate from DB receipt clock')
  await assert.rejects(owner.reducers.recordTiming({projectId,jobId:'test-job',agentId:'vector-agents',partId:'apple_body',status:'running',queuedAt:'',startedAt:'',finishedAt:'',runtimeMs:0n,hasRuntime:false,sequence:4}),/immutable/i);checks.push('terminal timing cannot regress')
  await new Promise<void>((resolve,reject)=>{
    const timer=setTimeout(()=>reject(new Error('Private-table subscription did not reject')),3000)
    outsider.subscriptionBuilder().onApplied(()=>{clearTimeout(timer);reject(new Error('PRIVATE TABLE LEAK'))}).onError(()=>{clearTimeout(timer);resolve()}).subscribe('SELECT * FROM part_edit')
  });checks.push('raw private table subscription denied')
  await owner.reducers.setMember({projectId,identity:editor.identity!,role:'removed',displayName:'Test editor'})
  await until(()=>Array.from(editor.db.myEdits.iter()).length===0 && Array.from(editor.db.myProjects.iter()).length===0,'revocation clears scoped subscription')
  await assert.rejects(editor.reducers.editPart({...initial,expectedRevision:2n,operationId:randomUUID()}),/membership/i);checks.push('revocation removes subscription and write rights')
  console.log(JSON.stringify({status:'passed',database:'local-only',projectId,checks:checks.length,assertions:checks},null,2))
} finally {clients.forEach(c=>c.disconnect())}
