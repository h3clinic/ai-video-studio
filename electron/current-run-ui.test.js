// Current-run/history rendering only. No app launch, provider request or job.
const test=require('node:test')
const assert=require('node:assert/strict')
const fs=require('node:fs')
const path=require('node:path')
const vm=require('node:vm')
const ts=require('typescript')
const React=require('react')
const {renderToStaticMarkup}=require('react-dom/server')

const source=fs.readFileSync(path.join(__dirname,'../src/App.tsx'),'utf8')
const compiled=ts.transpileModule(source,{compilerOptions:{module:ts.ModuleKind.CommonJS,target:ts.ScriptTarget.ES2020,jsx:ts.JsxEmit.ReactJSX}}).outputText
const exported={}
vm.runInNewContext(compiled,{exports:exported,require:name=>name==='react/jsx-runtime'?require(name):name==='./LiveSwarm'?{default:()=>null}:{}},{timeout:1000})
const {latestRuns,runProblem,runTimestamp,RunHistory}=exported

test('newest submission stays current even if a past failure finishes later',()=>{
  const jobs=[
    {id:'latest',createdAt:'2026-10-04T13:00:00Z',kind:'agent_swarm',status:'completed'},
    {id:'failed',createdAt:'2026-10-03T12:00:00Z',finishedAt:'2026-10-04T14:00:00Z',status:'failed'},
    {id:'blocked',createdAt:'2026-10-02T12:00:00Z',status:'blocked'},
  ]
  const before=JSON.stringify(jobs)
  const result=latestRuns(jobs)
  assert.equal(result.current,jobs[0])
  assert.equal(result.history[0],jobs[1])
  assert.equal(result.history[1],jobs[2])
  assert.equal(JSON.stringify(jobs),before)
})

test('a new blocked run is current even when an earlier run completed',()=>{
  const result=latestRuns([
    {id:'old-success',createdAt:100,status:'completed'},
    {id:'new-blocked',createdAt:200,status:'blocked'},
  ])
  assert.equal(result.current.id,'new-blocked')
  assert.equal(result.current.status,'blocked')
  assert.equal(result.history[0].status,'completed')
  assert.equal(latestRuns([]).current,undefined)
  assert.equal(latestRuns([{id:'a'},{id:'b'}]).current.id,'b')
  assert.equal(runTimestamp('invalid'),'Time not recorded')
})

test('history is initially collapsed and retains every timestamp, status and original error',()=>{
  const jobs=[
    {id:'blocked',kind:'apple_experiment',status:'blocked',createdAt:'2026-10-03T12:00:00Z',error:'Material/contact pipeline unavailable',review:'Not accepted'},
    {id:'failed',kind:'agent_task',status:'failed',createdAt:'2026-10-02T12:00:00Z',error:'Provider HTTP 401\nOriginal error context',reply:'Original task response'},
  ]
  const html=renderToStaticMarkup(React.createElement(RunHistory,{jobs,artwork:()=>undefined,onEnvironment:()=>{},onAssign:()=>{}}))
  assert.match(html,/<details data-run-history="true"/)
  assert.doesNotMatch(html,/<details[^>]*\sopen(?:[=> ])/)
  assert.match(html,/History · 2 earlier runs/)
  for(const job of jobs){
    assert.ok(html.includes(`data-history-run="${job.id}"`))
    assert.ok(html.includes(job.status))
    assert.ok(html.includes(runTimestamp(job.createdAt)))
    assert.ok(html.includes(job.error))
  }
  assert.match(html,/Original task response/)
  assert.match(html,/Not accepted/)
  assert.match(html,/Open Environment/)
})

test('failure guidance is concise and does not turn completed briefs into a geometry success',()=>{
  const failed={status:'failed',kind:'agent_task',error:`Provider HTTP 401 ${'x'.repeat(200)}\nDiagnostic detail`}
  const problem=runProblem(failed)
  assert.ok(problem.summary.length<=150)
  assert.match(problem.action,/credentials in Environment/)
  assert.equal(failed.status,'failed')
  assert.equal(runProblem({status:'completed',kind:'agent_swarm',stage:'Briefs prepared; geometry not executed'}),null)
  assert.match(runProblem({status:'blocked',kind:'apple_experiment'}).action,/missing scene-replacement prerequisites/)
})
