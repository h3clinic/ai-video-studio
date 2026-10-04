const test=require('node:test')
const assert=require('node:assert/strict')
const fs=require('node:fs')
const vm=require('node:vm')
const ts=require('typescript')
const source=fs.readFileSync(require('node:path').join(__dirname,'../src/AgentTimeline.tsx'),'utf8')
const code=ts.transpileModule(source,{compilerOptions:{module:ts.ModuleKind.CommonJS,target:ts.ScriptTarget.ES2020,jsx:ts.JsxEmit.ReactJSX}}).outputText
const out={};vm.runInNewContext(code,{exports:out,require:()=>({})})
test('latest timing scope includes unsuccessful current work and leaves history intact',()=>{
  const jobs=[{id:'mix',status:'completed'},{id:'team',status:'blocked'},{id:'worker',parentRunId:'team',status:'failed'},{id:'old',status:'failed'}]
  assert.equal(out.timingScope(jobs,['mix','team'],false).map(j=>j.id).join(','),'mix,team,worker')
  assert.equal(out.timingScope(jobs,['mix','team'],true),jobs)
  assert.equal(out.timingScope(jobs,[],false),jobs)
  assert.equal(jobs[3].status,'failed')
})
