const test=require('node:test')
const assert=require('node:assert/strict')
const fs=require('node:fs')
const path=require('node:path')
const vm=require('node:vm')
const ts=require('typescript')
function exportsOf(name){
  const source=fs.readFileSync(path.join(__dirname,'../src',name),'utf8')
  const compiled=ts.transpileModule(source,{compilerOptions:{module:ts.ModuleKind.CommonJS,target:ts.ScriptTarget.ES2020,jsx:ts.JsxEmit.ReactJSX}}).outputText
  const exports={};vm.runInNewContext(compiled,{exports,require:()=>({})},{timeout:1000});return exports
}
const {spotlightActivity}=exportsOf('AgentSpotlight.tsx')
test('spotlight follows running job before queued and never treats completed/blocked as active',()=>{
  const jobs=[{id:'q',status:'queued'},{id:'r',status:'running'},{id:'b',status:'blocked'},{id:'c',status:'completed'}]
  assert.equal(spotlightActivity(jobs,false).job.id,'r')
  assert.equal(spotlightActivity(jobs,false).running.length,1)
  assert.equal(spotlightActivity(jobs.slice(2),false).status,'Ready for a task')
  assert.equal(spotlightActivity([jobs[0]],false).status,'Queued')
  assert.equal(spotlightActivity(jobs,true).status,'Status unavailable')
})
test('owner sound key form does not persist credentials to browser/history and never generates on save',()=>{
  const source=fs.readFileSync(path.join(__dirname,'../src/ElevenLabsSettings.tsx'),'utf8')
  assert.match(source,/type="password"/)
  assert.match(source,/field\.current\.value=''/)
  assert.doesNotMatch(source,/localStorage|sessionStorage|console\.|\/api\/jobs/)
  assert.match(source,/finally\{key='';setBusy\(false\)\}/)
})
test('desktop has original agent art, timeline and audio player without weakening credential boundary',()=>{
  const app=fs.readFileSync(path.join(__dirname,'../src/App.tsx'),'utf8')
  assert.match(app,/<AgentSpotlight/)
  assert.match(app,/<AgentTimeline/)
  assert.match(app,/<audio controls preload="metadata"/)
  assert.match(app,/provider:'elevenlabs',key/)
  const csp=fs.readFileSync(path.join(__dirname,'../index.html'),'utf8')
  assert.match(csp,/ws:\/\/127\.0\.0\.1:3000/)
  assert.doesNotMatch(csp,/connect-src[^;]*\*/)
})
