const test=require('node:test')
const assert=require('node:assert/strict')
const fs=require('node:fs')
const path=require('node:path')
const {validateRequest}=require('./security')
test('team request remains bounded and cannot choose tools, code or URLs',()=>{
  const req={kind:'agent_swarm',projectId:'demo',prompt:'Prepare the source scene'}
  assert.equal(validateRequest('/api/jobs',req).method,'POST')
  for(const extra of [{maxCalls:10000},{command:'run'},{url:'https://example.com'},{projectId:''},{prompt:''}])assert.throws(()=>validateRequest('/api/jobs',{...req,...extra}))
})
test('recording controls and IPC are absent from production app',()=>{
  for(const file of ['src/App.tsx','electron/preload.js','electron/main.js']){
    const source=fs.readFileSync(path.join(__dirname,'..',file),'utf8')
    assert.doesNotMatch(source,/DemoRecorder|studio:record|Record app|recorder\./)
  }
})
test('concise app retains truthful output label and real task details',()=>{
  const source=fs.readFileSync(path.join(__dirname,'../src/LiveSwarm.tsx'),'utf8')
  assert.match(source,/Video unchanged/)
  assert.match(source,/chosen\.seconds/)
  assert.match(source,/chosen\.mediaId/)
  assert.match(source,/Team brief/)
})
