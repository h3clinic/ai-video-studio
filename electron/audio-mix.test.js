const test=require('node:test')
const assert=require('node:assert/strict')
const fs=require('node:fs')
const path=require('node:path')
const {validateRequest}=require('./security')
test('sound export uses fixed job IDs and finite bounded gains',()=>{
  const request={kind:'sound_mix',projectId:'default',sourceJobId:'a'.repeat(32),gains:[.25,.55,.55]}
  assert.equal(validateRequest('/api/jobs',request).method,'POST')
  for(const patch of [{sourceJobId:'../../file'},{gains:[1,1]},{gains:[1,1,NaN]},{gains:[true,1,1]},{gains:[2,1,1]},{command:'whoami'}])assert.throws(()=>validateRequest('/api/jobs',{...request,...patch}))
})
test('sound mixer exposes real tracks and no recording controls',()=>{
  const source=fs.readFileSync(path.join(__dirname,'../src/SceneAudioMixer.tsx'),'utf8')
  assert.match(source,/onPlay/);assert.match(source,/onPause/);assert.match(source,/onSeeking/)
  assert.match(source,/Export with sound/);assert.match(source,/visual|Original video/)
  assert.doesNotMatch(source,/getDisplayMedia|MediaRecorder|api.key/i)
})
