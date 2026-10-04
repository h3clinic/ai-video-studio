const test=require('node:test')
const assert=require('node:assert/strict')
const fs=require('node:fs')
const path=require('node:path')
const vm=require('node:vm')
const ts=require('typescript')
const source=fs.readFileSync(path.join(__dirname,'../src/App.tsx'),'utf8')
const compiled=ts.transpileModule(source,{compilerOptions:{module:ts.ModuleKind.CommonJS,target:ts.ScriptTarget.ES2020,jsx:ts.JsxEmit.ReactJSX}}).outputText
const exported={}
vm.runInNewContext(compiled,{exports:exported,require:()=>({})},{timeout:1000})
const {partContentState}=exported

test('three actual audio outputs and review briefs do not require Gaussian bindings',()=>{
  const ids=['audio_environment','audio_donkey','audio_handling']
  const workers=ids.map((partId,i)=>({id:`worker${i}`,partId,roleId:'sound-agents',status:'completed',execution:'elevenlabs_sound',output:`sound${i}.mp3`,mediaId:`media${i}`}))
  workers.push({id:'reviewer',partId:'verification_invariants',roleId:'verifiers',status:'completed',execution:'specialist_brief'})
  const jobs=[{id:'team',createdAt:100,workers}]
  const media=ids.map((_,i)=>({id:`audio_media${i}`,kind:'audio'}))
  for(const id of ids){
    const state=partContentState({id,ownerRoleId:'sound-agents',bindingVerified:false},jobs,media)
    assert.equal(state.requiresGeometryBinding,false)
    assert.match(state.label,/Audio ready/)
    assert.doesNotMatch(state.label,/blocked|Gaussian IDs/)
  }
  const review=partContentState({id:'verification_invariants',ownerRoleId:'verifiers'},jobs,media)
  assert.equal(review.kind,'review');assert.equal(review.requiresGeometryBinding,false)
  assert.match(review.label,/Review brief ready/);assert.match(review.label,/quality unverified/)
})

test('missing and failed sound outputs never become ready merely from a part name',()=>{
  const part={id:'audio_environment',ownerRoleId:'sound-agents'}
  assert.doesNotMatch(partContentState(part).label,/ready/)
  const jobs=[{id:'old',createdAt:100,workers:[{id:'sound',partId:part.id,roleId:'sound-agents',status:'completed',execution:'elevenlabs_sound',output:'old.mp3',mediaId:'old'}]},
    {id:'new',createdAt:200,workers:[{id:'sound',partId:part.id,roleId:'sound-agents',status:'failed'}]}]
  assert.match(partContentState(part,jobs,[{id:'audio_old',kind:'audio'}]).label,/failed/)
  assert.match(partContentState(part,[jobs[0]],[]).label,/Preview unavailable/)
  assert.equal(partContentState({id:'audio_whole_apple',ownerRoleId:'vector-agents'}).requiresGeometryBinding,true)
})

test('workflow briefs are distinct from unbound visual parts and invalid bindings',()=>{
  const jobs=[{id:'team',workers:[{partId:'scene_narrative',roleId:'idea-model',status:'completed',execution:'specialist_brief'}]}]
  const brief=partContentState({id:'scene_narrative'},jobs)
  assert.equal(brief.kind,'workflow');assert.match(brief.label,/Brief ready/)
  for(const binding of [false,'true',true]){
    const visual=partContentState({id:'whole_apple',ownerRoleId:'vector-agents',bindingVerified:binding})
    assert.equal(visual.requiresGeometryBinding,true)
    assert.match(visual.label,/Unbound/)
  }
  const bound=partContentState({id:'whole_apple',ownerRoleId:'verifiers',bindingVerified:true,gaussianCount:20000})
  assert.equal(bound.kind,'visual');assert.match(bound.label,/20000 Gaussian IDs/)
  assert.match(partContentState({id:'donkey',protected:true,ownerRoleId:'sound-agents'}).label,/Protected scene part/)
})
