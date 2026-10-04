// Pure UI state tests: no Electron launch, browser control, provider call or job.
const test = require('node:test')
const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')
const vm = require('node:vm')
const ts = require('typescript')

const source = fs.readFileSync(path.join(__dirname, '../src/App.tsx'), 'utf8')
const compiled = ts.transpileModule(source, {compilerOptions:{module:ts.ModuleKind.CommonJS,target:ts.ScriptTarget.ES2020,jsx:ts.JsxEmit.ReactJSX}}).outputText
const exportsForTest = {}
// App module initialization is inert; all UI imports are deliberately stubbed.
vm.runInNewContext(compiled, {exports:exportsForTest, require:()=>({})}, {timeout:1000})
const {appleReplacementReadiness, remoteFailure} = exportsForTest

test('replacement button fails closed without backend readiness', () => {
  for (const value of [undefined,null,{},[],{ready:'true',revision:'repair-v2',missing:[]},{ready:true,missing:[]},{ready:true,revision:'repair-v2'}]) {
    assert.equal(appleReplacementReadiness(value).ready, false)
  }
})

test('known rejected revision cannot be rerun even if incorrectly marked ready', () => {
  assert.equal(appleReplacementReadiness({ready:true,revision:'semantic-static-v1-rejected',missing:[]}).ready, false)
})

test('missing prerequisites and malformed metadata block only scene replacement', () => {
  for (const missing of [['Material binding'],[42],null]) {
    assert.equal(appleReplacementReadiness({ready:true,revision:'repair-v2',missing}).ready, false)
  }
  const state=appleReplacementReadiness({ready:false,revision:'repair-v2',missing:['Mouth-contact track'],reason:'Missing measured contact'})
  assert.equal(state.reason,'Missing measured contact')
  assert.equal(state.missing[0],'Mouth-contact track')
})

test('versioned ready repair without missing prerequisites becomes available', () => {
  assert.equal(appleReplacementReadiness({ready:true,revision:'contact-material-v2',missing:[]}).ready, true)
})

test('remote auth status is not confused with a configured key or successful run', () => {
  assert.equal(remoteFailure({status:'RUNNING'}),null)
  assert.match(remoteFailure({status:'auth_failed'}),/rejected the saved owner authorization/)
  assert.match(remoteFailure({status:'permission_denied'}),/cannot access/)
  assert.equal(remoteFailure({status:'auth_failed',error:'RunPod HTTP 401: owner credential rejected.'}),'RunPod HTTP 401: owner credential rejected.')
})
