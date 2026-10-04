const test = require('node:test')
const assert = require('node:assert/strict')
const path = require('path')
const fs = require('fs')
const { validateRequest, trustedSender, assetPath, MEDIA, backendErrorMessage, readBackendJson } = require('./security')

test('only fixed endpoint and method combinations are accepted', () => {
  assert.equal(validateRequest('/api/state').method, 'GET')
  for (const route of ['https://evil.test', '/api/state?token=x', '/api/media/reference', '/api/../credentials', '/api/runpod/start']) assert.throws(() => validateRequest(route))
  assert.throws(() => validateRequest('/api/state', {anything:true}))
  assert.throws(() => validateRequest('/api/jobs', {kind:'shell', command:'hello'}))
  assert.equal(validateRequest('/api/jobs', {kind:'gemini_reference'}).method, 'POST')
  assert.equal(validateRequest('/api/jobs', {kind:'remote_preflight'}).method, 'POST')
})
test('credentials have no readback request and bounded strict shape', () => {
  assert.throws(() => validateRequest('/api/credentials'))
  assert.throws(() => validateRequest('/api/credentials', {provider:'other',key:'secret'}))
  assert.throws(() => validateRequest('/api/credentials', {provider:'gemini',key:'secret',returnKey:true}))
  assert.throws(() => validateRequest('/api/credentials', {provider:'gemini',key:'x'.repeat(5000)}))
  assert.equal(validateRequest('/api/credentials', {provider:'gemini',key:'fixture'}).method,'POST')
  assert.equal(validateRequest('/api/credentials', {provider:'elevenlabs',key:'fixture'}).method,'POST')
})
test('project-scoped jobs have bounded prompts and exact fields', () => {
  for (const body of [
    {kind:'plan',projectId:'abc_1',prompt:'A donkey eats an apple'},
    {kind:'gemini_reference',projectId:'abc_1',prompt:'Apple skin'},
    {kind:'apple_experiment',projectId:'abc_1'},
  ]) assert.equal(validateRequest('/api/jobs',body).method,'POST')
  for (const body of [
    {kind:'plan',prompt:'Missing project'}, {kind:'plan',projectId:'abc',prompt:' '},
    {kind:'plan',projectId:'../abc',prompt:'test'}, {kind:'plan',projectId:'abc',prompt:'x'.repeat(4001)},
    {kind:'apple_experiment',projectId:'abc',command:'rm'}, {kind:'apple_experiment'},
    {kind:'gemini_reference',projectId:'a'.repeat(81)}, {kind:'remote_preflight',prompt:'extra'},
  ]) assert.throws(() => validateRequest('/api/jobs',body))
  assert.equal(validateRequest('/api/projects',{id:'abc',title:'My project'}).method,'POST')
  for (const body of [{id:'../a',title:'hello'}, {id:'a',title:' '}, {id:'a',title:'x'.repeat(201)}, {id:'a',title:'ok',extra:1}]) assert.throws(() => validateRequest('/api/projects',body))
})
test('only fixed media types and 32 lowercase hex suffixes allowed', () => {
  for (const id of ['reference','latest_video','comparison','reference_'+'a'.repeat(32),'audio_'+'b'.repeat(32)]) assert.equal(MEDIA.has(id),true)
  for (const id of ['../reference','reference_'+'a'.repeat(31),'reference_'+'A'.repeat(32),'reference?x=1',null]) assert.equal(MEDIA.has(id),false)
})

test('agent tasks have fixed roles and safe part scope', () => {
  const body={kind:'agent_task',projectId:'apple-experiment',agentId:'vector-agents',partId:'whole_apple',prompt:'Turn and return'}
  assert.equal(validateRequest('/api/jobs',body).method,'POST')
  for (const change of [{agentId:'shell'},{partId:'../file'},{prompt:''},{code:'print(1)'},{projectId:undefined}]) assert.throws(()=>validateRequest('/api/jobs',{...body,...change}))
  assert.equal(validateRequest('/api/jobs',{kind:'remote_preflight',projectId:'p'}).method,'POST')
})
test('remote connection cannot address arbitrary URLs', () => {
  const podId='abcdefgh123456'
  assert.equal(validateRequest('/api/remote/connect',{podId,baseUrl:`https://${podId}-8888.proxy.runpod.net`}).method,'POST')
  for (const baseUrl of ['http://127.0.0.1:22', 'https://evil.test', `https://${podId}-8888.proxy.runpod.net@evil.test`]) assert.throws(() => validateRequest('/api/remote/connect',{podId,baseUrl}))
  assert.throws(() => validateRequest('/api/remote/stop',{podId,start:true}))
})
test('sender trust pins origin and rejects credentials', () => {
  assert.equal(trustedSender('app://studio/',true),true)
  assert.equal(trustedSender('http://localhost:5173/',false),true)
  for (const url of ['https://evil.test','http://localhost:51730/','app://evil/','app://user@studio/']) assert.equal(trustedSender(url,true),false)
})
test('assets cannot escape root through encodings or backslashes', () => {
  const root=path.resolve('dist')
  assert.equal(assetPath(root,'app://studio/'),path.join(root,'index.html'))
  for (const url of ['app://studio/%2e%2e%2fsecret','app://studio/%5c..%5csecret','app://other/index.html']) assert.throws(() => assetPath(root,url))
})
test('Electron uses isolated sandbox and no raw IPC exposed', () => {
  const main=fs.readFileSync(path.join(__dirname,'main.js'),'utf8')
  const preload=fs.readFileSync(path.join(__dirname,'preload.js'),'utf8')
  assert.match(main,/sandbox: true/)
  assert.match(main,/contextIsolation: true/)
  assert.match(main,/nodeIntegration: false/)
  assert.match(main,/GAUSSIAN_STUDIO_TOKEN: TOKEN/)
  assert.match(main,/GAUSSIAN_STUDIO_BUILT === '1'/)
  assert.match(main,/trustedSender\(event.senderFrame.url, BUILT\)/)
  assert.match(main,/AbortSignal.timeout\(30000\)/)
  assert.doesNotMatch(preload,/TOKEN|process\.env|sendSync/)
})

test('all main-process HTTP uses Chromium and readiness consumes bounded body', () => {
  const main=fs.readFileSync(path.join(__dirname,'main.js'),'utf8')
  assert.doesNotMatch(main,/await fetch\(/)
  assert.match(main,/await net\.fetch\(`http:\/\/127\.0\.0\.1:\$\{PORT\}\/api\/state`/)
  assert.match(main,/await readBackendJson\(r\); backendReady = true/)
  assert.match(main,/await r\.body\?\.cancel\(\)/)
})

test('only allowlisted error codes select fixed local messages, never raw provider content', () => {
  const secret='fixture-secret-do-not-return'
  assert.match(backendErrorMessage({errorCode:'auth_failed',error:secret}),/HTTP 401/)
  assert.match(backendErrorMessage({errorCode:'permission_denied',error:secret}),/HTTP 403/)
  for (const code of ['auth_failed','permission_denied','not_found','rate_limited','gpu_capacity_unavailable','insufficient_balance','provider_error','invalid_response','connection_failed','credential_unavailable']) {
    const text=backendErrorMessage({errorCode:code,error:secret,providerHttpStatus:secret,key:secret})
    assert.doesNotMatch(text,new RegExp(secret))
    assert.notEqual(text,'Studio backend request failed')
  }
  for (const payload of [null,[],secret,{error:secret},{errorCode:secret},{errorCode:'constructor'},{errorCode:42},Object.create({errorCode:'auth_failed'})]) {
    assert.equal(backendErrorMessage(payload),'Studio backend request failed')
  }
})

test('success and failure JSON bodies use bounded stream parsing', async () => {
  for (const status of [200,502]) {
    const response=new Response(JSON.stringify({errorCode:'auth_failed',error:'never display raw text'}),{status})
    const payload=await readBackendJson(response)
    assert.equal(payload.errorCode,'auth_failed')
    assert.doesNotMatch(backendErrorMessage(payload),/never display raw text/)
  }
  await assert.rejects(readBackendJson(new Response('not json')))
  await assert.rejects(readBackendJson(new Response(null)))
})

test('stream limit includes oversized error bodies and cancels the reader', async () => {
  let cancelled=false;let reads=0
  const response={body:{getReader:()=>({
    read:async()=>({done:false,value:Buffer.alloc(++reads===1?2*1024*1024:1)}),
    cancel:async()=>{cancelled=true},
  })}}
  await assert.rejects(readBackendJson(response),/Studio backend request failed/)
  assert.equal(cancelled,true)
  assert.equal(reads,2)
})
