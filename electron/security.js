const path = require('path')
const ROUTES = Object.freeze({ '/api/state': 'GET', '/api/credentials': 'POST', '/api/jobs': 'POST', '/api/projects': 'POST', '/api/remote/connect': 'POST', '/api/remote/stop': 'POST' })
const MEDIA = Object.freeze({ has: id => typeof id === 'string' && /^(reference|latest_video|comparison|audio)(?:_[a-f0-9]{32})?$/.test(id) })
const validProjectId = id => typeof id === 'string' && /^[A-Za-z0-9_-]{1,80}$/.test(id)
const validPrompt = prompt => typeof prompt === 'string' && prompt.trim().length > 0 && prompt.length <= 4000
const BACKEND_FAILURE = 'Studio backend request failed'
const BACKEND_ERROR_MESSAGES = Object.freeze({
  auth_failed: 'RunPod rejected the saved owner key (HTTP 401). The key may be disabled or revoked; adding GPU credit will not fix authentication.',
  permission_denied: 'RunPod denied this operation (HTTP 403). The saved owner key does not have the required permission.',
  not_found: 'RunPod could not find this Pod (HTTP 404). The saved Pod ID may no longer exist.',
  rate_limited: 'RunPod rate-limited this request (HTTP 429). No automatic retry was made.',
  gpu_capacity_unavailable: 'RunPod reported unavailable GPU capacity. This is separate from authentication or model quality.',
  insufficient_balance: 'RunPod reported insufficient credit. No top-up was attempted.',
  provider_error: 'RunPod returned a provider error. No automatic retry was made.',
  invalid_response: 'RunPod returned an invalid or oversized response.',
  connection_failed: 'Could not reach the RunPod control API. No automatic retry was made.',
  credential_unavailable: 'The saved owner RunPod credential is unavailable or could not be decrypted.',
})

function backendErrorMessage(payload) {
  // Error text, URLs, provider response bodies and credentials never cross IPC.
  // A recognized code selects a fixed local sentence, not server-supplied text.
  if (!payload || typeof payload !== 'object' || Array.isArray(payload) || !Object.hasOwn(payload, 'errorCode')) return BACKEND_FAILURE
  const code = payload.errorCode
  return typeof code === 'string' && Object.hasOwn(BACKEND_ERROR_MESSAGES, code) ? BACKEND_ERROR_MESSAGES[code] : BACKEND_FAILURE
}

async function readBackendJson(response) {
  if (!response.body || typeof response.body.getReader !== 'function') throw new Error(BACKEND_FAILURE)
  const reader = response.body.getReader(); const chunks = []; let length = 0
  while (true) {
    const { value, done } = await reader.read()
    if (done) break
    length += value.byteLength
    if (length > 2 * 1024 * 1024) { await reader.cancel(); throw new Error(BACKEND_FAILURE) }
    chunks.push(Buffer.from(value))
  }
  return JSON.parse(Buffer.concat(chunks).toString('utf8'))
}

function validateRequest(route, body) {
  if (typeof route !== 'string' || !Object.hasOwn(ROUTES, route)) throw new Error('Unsupported studio route')
  const method = ROUTES[route]
  if (method === 'GET' && body != null) throw new Error('GET body forbidden')
  if (method === 'POST' && (!body || Object.getPrototypeOf(body) !== Object.prototype)) throw new Error('JSON object required')
  const encoded = method === 'POST' ? JSON.stringify(body) : undefined
  if (encoded && Buffer.byteLength(encoded) > 32768) throw new Error('Request too large')
  if (route === '/api/credentials' && (!['gemini','runway','runpod','elevenlabs'].includes(body.provider) || typeof body.key !== 'string' || !body.key || body.key.length > 4096 || Object.keys(body).some(k => !['provider','key'].includes(k)))) throw new Error('Invalid credential request')
  if (route === '/api/jobs') {
    const fields = { plan: ['kind','prompt','projectId'], agent_swarm: ['kind','prompt','projectId','resumeJobId'], sound_mix:['kind','projectId','sourceJobId','gains'], agent_task: ['kind','prompt','projectId','agentId','partId'], gemini_reference: ['kind','prompt','projectId'], apple_experiment: ['kind','projectId'], remote_preflight: ['kind','projectId'] }
    if (!Object.hasOwn(fields, body.kind) || Object.keys(body).some(k => !fields[body.kind].includes(k))) throw new Error('Unsupported job or field')
    if (['plan','agent_swarm','agent_task','sound_mix','apple_experiment'].includes(body.kind) && !validProjectId(body.projectId)) throw new Error('Project ID required')
    if (body.projectId !== undefined && !validProjectId(body.projectId)) throw new Error('Invalid project ID')
    if (body.resumeJobId !== undefined && (typeof body.resumeJobId!=='string'||!/^[a-f0-9]{32}$/.test(body.resumeJobId))) throw new Error('Invalid source job')
    if(body.kind==='sound_mix' && (typeof body.sourceJobId!=='string'||!/^[a-f0-9]{32}$/.test(body.sourceJobId)||!Array.isArray(body.gains)||body.gains.length<3||body.gains.length>4||body.gains.some(x=>typeof x!=='number'||!Number.isFinite(x)||x<0||x>1))) throw new Error('Invalid sound mix')
    if (['plan','agent_swarm','agent_task'].includes(body.kind) && !validPrompt(body.prompt)) throw new Error('Bounded nonempty prompt required')
    if (body.kind === 'agent_task') {
      if (!['idea-model','video-controller','crawler','decision','vision-sensor','section-leads','field-agents','surroundings-list','kind-checkers','mini-object-agents','task-assigner','vector-agents','sound-agents','verifiers'].includes(body.agentId)) throw new Error('Unknown agent role')
      if (body.partId !== undefined && (typeof body.partId !== 'string' || !/^[a-z0-9_-]{1,60}$/.test(body.partId))) throw new Error('Invalid part')
    }
    if (body.prompt !== undefined && !validPrompt(body.prompt)) throw new Error('Invalid prompt')
  }
  if (route === '/api/projects' && (!validProjectId(body.id) || typeof body.title !== 'string' || !body.title.trim() || body.title.length > 200 || (body.archived !== undefined && typeof body.archived !== 'boolean') || Object.keys(body).some(k => !['id','title','archived'].includes(k)))) throw new Error('Invalid project')
  if (route.startsWith('/api/remote/')) {
    if (typeof body.podId !== 'string' || !/^[a-z0-9]{8,32}$/.test(body.podId)) throw new Error('Invalid pod ID')
    const keys = route.endsWith('/connect') ? ['podId','baseUrl'] : ['podId']
    if (Object.keys(body).some(k => !keys.includes(k))) throw new Error('Unexpected remote option')
    if (route.endsWith('/connect')) {
      const expected = `https://${body.podId}-8888.proxy.runpod.net`
      if (body.baseUrl !== expected && body.baseUrl !== expected + '/') throw new Error('Invalid pod proxy')
    }
  }
  return { method, body: encoded }
}
function trustedSender(url, packaged) {
  try { const u = new URL(url); return !u.username && !u.password && (packaged ? u.protocol === 'app:' && u.hostname === 'studio' && !u.port : u.origin === 'http://localhost:5173') } catch { return false }
}
function assetPath(root, url) {
  const u = new URL(url)
  if (u.protocol !== 'app:' || u.hostname !== 'studio' || u.username || u.password || u.port) throw new Error('Invalid asset origin')
  const rel = decodeURIComponent(u.pathname)
  if (rel.includes('\\') || rel.includes('\0')) throw new Error('Invalid asset path')
  const resolved = path.resolve(root, '.' + (rel === '/' ? '/index.html' : rel))
  if (resolved !== root && !resolved.startsWith(root + path.sep)) throw new Error('Asset outside application')
  return resolved
}
module.exports = { ROUTES, MEDIA, validateRequest, trustedSender, assetPath, backendErrorMessage, readBackendJson }
