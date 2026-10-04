const { app, BrowserWindow, protocol, net, ipcMain, session, powerSaveBlocker } = require('electron')
const { spawn } = require('child_process')
const { pathToFileURL } = require('url')
const fs = require('fs')
const { randomBytes } = require('crypto')
const path = require('path')
const { validateRequest, trustedSender, assetPath, MEDIA, backendErrorMessage, readBackendJson } = require('./security')

const DIST = path.join(__dirname, '../dist')
const PORT = 8790
const BUILT = app.isPackaged || process.env.GAUSSIAN_STUDIO_BUILT === '1'
const TOKEN = randomBytes(32).toString('hex')
let backend = null
let backendReady = false
let awakeId = null
function workingAwake(active) {
  if(active&&awakeId===null) awakeId=powerSaveBlocker.start('prevent-app-suspension')
  if(!active&&awakeId!==null){powerSaveBlocker.stop(awakeId);awakeId=null}
}
const hasInstanceLock = app.requestSingleInstanceLock()
if (!hasInstanceLock) app.quit()
app.on('second-instance', () => {
  const window = BrowserWindow.getAllWindows()[0]
  if (window) { if (window.isMinimized()) window.restore(); window.focus() }
})

// A real origin (instead of file://) so absolute asset paths, WebCrypto and localStorage all work.
protocol.registerSchemesAsPrivileged([
  { scheme: 'app', privileges: { standard: true, secure: true, supportFetchAPI: true, stream: true } },
  { scheme: 'gaussian-media', privileges: { standard: true, secure: true, supportFetchAPI: true, stream: true } },
])

async function startBackend() {
  const workspace = path.resolve(__dirname, '../../..')
  const sibling = path.join(workspace, 'outputs/gaussian_vector_prototype')
  const hasSibling = fs.existsSync(path.join(sibling, 'real_video/studio_backend.py'))
  let root
  if (app.isPackaged) {
    // Installed resources can be read-only. Only verified source is copied to
    // this writable project; existing jobs, media and credentials are not copied.
    root = process.env.GAUSSIAN_PROJECT_ROOT || path.join(app.getPath('userData'), 'outputs/gaussian_vector_prototype')
    try {
      require('./stage-backend.cjs').materializeSources(path.join(process.resourcesPath, 'backend'), root)
    } catch { console.error('Studio backend source installation failed; check the writable project directory.'); return }
  } else {
    root = process.env.GAUSSIAN_PROJECT_ROOT || (hasSibling ? sibling : path.resolve(__dirname, '../backend'))
  }
  const localPython = path.resolve(__dirname, '../.venv/Scripts/python.exe')
  const legacyPython = path.join(workspace, 'work/.venv/Scripts/python.exe')
  const python = process.env.GAUSSIAN_PYTHON || (!app.isPackaged && hasSibling && fs.existsSync(legacyPython) ? legacyPython : localPython)
  if (!fs.existsSync(path.join(root, 'real_video/studio_backend.py')) || !fs.existsSync(python)) return
  backend = spawn(python, ['-m', 'real_video.studio_backend', '--port', String(PORT)], {
    cwd: root, windowsHide: true, stdio: 'ignore', env: { ...process.env, GAUSSIAN_STUDIO_TOKEN: TOKEN },
  })
  backend.on('error', () => { backendReady = false })
  backend.on('exit', () => { backendReady = false; backend = null })
  for (let attempt = 0; attempt < 40 && backend; attempt++) {
    try {
      const r = await net.fetch(`http://127.0.0.1:${PORT}/api/state`, { headers: { Authorization: `Bearer ${TOKEN}` }, redirect: 'error', signal: AbortSignal.timeout(1500) })
      if (r.ok) { await readBackendJson(r); backendReady = true; return }
      await r.body?.cancel()
    } catch { /* Only our token-authenticated backend is accepted. */ }
    await new Promise(resolve => setTimeout(resolve, 200))
  }
}

function createWindow() {
  const win = new BrowserWindow({
    width: 1440,
    height: 900,
    minWidth: 1024,
    minHeight: 700,
    titleBarStyle: 'hiddenInset',
    trafficLightPosition: { x: 16, y: 16 },
    backgroundColor: '#0a0a0f',
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      sandbox: true,
      webSecurity: true,
      preload: path.join(__dirname, 'preload.js'),
    },
  })

  win.webContents.setWindowOpenHandler(() => ({ action: 'deny' }))
  win.webContents.on('will-navigate', (event, url) => { if (!trustedSender(url, BUILT)) event.preventDefault() })
  win.webContents.on('will-attach-webview', event => event.preventDefault())
  if (BUILT) {
    win.loadURL('app://studio/')
  } else {
    win.loadURL('http://localhost:5173')
  }
}

app.whenReady().then(async () => {
  if (!hasInstanceLock) return
  session.defaultSession.setPermissionRequestHandler((_webContents, _permission, callback) => callback(false))
  session.defaultSession.setPermissionCheckHandler(() => false)
  protocol.handle('app', (request) => {
    try { return net.fetch(pathToFileURL(assetPath(DIST, request.url)).toString()) }
    catch { return new Response('Forbidden', { status: 403 }) }
  })
  protocol.handle('gaussian-media', async request => {
    const u = new URL(request.url)
    const id = u.pathname.slice(1)
    if (u.hostname !== 'artifact' || !MEDIA.has(id) || u.search || u.username || u.password || !backendReady) return new Response('Not found', { status: 404 })
    try {
      const headers = { Authorization: `Bearer ${TOKEN}` }
      const range = request.headers.get('Range')
      if (range && /^bytes=\d+-\d*$/.test(range)) headers.Range = range
      // Chromium's network stack handles video range cancellation here. Passing
      // Node/Undici streaming responses through protocol.handle caused an
      // uncaught Parser.finish assertion when the media element cancelled a read.
      return await net.fetch(`http://127.0.0.1:${PORT}/api/media/${id}`, { headers, redirect: 'error', signal: AbortSignal.timeout(30000) })
    } catch { return new Response('Media unavailable', { status: 503 }) }
  })
  ipcMain.handle('studio:request', async (event, route, body) => {
    if (!event.senderFrame || event.senderFrame !== event.sender.mainFrame || !trustedSender(event.senderFrame.url, BUILT)) throw new Error('Untrusted studio caller')
    const checked = validateRequest(route, body)
    if (!backendReady) throw new Error('Studio backend unavailable; configure GAUSSIAN_PYTHON and GAUSSIAN_PROJECT_ROOT')
    let response, payload
    try {
      // Use Chromium for JSON too: Node/Undici also asserted during startup
      // polling, not only when cancelled video streams were proxied.
      response = await net.fetch(`http://127.0.0.1:${PORT}${route}`, {
        method: checked.method, body: checked.body, redirect: 'error',
        headers: { Authorization: `Bearer ${TOKEN}`, 'Content-Type': 'application/json' }, signal: AbortSignal.timeout(30000),
      })
      // Parse errors under the same response-size ceiling as successful state.
      payload = await readBackendJson(response)
    } catch { throw new Error('Studio backend request failed') }
    if (!response.ok) throw new Error(backendErrorMessage(payload))
    if(route==='/api/jobs') workingAwake(true)
    if(route==='/api/state') workingAwake(payload.jobs?.some(j=>['queued','running'].includes(j.status)))
    return payload
  })
  await startBackend()
  createWindow()
})
app.on('before-quit', () => { workingAwake(false); if (backend) backend.kill() })

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit()
})

app.on('activate', () => {
  if (BrowserWindow.getAllWindows().length === 0) createWindow()
})
