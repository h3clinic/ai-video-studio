const { app, BrowserWindow, protocol, net } = require('electron')
const { spawn } = require('child_process')
const { pathToFileURL } = require('url')
const fs = require('fs')
const os = require('os')
const path = require('path')
const tcp = require('net')

const DIST = path.join(__dirname, '../dist')

// A real origin (instead of file://) so absolute asset paths, WebCrypto and localStorage all work.
protocol.registerSchemesAsPrivileged([
  { scheme: 'app', privileges: { standard: true, secure: true, supportFetchAPI: true, stream: true } },
])

// Best effort: start the local SpaceTimeDB server if nothing is listening yet.
function ensureDatabaseServer() {
  const probe = tcp.connect(3000, '127.0.0.1')
  probe.on('connect', () => probe.destroy())
  probe.on('error', () => {
    const bin = path.join(os.homedir(), '.local/bin/spacetime')
    if (fs.existsSync(bin)) spawn(bin, ['start'], { detached: true, stdio: 'ignore' }).unref()
  })
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
    },
  })

  if (app.isPackaged) {
    win.loadURL('app://studio/')
  } else {
    win.loadURL('http://localhost:5173')
  }
}

app.whenReady().then(() => {
  protocol.handle('app', (request) => {
    const rel = decodeURIComponent(new URL(request.url).pathname)
    const file = path.join(DIST, rel === '/' ? 'index.html' : rel)
    if (!file.startsWith(DIST)) return new Response('Forbidden', { status: 403 })
    return net.fetch(pathToFileURL(file).toString(), { headers: request.headers })
  })
  ensureDatabaseServer()
  createWindow()
})

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit()
})

app.on('activate', () => {
  if (BrowserWindow.getAllWindows().length === 0) createWindow()
})
