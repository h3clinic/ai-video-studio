const { contextBridge, ipcRenderer } = require('electron')
contextBridge.exposeInMainWorld('gaussianStudio', Object.freeze({
  request: (route, body) => ipcRenderer.invoke('studio:request', route, body),
  mediaUrl: id => typeof id === 'string' && /^(reference|latest_video|comparison|audio)(?:_[a-f0-9]{32})?$/.test(id) ? `gaussian-media://artifact/${id}` : null,
}))
