// One-off app-only demo recording. Run with Electron after closing Studio.
// This dev harness never reads keys or adds controls to the product.
const {app,powerSaveBlocker}=require('electron')
const path=require('node:path')
const recorder=require('./app-recorder.cjs')
process.env.GAUSSIAN_STUDIO_BUILT='1'
app.setName('ai-video-studio')
app.setPath('userData',path.join(app.getPath('appData'),'ai-video-studio'))
app.on('browser-window-created',(_event,window)=>{
  window.webContents.once('did-finish-load',()=>{
    setTimeout(()=>{
      const awake=powerSaveBlocker.start('prevent-display-sleep')
      void recorder.start(window,path.resolve(__dirname,'../../..'),600).catch(()=>{})
      const release=()=>{if(powerSaveBlocker.isStarted(awake))powerSaveBlocker.stop(awake)}
      setTimeout(release,612000)
      window.once('closed',release)
    },1500)
  })
})
app.on('before-quit',()=>{void recorder.stop()})
require('../electron/main.js')
