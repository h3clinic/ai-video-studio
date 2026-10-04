import { useRef, useState } from 'react'

export default function ElevenLabsSettings({ configured, save }: {configured:boolean;save:(key:string)=>Promise<void>}) {
  const field=useRef<HTMLInputElement>(null)
  const [busy,setBusy]=useState(false)
  const [message,setMessage]=useState('')
  const [expanded,setExpanded]=useState(!configured)
  return <section className="owner-sound-settings">
    <button aria-expanded={expanded} onClick={()=>setExpanded(!expanded)}>Sound Agent · {configured?'Update ElevenLabs key':'Connect ElevenLabs'}</button>
    {expanded&&<form onSubmit={async e=>{
      e.preventDefault();if(busy||!field.current)return
      let key=field.current.value.trim();field.current.value='';setBusy(true);setMessage('')
      try{await save(key);setMessage('Key saved.');setExpanded(false)}
      catch{setMessage('Key not saved. Check its format and try again.')}
      finally{key='';setBusy(false)}
    }}>
      <label>Owner API key<input ref={field} type="password" autoComplete="off" spellCheck={false} maxLength={512} required placeholder="Paste ElevenLabs API key" aria-label="ElevenLabs API key"/></label>
      <p>Stored with Windows encryption. Not shared.</p>
      <button disabled={busy} type="submit">{busy?'Saving…':'Save encrypted key'}</button>
    </form>}
    {message&&<p role="status">{message}</p>}
    <p>Up to 5s per effect · billed by ElevenLabs · separate audio</p>
  </section>
}
