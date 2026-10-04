// Local-only server signing identity, never an external-service API key.
import { generateKeyPairSync } from 'node:crypto'
import { existsSync, mkdirSync, writeFileSync } from 'node:fs'
import { resolve, join } from 'node:path'
const folder = resolve(process.argv[2] || '')
if (!process.argv[2] || !folder.endsWith('studio-data')) throw new Error('Expected the explicit local studio-data directory')
mkdirSync(folder, {recursive:true})
const privatePath=join(folder,'identity-private.pem'), publicPath=join(folder,'identity-public.pem')
if (existsSync(privatePath)!==existsSync(publicPath)) throw new Error('Incomplete existing signing identity; refusing to overwrite')
if (!existsSync(privatePath)) {
  const keys=generateKeyPairSync('ec',{namedCurve:'prime256v1',publicKeyEncoding:{type:'spki',format:'pem'},privateKeyEncoding:{type:'pkcs8',format:'pem'}})
  writeFileSync(privatePath,keys.privateKey,{flag:'wx',mode:0o600})
  writeFileSync(publicPath,keys.publicKey,{flag:'wx',mode:0o600})
}
console.log('Local signing identity is available; no key material printed.')
