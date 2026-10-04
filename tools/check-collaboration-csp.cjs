// Read-only real loopback connection. Run with Node's string-codegen ban:
// node --disallow-code-generation-from-strings tools/check-collaboration-csp.cjs
const fs = require('node:fs')
const ts = require('typescript')
require.extensions['.ts'] = (module, filename) => module._compile(ts.transpileModule(fs.readFileSync(filename,'utf8'), {
  compilerOptions:{module:ts.ModuleKind.CommonJS,target:ts.ScriptTarget.ES2020},
}).outputText, filename)
const {installCspSafeCodecs} = require('../src/spacetimeCsp.ts')
const {DbConnection,tables} = require('../src/module_bindings/collaboration')
installCspSafeCodecs()
let connection
const timer=setTimeout(()=>finish(false),8000)
function finish(passed) {
  clearTimeout(timer)
  connection?.disconnect()
  console.log(JSON.stringify({status:passed?'passed':'failed',database:'local-only',
    string_codegen_disabled:process.execArgv.includes('--disallow-code-generation-from-strings'),
    checks:['SDK connection construction','server identity handshake','six private-view subscriptions applied'],
    reducer_calls:0,credentials_logged:false}))
  process.exitCode=passed?0:1
}
try {
  if (!process.execArgv.includes('--disallow-code-generation-from-strings')) throw new Error('Codegen ban required')
  connection=DbConnection.builder().withUri('ws://127.0.0.1:3000').withDatabaseName('ai-video-studio-collaboration').withToken(undefined)
    .onConnect(client=>client.subscriptionBuilder().onApplied(()=>finish(true)).onError(()=>finish(false))
      .subscribe([tables.myProjects,tables.myMembers,tables.myDrafts,tables.myEdits,tables.myTimings,tables.myActivity]))
    .onConnectError(()=>finish(false)).build()
} catch {finish(false)}
