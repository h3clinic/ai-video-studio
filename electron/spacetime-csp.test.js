const test = require('node:test')
const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')
const ts = require('typescript')
const vm = require('node:vm')
const sdk = require('spacetimedb')
require.extensions['.ts'] = (module, filename) => module._compile(ts.transpileModule(fs.readFileSync(filename,'utf8'), {
  compilerOptions:{module:ts.ModuleKind.CommonJS,target:ts.ScriptTarget.ES2020},
}).outputText, filename)
const {installCspSafeCodecs} = require('../src/spacetimeCsp.ts')
const {AlgebraicType:A, BinaryWriter, BinaryReader} = sdk
const field = (name, algebraicType) => ({name, algebraicType})
const product = elements => A.Product({elements})
const unit = product([])
const sum = variants => A.Sum({variants})
const option = value => sum([field('some',value),field('none',unit)])
function encode(type, value, typespace) {
  const writer = new BinaryWriter(16)
  A.makeSerializer(type, typespace)(writer, value)
  return writer.getBuffer()
}

test('interpreted codecs match upstream bytes and typed wrappers without runtime compilation', () => {
  const values = {
    flag:true, i8:-12, u8:250, i16:-1000, u16:60000, i32:-123456, u32:345678,
    i64:-987654321n, u64:987654321n, i128:-(1n<<90n), u128:1n<<100n,
    i256:-(1n<<200n), u256:1n<<230n, f32:0.5, f64:Math.PI, text:'source — unchanged',
  }
  const numeric = product(Object.keys(values).map(name=>field(name,A[name==='flag'?'Bool':name==='text'?'String':name.toUpperCase()])))
  const cases = [
    [numeric,values], [unit,{}], [A.Array(A.U8),new Uint8Array([0,1,255])],
    [A.Array(numeric),[values,values]], [option(A.String),'hello'], [option(A.String),undefined],
    [sum([field('ready',unit),field('rows',A.Array(numeric))]),{tag:'rows',value:[values]}],
    [sum([field('ok',A.String),field('err',A.String)]),{err:'expected fixture error'}],
    [sdk.Identity.getAlgebraicType(),new sdk.Identity(123n)],
    [sdk.ConnectionId.getAlgebraicType(),new sdk.ConnectionId(45n)],
    [sdk.Timestamp.getAlgebraicType(),new sdk.Timestamp(1234567n)],
    [sdk.TimeDuration.getAlgebraicType(),new sdk.TimeDuration(-100n)],
    [sdk.Uuid.getAlgebraicType(),new sdk.Uuid(123n)],
  ]
  const recursive = product([field('value',A.U32),field('next',option(A.Ref(0)))])
  cases.push([recursive,{value:1,next:{value:2,next:undefined}},{types:[recursive]}])
  const expected = cases.map(([type,value,space])=>({bytes:encode(type,value,space),value:A.makeDeserializer(type,space)(new BinaryReader(encode(type,value,space)))}))
  const originalFunction = global.Function
  global.Function = function(){throw new Error('Runtime string code generation forbidden')}
  try {
    installCspSafeCodecs();installCspSafeCodecs()
    cases.forEach(([type,value,space],index)=>{
      const bytes=encode(type,value,space)
      assert.deepEqual(bytes,expected[index].bytes)
      assert.deepEqual(A.makeDeserializer(type,space)(new BinaryReader(bytes)),expected[index].value)
    })
    assert.throws(()=>A.makeDeserializer(sum([field('only',unit)]))(new BinaryReader(new Uint8Array([10]))),/Unknown/)
    assert.throws(()=>encode(sum([field('only',unit)]),{tag:'missing'}),/Unknown/)
  } finally {global.Function=originalFunction}
})

test('connection installs the adapter and keeps strict CSP',()=>{
  const root=path.join(__dirname,'..')
  assert.match(fs.readFileSync(path.join(root,'src/spacetime.ts'),'utf8'),/installCspSafeCodecs\(\)/)
  const html=fs.readFileSync(path.join(root,'index.html'),'utf8')
  assert.match(html,/script-src 'self';/)
  assert.doesNotMatch(html,/unsafe-eval|wasm-unsafe-eval/)
  assert.equal(JSON.parse(fs.readFileSync(path.join(root,'node_modules/spacetimedb/package.json'),'utf8')).version,'2.10.2', 'Review the adapter before changing SDK versions')
})

test('a synchronous builder failure can be retried without retaining a rejected promise',async()=>{
  const source=fs.readFileSync(path.join(__dirname,'../src/spacetime.ts'),'utf8')
  const compiled=ts.transpileModule(source,{compilerOptions:{module:ts.ModuleKind.CommonJS,target:ts.ScriptTarget.ES2020}}).outputText
  let attempts=0
  const exports={}
  vm.runInNewContext(compiled,{
    exports,
    require:name=>name==='./spacetimeCsp'?{installCspSafeCodecs(){}}:{DbConnection:{builder(){attempts++;throw new Error('Offline fixture')}},tables:{}},
    window:{setTimeout,clearTimeout},localStorage:{getItem(){return null}},console,
  },{contextCodeGeneration:{strings:false,wasm:false}})
  await assert.rejects(exports.connectCollaboration(),/Could not connect/)
  await assert.rejects(exports.connectCollaboration(),/Could not connect/)
  assert.equal(attempts,2)
  assert.equal(exports.getCollaborationSnapshot().state,'error')
})
