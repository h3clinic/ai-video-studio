// Checked-in source bundle only: never keys, environments, checkpoints or videos.
const fs = require('fs')
const path = require('path')
const crypto = require('crypto')
const target = path.resolve(__dirname, '../backend')
const DIRECTORIES = ['real_video', 'cloud', 'gv', 'tests']
const SOURCE_PATH = /^(real_video|cloud|gv|tests)\/[A-Za-z0-9_]+\.py$/
const sha256 = bytes => crypto.createHash('sha256').update(bytes).digest('hex')

function readManifest(root) {
  const manifest = JSON.parse(fs.readFileSync(path.join(root, 'source-manifest.json'), 'utf8'))
  const seen = new Set()
  if (!Array.isArray(manifest) || !manifest.length || manifest.length > 2000) throw new Error('Invalid backend source manifest')
  for (const row of manifest) {
    if (!row || !SOURCE_PATH.test(row.path) || !/^[a-f0-9]{64}$/.test(row.sha256) || seen.has(row.path)) throw new Error('Invalid backend source entry')
    seen.add(row.path)
    const file = path.join(root, row.path)
    if (fs.lstatSync(file).isSymbolicLink() || !fs.statSync(file).isFile() || sha256(fs.readFileSync(file)) !== row.sha256) throw new Error('Backend source checksum mismatch: ' + row.path)
  }
  if (!seen.has('real_video/studio_backend.py')) throw new Error('Studio backend missing from source bundle')
  return manifest
}

function refreshSources(source, destination = target) {
  if (path.resolve(source) === path.resolve(destination)) throw new Error('Source and bundle must be different directories')
  const manifest = []
  for (const directory of DIRECTORIES) {
    const folder = path.join(source, directory)
    for (const name of fs.readdirSync(folder).sort()) {
      const relative = `${directory}/${name}`
      if (!SOURCE_PATH.test(relative) || !fs.lstatSync(path.join(folder,name)).isFile()) continue
      const bytes = fs.readFileSync(path.join(source,relative))
      fs.mkdirSync(path.dirname(path.join(destination,relative)),{recursive:true})
      fs.writeFileSync(path.join(destination,relative),bytes)
      manifest.push({path:relative,sha256:sha256(bytes)})
    }
  }
  fs.writeFileSync(path.join(destination,'source-manifest.json'),JSON.stringify(manifest,null,2)+'\n')
  return readManifest(destination)
}

function materializeSources(source, destination) {
  const manifest = readManifest(source)
  fs.mkdirSync(destination, {recursive:true})
  const previousPath = path.join(destination, 'source-manifest.json')
  const previous = fs.existsSync(previousPath) ? JSON.parse(fs.readFileSync(previousPath, 'utf8')) : []
  const priorHashes = new Map(Array.isArray(previous) ? previous.map(row=>[row.path,row.sha256]) : [])
  // Check every destination before overwriting any previously installed source.
  for (const row of manifest) {
    const folder = path.dirname(path.join(destination,row.path))
    if (fs.existsSync(folder) && fs.lstatSync(folder).isSymbolicLink()) throw new Error('Backend source directory cannot be a link')
    const file = path.join(destination,row.path)
    if (fs.existsSync(file)) {
      if (!fs.lstatSync(file).isFile()) throw new Error('Backend source target is not a regular file')
      const current = sha256(fs.readFileSync(file))
      if (current !== row.sha256 && current !== priorHashes.get(row.path)) throw new Error('Locally modified backend source: ' + row.path)
    }
  }
  for (const row of manifest) {
    const file = path.join(destination,row.path)
    fs.mkdirSync(path.dirname(file), {recursive:true})
    fs.copyFileSync(path.join(source,row.path),file)
  }
  fs.writeFileSync(previousPath,JSON.stringify(manifest,null,2)+'\n')
  return manifest
}

if (require.main === module) {
  const args = process.argv.slice(2)
  if (args.length && (args.length !== 2 || args[0] !== '--refresh-from')) throw new Error('Usage: stage-backend.cjs [--refresh-from SOURCE_DIRECTORY]')
  const manifest = args.length ? refreshSources(path.resolve(args[1])) : readManifest(target)
  const listed = new Set(manifest.map(row=>row.path))
  for (const directory of DIRECTORIES) {
    for (const name of fs.readdirSync(path.join(target,directory))) {
      if (name.endsWith('.py') && !listed.has(`${directory}/${name}`)) throw new Error('Unlisted backend source requires review: ' + directory + '/' + name)
    }
  }
  console.log(`Verified ${manifest.length} Python source files; no credentials or model data`)
}
module.exports = {readManifest, refreshSources, materializeSources}
