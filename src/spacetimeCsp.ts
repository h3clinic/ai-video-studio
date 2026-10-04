import {AlgebraicType, ProductType, SumType, type Serializer, type Deserializer} from 'spacetimedb'

// SpacetimeDB 2.10.2 emits product/sum codecs with the Function constructor.
// Its public factory objects can instead use ordinary closures. Keep primitive,
// array, option/result and special identity/time codecs supplied by the SDK.
// Install before constructing a connection; no CSP exception is required.
type Typespace = Parameters<typeof ProductType.makeSerializer>[1]
type Cache<F> = WeakMap<object, Map<Typespace, F>>
let installed = false

function cacheFor<F>(cache: Cache<F>, type: object): Map<Typespace, F> {
  let entries = cache.get(type)
  if (!entries) { entries = new Map(); cache.set(type, entries) }
  return entries
}

export function installCspSafeCodecs() {
  if (installed) return
  const originalProductDeserializer = ProductType.makeDeserializer
  const originalSumSerializer = SumType.makeSerializer
  const originalSumDeserializer = SumType.makeDeserializer
  const serializers: Cache<Serializer<any>> = new WeakMap()
  const deserializers: Cache<Deserializer<any>> = new WeakMap()
  const specialProducts = new Set(['__time_duration_micros__', '__timestamp_micros_since_unix_epoch__', '__identity__', '__connection_id__', '__uuid__'])
  const specialSum = (type: Parameters<typeof SumType.makeSerializer>[0]) => type.variants.length === 2 &&
    ((type.variants[0].name === 'some' && type.variants[1].name === 'none') ||
     (type.variants[0].name === 'ok' && type.variants[1].name === 'err'))

  ProductType.makeSerializer = (type, typespace) => {
    const entries = cacheFor(serializers, type)
    const cached = entries.get(typespace)
    if (cached) return cached
    const fields: [string, Serializer<any>][] = []
    const serialize: Serializer<any> = (writer, value) => {
      for (const [name, write] of fields) write(writer, value[name])
    }
    // Cache the closure before resolving child codecs: types may be recursive.
    entries.set(typespace, serialize)
    try {
      for (const field of type.elements) fields.push([String(field.name), AlgebraicType.makeSerializer(field.algebraicType, typespace)])
    } catch (error) { entries.delete(typespace); throw error }
    return serialize
  }

  ProductType.makeDeserializer = (type, typespace) => {
    if (type.elements.length === 1 && specialProducts.has(String(type.elements[0].name))) {
      // These five SDK branches directly construct typed wrappers; no codegen.
      return originalProductDeserializer(type, typespace)
    }
    const entries = cacheFor(deserializers, type)
    const cached = entries.get(typespace)
    if (cached) return cached
    const fields: [string, Deserializer<any>][] = []
    const deserialize: Deserializer<any> = reader => {
      const value: Record<string, any> = {}
      for (const [name, read] of fields) Object.defineProperty(value, name, {
        value: read(reader), enumerable: true, writable: true, configurable: true,
      })
      return value
    }
    entries.set(typespace, deserialize)
    try {
      for (const field of type.elements) fields.push([String(field.name), AlgebraicType.makeDeserializer(field.algebraicType, typespace)])
    } catch (error) { entries.delete(typespace); throw error }
    return deserialize
  }

  SumType.makeSerializer = (type, typespace) => {
    if (specialSum(type)) return originalSumSerializer(type, typespace)
    const entries = cacheFor(serializers, type)
    const cached = entries.get(typespace)
    if (cached) return cached
    const variants = new Map<string | undefined, {tag: number; write: Serializer<any>}>()
    const serialize: Serializer<any> = (writer, value) => {
      const variant = variants.get(value.tag)
      if (!variant) throw new TypeError('Unknown SpacetimeDB sum tag')
      writer.writeU8(variant.tag)
      variant.write(writer, value.value)
    }
    entries.set(typespace, serialize)
    try {
      type.variants.forEach((variant, tag) => variants.set(variant.name, {
        tag, write: AlgebraicType.makeSerializer(variant.algebraicType, typespace),
      }))
    } catch (error) { entries.delete(typespace); throw error }
    return serialize
  }

  SumType.makeDeserializer = (type, typespace) => {
    if (specialSum(type)) return originalSumDeserializer(type, typespace)
    const entries = cacheFor(deserializers, type)
    const cached = entries.get(typespace)
    if (cached) return cached
    const variants: {name: string | undefined; read: Deserializer<any>}[] = []
    const deserialize: Deserializer<any> = reader => {
      const variant = variants[reader.readU8()]
      if (!variant) throw new TypeError('Unknown SpacetimeDB sum tag')
      return {tag: variant.name, value: variant.read(reader)}
    }
    entries.set(typespace, deserialize)
    try {
      for (const variant of type.variants) variants.push({
        name: variant.name, read: AlgebraicType.makeDeserializer(variant.algebraicType, typespace),
      })
    } catch (error) { entries.delete(typespace); throw error }
    return deserialize
  }
  installed = true
}
