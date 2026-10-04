"""General related-part planning. No model suggestion is deletion authority.

Inventory/bindings come from a trusted scene writer, not from Gemini's JSON.
Unmapped leftovers, ambiguous ownership and unreviewed frames block execution.
"""
import math

RELATIONS=('target','part_of','detached_from','effect_of','contact_with','supports','occludes','theme_related','unrelated')


def plan_elimination(target, inventory, proposal, frame_ids):
    if not isinstance(inventory,dict) or target not in inventory or not 1<=len(inventory)<=512:
        raise ValueError('Bounded trusted inventory and existing target required')
    if not frame_ids or any(type(i) is not int or i<0 for i in frame_ids):
        raise ValueError('Explicit sampled-frame IDs required')
    if not isinstance(proposal,dict) or set(proposal)!={'candidates','unknown_parts','reviewed_frames'}:
        raise ValueError('Unexpected proposal fields')
    if not isinstance(proposal['candidates'],list) or len(proposal['candidates'])>512:
        raise ValueError('Bounded candidate list required')
    if not isinstance(proposal['unknown_parts'],list) or len(proposal['unknown_parts'])>128:
        raise ValueError('Bounded unknown-part list required')
    reviewed=proposal['reviewed_frames']
    if not isinstance(reviewed,list) or any(type(i) is not int or i<0 for i in reviewed) or len(set(reviewed))!=len(reviewed):
        raise ValueError('Distinct integer reviewed frame IDs required')
    blockers=[];remove={target};rerender=set();inspect=set();seen=set()
    if set(proposal['reviewed_frames'])!=set(frame_ids):blockers.append('Not every supplied frame was reviewed')
    # Transitive ownership closure; a peel can itself own pith/fragments.
    while True:
        more={oid for oid,node in inventory.items() if node.get('parent') in remove
              and node.get('relation') in ('part_of','detached_from')}
        if more<=remove:break
        remove|=more
    for oid,node in inventory.items():
        if node.get('parent') in remove and node.get('relation') in ('effect_of','contact_with','supports','occludes'):
            rerender.add(oid)
    for item in proposal['candidates']:
        if not isinstance(item,dict) or set(item)!={'entity_id','relation','confidence','frames','reason'}:
            raise ValueError('Malformed candidate')
        oid=item['entity_id'];relation=item['relation'];confidence=item['confidence']
        if not isinstance(oid,str) or oid in seen or relation not in RELATIONS:
            raise ValueError('Duplicate or malformed entity/relation')
        seen.add(oid)
        if type(confidence) not in (int,float) or not math.isfinite(confidence) or not 0<=confidence<=1:
            raise ValueError('Invalid confidence')
        if not isinstance(item['reason'],str) or not item['reason'].strip():raise ValueError('Reason required')
        if not isinstance(item['frames'],list) or not item['frames'] or any(type(i) is not int or i not in frame_ids for i in item['frames']):
            raise ValueError('Frame evidence missing')
        if oid not in inventory:
            blockers.append('Unmapped candidate: '+oid);continue
        if confidence<.85:
            inspect.add(oid);blockers.append('Uncertain candidate: '+oid);continue
        if relation in ('target','part_of','detached_from') and oid not in remove:
            blockers.append('Ownership not verified: '+oid);inspect.add(oid)
        elif relation in ('effect_of','contact_with','supports','occludes'):
            rerender.add(oid)
        elif relation=='theme_related':
            inspect.add(oid);blockers.append('Theme association is not removal authority: '+oid)
    for item in proposal['unknown_parts']:
        if not isinstance(item,str) or not item.strip() or len(item)>1000:raise ValueError('Invalid unknown-part description')
        blockers.append('Needs detection/segmentation: '+item)
    for oid in remove:
        node=inventory[oid]
        if node.get('protected') is not False or node.get('binding_verified') is not True:
            blockers.append('Protected or unverified Gaussian binding: '+oid)
        if type(node.get('revision')) is not int or node['revision']<0:
            blockers.append('Missing state revision: '+oid)
    return dict(remove=sorted(remove),rerender=sorted(rerender-remove),inspect=sorted(inspect),
        preserve=sorted(set(inventory)-remove),blockers=blockers,
        scope_verified=not blockers,executable=False,
        required_next=['Resolve unknown parts and verify segmentation across time',
        'Bind masks to stable Gaussian/mesh IDs and lock exact revisions',
        'Repair exposed background and old shadows/reflections',
        'Render new lighting/contact/occlusion; test for residual source parts',
        'Independent residual and temporal review before committing'],
        note='Planning only. Gemini confidence is not calibrated geometry evidence; no deletion or rendering performed.')
