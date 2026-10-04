"""Trusted server-owned asset catalog. Prompt text cannot supply file paths."""
import hashlib
import json
from pathlib import Path


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def catalog(root):
    root = Path(root).resolve()
    path = root/'artifacts/studio/assets.json'
    if not path.exists(): return []
    rows = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(rows, list) or len(rows)>64: raise ValueError('Invalid asset catalog')
    for row in rows:
        target = (root/row['path']).resolve()
        if not target.is_relative_to(root/'artifacts') or not target.is_file():
            raise ValueError('Asset outside artifact store')
    return rows


def select_asset(root, project_id, part_id):
    found = [r for r in catalog(root) if r['projectId']==project_id and r['partId']==part_id]
    if len(found)!=1: return None
    row = found[0]
    if digest(Path(root)/row['path']) != row['sha256']: raise ValueError('Canonical asset changed')
    return dict(row)


def public_parts(root):
    result=[]
    for r in catalog(root):
        verified=False
        try:
            selected=select_asset(root,r['projectId'],r['partId'])
            # Read only bounded array headers, not a model/geometry evaluation.
            import zipfile
            from numpy.lib.format import read_magic, read_array_header_1_0, read_array_header_2_0
            with zipfile.ZipFile(Path(root)/r['path']) as archive:
                with archive.open('position.npy') as stream:
                    version=read_magic(stream)
                    shape,_,dtype=(read_array_header_1_0 if version==(1,0) else read_array_header_2_0)(stream)
            verified=bool(selected and shape==(r['gaussianCount'],3) and dtype.kind=='f')
        except (ValueError,KeyError,OSError):pass
        result.append(dict(id=r['partId'], label=r['label'], projectId=r['projectId'],
            agentId='task-assigner', ownerRoleId='task-assigner', protected=False,
            gaussianCount=r['gaussianCount'] if verified else None,bindingVerified=verified,
            bindingScope='canonical asset only; scene placement and contact unverified',
            geometryQuality=r.get('geometry_quality','Unreviewed'),
            task='Motion/recolour on canonical IDs; texture requires registered UV mapping'))
    return result
