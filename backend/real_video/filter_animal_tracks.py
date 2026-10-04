"""Correct mixed-object inclusion while preserving original experiments."""
import json
import torch
from .animal_graph_data import DATA
from .train_animal_visual import CACHE
from .prepare_animal_motion import WORK
from .checkpoint_io import load_verified,save_inference_checkpoint,digest


def main():
    destination=WORK/'control_tracks_animal_only_v2.pt'; visual_destination=WORK/'visual_targets_animal_only_v2.pt'
    if destination.exists() or visual_destination.exists(): raise FileExistsError('Preserve filtered caches')
    data=load_verified(DATA); visual=load_verified(CACHE); assert data['metadata']==visual['metadata']
    keep=[i for i,m in enumerate(data['metadata']) if not (m['sequence']=='dogs-scale' and m['object'] in [3,4])]
    removed=[m for i,m in enumerate(data['metadata']) if i not in keep]
    filtered={key:value[keep] if isinstance(value,torch.Tensor) and len(value)==len(data['metadata']) else value for key,value in data.items()}
    filtered['metadata']=[data['metadata'][i] for i in keep]
    filtered['config']=dict(data['config'],animal_only_correction='dogs-scale objects1,2 are dogs;3 is hand;4 is scale. First images and annotations inspected. Exclude3,4. Other contributing IDs are animals.')
    sha=save_inference_checkpoint(filtered,destination)
    visual_sha=save_inference_checkpoint(dict(records=[visual['records'][i] for i in keep],metadata=filtered['metadata'],source_tracks_sha256=sha,
                                              scope=visual['scope']),visual_destination)
    report=dict(removed=removed,train=sum(m['split']=='train' for m in filtered['metadata']),validation=sum(m['split']=='validation' for m in filtered['metadata']),
                control_sha256=sha,visual_sha256=visual_sha,original_control_sha256=digest(DATA),original_visual_sha256=digest(CACHE),
                note='Original mixed-object data and models retained. Validation unchanged. No final-test or cat-demo based filtering.')
    (WORK/'animal_only_correction.json').write_text(json.dumps(report,indent=2)); print(json.dumps(report),flush=True)


if __name__=='__main__': main()
