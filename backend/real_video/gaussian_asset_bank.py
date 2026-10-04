"""Stable-ID appearance/geometry bank plus separately attachable motion.

This is a representation contract, NOT learned anatomy or a fix for wrong flow.
IDs identify saved points within this asset; they are not inferred physical IDs.
Storage order is not depth/occlusion order. The existing renderer is normalized
planar blending and does not use a sorted front-to-back visibility order.
"""
import hashlib
import math
import torch

VERSION=1


def _finite_shape(value,shape,name):
    if not isinstance(value,torch.Tensor) or tuple(value.shape)!=tuple(shape):
        raise ValueError(f'{name}: expected tensor of shape {shape}')
    if value.dtype not in (torch.float32,torch.float64):
        raise ValueError(f'{name}: float32 or float64 tensor required')
    if not torch.isfinite(value).all(): raise ValueError(f'{name}: nonfinite values')


def _validate_ids(ids,name):
    if not isinstance(ids,torch.Tensor) or ids.ndim!=1 or ids.dtype!=torch.int64:
        raise ValueError(f'{name}: one-dimensional int64 IDs required')
    if len(ids)==0 or (ids<0).any() or len(torch.unique(ids))!=len(ids):
        raise ValueError(f'{name}: IDs must be nonempty, unique and nonnegative')


def geometry_hash(bank):
    """Row-order-independent geometry identity. Colour edits do not rebind motion."""
    index=bank['ids'].argsort()
    value=hashlib.sha256(b'planar-gaussian-reference-v1')
    for tensor in [bank['ids'][index],bank['geometry']['position'][index],
                   bank['geometry']['log_scale'][index],bank['geometry']['axis'][index]]:
        cpu=tensor.detach().contiguous().cpu()
        value.update(str((str(cpu.dtype),tuple(cpu.shape))).encode('ascii'))
        value.update(cpu.numpy().tobytes())
    return value.hexdigest()


def validate_bank(bank):
    if bank.get('version')!=VERSION: raise ValueError('Unsupported bank version')
    for field in ['ids','appearance','geometry','geometry_hash']:
        if field not in bank: raise ValueError(f'Bank missing {field}; colour/order alone cannot render a Gaussian')
    _validate_ids(bank['ids'],'bank'); n=len(bank['ids'])
    for name,dims in [('colour',3),('opacity',1)]:
        _finite_shape(bank['appearance'].get(name),(n,dims),name)
    for name in ['position','log_scale','axis']:
        _finite_shape(bank['geometry'].get(name),(n,2),name)
    reference=bank['geometry']['position']
    for section in ['appearance','geometry']:
        for tensor in bank[section].values():
            if tensor.dtype!=reference.dtype or tensor.device!=reference.device:
                raise ValueError('Bank fields must have matching dtype and device')
    if bank['ids'].device!=reference.device:
        raise ValueError('Bank IDs and geometry must use the same device')
    if (bank['appearance']['colour'].abs()>1).any(): raise ValueError('Colour must be in [-1,1]')
    alpha=bank['appearance']['opacity']
    if ((alpha<0)|(alpha>1)).any(): raise ValueError('Opacity must be in [0,1]')
    if (bank['geometry']['axis'].norm(dim=-1)-1).abs().max()>1e-4:
        raise ValueError('Reference axes must be unit length')
    if geometry_hash(bank)!=bank['geometry_hash']: raise ValueError('Reference geometry hash mismatch')


def bank_from_points(points,source_sha256='',ids=None):
    if points.ndim!=2 or points.shape[1]!=10: raise ValueError('Expected N,10 explicit planar Gaussian points')
    points=points.detach().cpu().clone()
    ids=torch.arange(len(points),dtype=torch.int64) if ids is None else ids.detach().cpu().clone()
    _validate_ids(ids,'bank')
    if len(ids)!=len(points): raise ValueError('One ID per Gaussian required')
    _finite_shape(points,(len(points),10),'points')
    bank=dict(version=VERSION,ids=ids,
              appearance=dict(colour=points[:,6:9].contiguous(),opacity=points[:,9:10].contiguous()),
              geometry=dict(position=points[:,:2].contiguous(),log_scale=points[:,2:4].contiguous(),axis=points[:,4:6].contiguous()),
              source_sha256=source_sha256,
              coordinate_system='Image-height normalized planar xy; reference to one observed pose, NOT canonical 3D anatomy',
              order_semantics='IDs define storage correspondence, not depth, anatomy or verified real-world identity')
    bank['geometry_hash']=geometry_hash(bank)
    validate_bank(bank)
    return bank


def make_motion(bank,ids,displacement,*,angle=None,log_scale_delta=None,visibility=None,partial=False,time_seconds=0.):
    """Absolute deformation from reference, never an implicit accumulated update."""
    if not isinstance(partial,bool): raise ValueError('partial must be an explicit boolean')
    if not math.isfinite(float(time_seconds)): raise ValueError('time_seconds must be finite')
    motion=dict(version=VERSION,geometry_hash=bank['geometry_hash'],ids=ids.detach().clone(),
                displacement=displacement.detach().clone(),partial=bool(partial),time_seconds=float(time_seconds),
                reference_mode='absolute_from_reference')
    for key,value in [('angle',angle),('log_scale_delta',log_scale_delta),('visibility',visibility)]:
        if value is not None: motion[key]=value.detach().clone()
    return motion


class GaussianAssetBank:
    """Owns a static copy; associates motion by explicit ID, not array position."""
    def __init__(self,bank,device='cpu'):
        validate_bank(bank)
        self.geometry_hash=bank['geometry_hash']
        self.ids=bank['ids'].to(device).clone()
        self.reference=torch.cat((bank['geometry']['position'],bank['geometry']['log_scale'],bank['geometry']['axis'],
                                  bank['appearance']['colour'],bank['appearance']['opacity']),-1).to(device).clone()
        self.sorted_ids,self.sorted_rows=self.ids.sort()

    def attach(self,motion):
        if motion.get('version')!=VERSION or motion.get('reference_mode')!='absolute_from_reference':
            raise ValueError('Unsupported motion contract')
        if motion.get('geometry_hash')!=self.geometry_hash: raise ValueError('Motion belongs to a different geometry bank')
        if not isinstance(motion.get('partial',False),bool): raise ValueError('partial must be an explicit boolean')
        timestamp=motion.get('time_seconds')
        if not isinstance(timestamp,(int,float)) or isinstance(timestamp,bool) or not math.isfinite(timestamp):
            raise ValueError('time_seconds must be a finite number')
        _validate_ids(motion.get('ids'),'motion')
        ids=motion['ids'].to(self.ids.device); n=len(ids)
        if not motion.get('partial',False) and n!=len(self.ids): raise ValueError('Missing IDs in a complete motion packet')
        where=torch.searchsorted(self.sorted_ids,ids)
        if (where>=len(self.ids)).any() or not torch.equal(self.sorted_ids[where.clamp_max(len(self.ids)-1)],ids):
            raise ValueError('Unknown Gaussian ID')
        rows=self.sorted_rows[where]
        _finite_shape(motion.get('displacement'),(n,2),'displacement')
        result=self.reference.clone()
        result[rows,:2]+=motion['displacement'].to(result)
        if 'angle' in motion:
            _finite_shape(motion['angle'],(n,),'angle')
            theta=motion['angle'].to(result); axis=result[rows,4:6].clone(); c,s=theta.cos(),theta.sin()
            result[rows,4:6]=torch.stack((c*axis[:,0]-s*axis[:,1],s*axis[:,0]+c*axis[:,1]),-1)
        if 'log_scale_delta' in motion:
            _finite_shape(motion['log_scale_delta'],(n,2),'log_scale_delta')
            result[rows,2:4]+=motion['log_scale_delta'].to(result)
        if 'visibility' in motion:
            _finite_shape(motion['visibility'],(n,1),'visibility')
            visibility=motion['visibility'].to(result)
            if ((visibility<0)|(visibility>1)).any(): raise ValueError('Visibility must be in [0,1]')
            result[rows,9:10]*=visibility
        if not torch.isfinite(result).all(): raise ValueError('Nonfinite deformed state')
        return result

    def storage_bytes(self):
        return sum(t.numel()*t.element_size() for t in [self.reference,self.ids,self.sorted_ids,self.sorted_rows])


def tensor_bytes(value):
    if isinstance(value,torch.Tensor): return value.numel()*value.element_size()
    if isinstance(value,dict): return sum(tensor_bytes(v) for v in value.values())
    return 0
