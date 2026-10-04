"""Conservative local repair utilities; not learned generation or physical lighting."""
import cv2
import numpy as np

def repair_selected(image, write_mask, excluded_donors):
    """Exclude the ENTIRE old object from donors, write only the approved subset.

    A replaced core may be hidden by a new asset but must not leak its old colour
    into repaired peel boundaries. This does not make Telea temporally coherent.
    """
    image=np.asarray(image)
    write_mask=np.asarray(write_mask,dtype=bool)
    excluded_donors=np.asarray(excluded_donors,dtype=bool)
    if image.dtype!=np.uint8 or image.ndim!=3 or image.shape[-1]!=3:
        raise ValueError('Expected RGB uint8 image')
    if write_mask.shape!=image.shape[:2] or excluded_donors.shape!=write_mask.shape:
        raise ValueError('Mask shape mismatch')
    if np.any(write_mask & ~excluded_donors):raise ValueError('Writes must be excluded from donors')
    if excluded_donors.all():raise ValueError('No observed donor pixels')
    result=image.copy()
    if write_mask.any():
        repaired=cv2.inpaint(image,excluded_donors.astype('uint8')*255,3,cv2.INPAINT_TELEA)
        result[write_mask]=repaired[write_mask]
    return result

def fit_bounded_exposure(colours, reference_colours, max_gain=2.5):
    """One frozen scene-relative display exposure, not a relighting model.

    Reference colours must be trusted non-target samples. Linear-light gain is
    capped; unseen geometry, shadows, roughness and contact are unchanged.
    """
    c=np.asarray(colours,dtype=np.float32)
    r=np.asarray(reference_colours,dtype=np.float32)
    if c.ndim!=2 or r.ndim!=2 or c.shape[1]!=3 or r.shape[1]!=3 or min(len(c),len(r))<1:
        raise ValueError('Nonempty Nx3 colours required')
    if not np.isfinite(c).all() or not np.isfinite(r).all() or np.any(c<0) or np.any(c>1) or np.any(r<0) or np.any(r>1):
        raise ValueError('Finite normalized colours required')
    if not 1<=max_gain<=3:raise ValueError('Exposure cap outside supported range')
    def linear(x):return np.where(x<=.04045,x/12.92,((x+.055)/1.055)**2.4)
    weights=np.array([.2126,.7152,.0722],dtype=np.float32)
    lc=linear(c); lr=linear(r)
    baseline=float(np.median(lc@weights))
    gain=1. if baseline<1e-6 else float(np.clip(np.median(lr@weights)/baseline,1,max_gain))
    adjusted=np.clip(lc*gain,0,1)
    srgb=np.where(adjusted<=.0031308,adjusted*12.92,1.055*adjusted**(1/2.4)-.055)
    return srgb.clip(0,1).astype(np.float32),gain
