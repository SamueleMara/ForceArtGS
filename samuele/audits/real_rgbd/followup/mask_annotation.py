"""Automatic mask/depth availability; predictions never become verified truth."""
import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt


def mask_metrics(depth,predicted,previous_missing=None):
    valid=np.isfinite(depth)&(depth>0);missing=~valid
    n,labels,stats,centroids=cv2.connectedComponentsWithStats(missing.astype(np.uint8),8)
    result=dict(mask_truth='MASK_TRUTH_UNVERIFIED',predicted_pixels=int(predicted.sum()) if predicted is not None else None,
        predicted_valid_pixels=int((predicted&valid).sum()) if predicted is not None else None,
        missing_component_count=n-1,largest_missing_component_pixels=int(stats[1:,cv2.CC_STAT_AREA].max()) if n>1 else 0,
        persistent_missing_pixels=int((missing&previous_missing).sum()) if previous_missing is not None else None,
        previous_missing_pixels=int(previous_missing.sum()) if previous_missing is not None else None,
        human_verified_images=0,human_verified_pixels=0,manual_regions=[])
    if predicted is not None:
        count,_,objects,_=cv2.connectedComponentsWithStats((missing&predicted).astype(np.uint8),8)
        result.update(object_missing_component_count=count-1,largest_object_missing_component_pixels=int(objects[1:,cv2.CC_STAT_AREA].max()) if count>1 else 0,
            object_missing_pixels=int((missing&predicted).sum()),object_persistent_missing_pixels=int((missing&predicted&previous_missing).sum()) if previous_missing is not None else None)
        distance=distance_transform_edt(predicted)
        result['mask_boundary_bands']=[dict(low_px=l,high_px=h,pixels=int((predicted&(distance>=l)&(distance<h)).sum()),valid_pixels=int((valid&predicted&(distance>=l)&(distance<h)).sum())) for l,h in [(0,2),(2,4),(4,8),(8,16),(16,32),(32,1000000)]]
    return result,missing
