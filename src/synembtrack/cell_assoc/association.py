
# association.py
# Frame-to-frame association: IoU → distance (prev) → IoU (waiting) → distance (waiting) → new IDs.

from typing import List, Tuple, Optional, Any
import numpy as np

def _iou_set(a_pts: List[Tuple[int,int]], b_pts: List[Tuple[int,int]]) -> float:
    aset = set(a_pts); bset = set(b_pts)
    inter = len(aset & bset)
    if inter == 0:
        return 0.0
    union = len(aset | bset)
    return inter / union

def associate_cells(
    frame: int,
    pos, lbled_pts, 
    pre_associ, pre_associ_pxs,
    wait_tab, wait_pxs,
    next_id: int,
    neighbor_px: float,
    iou_th: float,
    speed_px_per_frame: float,
        *,
    use_mask: bool = False,
    use_box: bool = False,
    use_misc: bool = False,
    mask_info: Optional[List[List[float]]] = None,   # [[area, angle_x_fit], ...]
    boxs: Optional[List[List[float]]] = None,        # [[bbox_angle, bbox_w, bbox_h], ...]
    miscels: Optional[List[List[Any]]] = None,       # [[w1y, w1x, w2y, w2x], ...]
):
    """Return (associ_tab, associ_pxs, lost_tab, app_tab, wait_tab, wait_pxs, next_id)."""
    associ_tab, associ_pxs = [], []
    # pos = list(pos); boxs = list(boxs); lbled_pts = list(lbled_pts); miscels = list(miscels)
    pos = list(pos)
    lbled_pts = list(lbled_pts)
    boxs = list(boxs) if boxs is not None else None
    miscels = list(miscels) if miscels is not None else None

    associ_flag = np.zeros(len(pos), dtype=np.uint8)        # 0->1 if matched
    fin_flag    = np.ones(len(pre_associ), dtype=np.uint8)  # 1->0 if pre was taken

    pre_pos = pre_associ[:, 2:4].astype(np.float64) if len(pre_associ) else np.empty((0,2))

    # ADD near top of function
    n = len(pos)
    mask_iter = (mask_info if (use_mask and mask_info is not None) else [None] * n)
    box_iter  = (boxs      if (use_box  and boxs is not None)      else [None] * n)
    misc_iter = (miscels   if (use_misc and miscels is not None)   else [None] * n)

    # Pass 1-1: IoU vs previous within neighbor radius
    for i, (p, pts, msk, b, mis) in enumerate(zip(pos, lbled_pts, mask_iter, box_iter, misc_iter)):
        if len(pre_associ) == 0: break
        d = np.linalg.norm(pre_pos - np.array([p]), axis=1)
        neigh = np.where(d < neighbor_px)[0]
        if neigh.size == 0:
            continue
        best_iou, best_j = -1.0, -1
        for j in neigh:
            if fin_flag[j] == 0: continue
            iou = _iou_set(pre_associ_pxs[j][-1], pts)
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_iou >= iou_th and best_j >= 0:
            linked = pre_associ[best_j]
            associ_flag[i] = 1
            fin_flag[best_j] = 0

            row = [frame, linked[1]] + p
            if use_mask and msk is not None:  row += list(msk)  # [area, angle_x_fit]
            if use_box and b is not None:     row += list(b)    # [bbox_angle, bbox_w, bbox_h]
            if use_misc and mis is not None:  row += list(mis)  # [w1y, w1x, w2y, w2x]
            row += [1] ### association code

            associ_tab.append(row)
            associ_pxs.append([frame, linked[1]] + p + [pts])

    # Pass 1-2: distance vs previous
    for i, (p, pts, msk, b, mis) in enumerate(zip(pos, lbled_pts, mask_iter, box_iter, misc_iter)):
        if associ_flag[i] == 1 or len(pre_associ) == 0: continue
        d = np.linalg.norm(pre_pos - np.array([p]), axis=1)
        neigh = np.where(d < neighbor_px)[0]
        if neigh.size == 0: continue
        order = np.argsort(d[neigh])
        for k in order:
            j = neigh[k]
            if fin_flag[j] == 0: continue
            if d[neigh[k]] <= speed_px_per_frame:
                linked = pre_associ[j]
                associ_flag[i] = 1
                fin_flag[j] = 0

                row = [frame, linked[1]] + p
                if use_mask and msk is not None:  row += list(msk)  # [area, angle_x_fit]
                if use_box and b is not None:     row += list(b)    # [bbox_angle, bbox_w, bbox_h]
                if use_misc and mis is not None:  row += list(mis)  # [w1y, w1x, w2y, w2x]
                row += [2] ### association code

                associ_tab.append(row)
                associ_pxs.append([frame, linked[1]] + p + [pts])

                break




    # Pass 2-1: IoU vs waiting
    if len(wait_tab):
        for i, (p, pts, msk, b, mis) in enumerate(zip(pos, lbled_pts, mask_iter, box_iter, misc_iter)):
            if associ_flag[i] == 1 or len(wait_tab) == 0: continue

            # Find best IoU over current wait_pxs
            best_iou, best_j = -1.0, -1
            for j, wpx in enumerate(wait_pxs):
                iou = _iou_set(wpx[-1], pts)
                if iou > best_iou:
                    best_iou, best_j = iou, j
            if best_iou >= iou_th and best_j >= 0:
                linked = wait_tab[best_j]
                associ_flag[i] = 1

                row = [frame, linked[1]] + p
                if use_mask and msk is not None:  row += list(msk)  # [area, angle_x_fit]
                if use_box and b is not None:     row += list(b)    # [bbox_angle, bbox_w, bbox_h]
                if use_misc and mis is not None:  row += list(mis)  # [w1y, w1x, w2y, w2x]
                row += [3] ### association code

                associ_tab.append(row)
                associ_pxs.append([frame, linked[1]] + p + [pts])

                wait_tab = np.delete(wait_tab, best_j, axis=0)
                del wait_pxs[best_j]
                # wait_pxs = list(np.delete(wait_pxs, best_j, axis=0))
                if len(wait_tab) == 0: break

    # Pass 2-2: distance vs waiting
    if len(wait_tab):
        for i, (p, pts, msk, b, mis) in enumerate(zip(pos, lbled_pts, mask_iter, box_iter, misc_iter)):
            if associ_flag[i] == 1 or len(wait_tab) == 0: continue


            wpos = wait_tab[:, 2:4].astype(np.float64)
            d = np.linalg.norm(wpos - np.array([p]), axis=1)
            j = int(np.argmin(d)) if len(d) else -1
            if j >= 0 and d[j] <= 5.0 * speed_px_per_frame:
                try:
                    linked = wait_tab[j]
                except: print(len(wpos), len(d), len(wait_tab),j)
                associ_flag[i] = 1

                row = [frame, linked[1]] + p
                if use_mask and msk is not None:  row += list(msk)  # [area, angle_x_fit]
                if use_box and b is not None:     row += list(b)    # [bbox_angle, bbox_w, bbox_h]
                if use_misc and mis is not None:  row += list(mis)  # [w1y, w1x, w2y, w2x]
                row += [4] ### association code

                associ_tab.append(row)
                associ_pxs.append([frame, linked[1]] + p + [pts])

                wait_tab = np.delete(wait_tab, j, axis=0)
                del wait_pxs[j]

                if len(wait_tab) == 0: break
                # wait_pxs = list(np.delete(wait_pxs, j, axis=0))

    # New appearances
    #app_tab = []
    for i, (p, pts, msk, b, mis) in enumerate(zip(pos, lbled_pts, mask_iter, box_iter, misc_iter)):
        if associ_flag[i] == 1: continue

        row = [frame, next_id] + p
        if use_mask and msk is not None:  row += list(msk)  # [area, angle_x_fit]
        if use_box and b is not None:     row += list(b)    # [bbox_angle, bbox_w, bbox_h]
        if use_misc and mis is not None:  row += list(mis)  # [w1y, w1x, w2y, w2x]
        row += [0] ### association code

        associ_tab.append(row)
        associ_pxs.append([frame, next_id] + p + [pts])
        next_id += 1

    associ_tab = np.array(associ_tab, dtype=object)

    # Lost from previous
    assigned_prev_idx = np.where(fin_flag == 0)[0]
    lost_idx = [i for i in range(len(pre_associ)) if i not in assigned_prev_idx]
    if lost_idx:
        lost_tab = pre_associ[lost_idx]
        lost_pxs = [pre_associ_pxs[i] for i in lost_idx]
        wait_tab = np.append(wait_tab, lost_tab, axis=0)
        wait_pxs.extend(lost_pxs)
    else:
        lost_tab = np.empty((0, associ_tab.shape[1]), dtype=object)

    return associ_tab, associ_pxs, lost_tab, wait_tab, wait_pxs, next_id
