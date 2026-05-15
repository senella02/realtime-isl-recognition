This is a markdown doc, not a Word doc request — I'll write it as a clean inline reference.

---

## Bbox Crop — Developer Reference

### Input
```
L = all MediaPipe landmarks
each landmark lᵢ has: (xᵢ, yᵢ, vᵢ)   # x, y coords + visibility score
```

---

### Step 1 — Raw Bounding Box
Filter landmarks where `vᵢ > 0.5`, then:
```
x_min = min(xᵢ)
x_max = max(xᵢ)
y_min = min(yᵢ)
y_max = max(yᵢ)
```

---

### Step 2 — Adaptive Padding
```
pad_x    = (x_max - x_min) * 0.10
pad_y    = (y_max - y_min) * 0.10
pad_head = (y_max - y_min) * 0.20   # top gets more padding for the head
```

---

### Step 3 — Final Crop (clamped to frame)
```
crop_x1 = max(0,      x_min - pad_x)
crop_y1 = max(0,      y_min - pad_head)   # use head padding on top
crop_x2 = min(width,  x_max + pad_x)
crop_y2 = min(height, y_max + pad_y)      # normal padding on bottom
```

Output: `(crop_x1, crop_y1, crop_x2, crop_y2)` → top-left to bottom-right crop region.

---

**Notes:**
- `L` includes **both pose and hand landmarks** — do not use body landmarks alone; fingertips can be the outermost points and will expand the bbox
- Visibility threshold `0.5` — skip unreliable landmarks
- Padding is **proportional** to bbox size, not fixed pixels
- Top padding (20%) ≠ bottom padding (10%) — asymmetric by design to include the head