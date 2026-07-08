# Day 47 — Datasets: KITTI & NuScenes

> **Week 7 · CV & VLMs** — models are only half of research; **data** is the other half. KITTI and
> NuScenes are *the* academic autonomous-driving datasets — the ones your advisor named for running
> inference. Today you learn their formats so you can actually load them.
>
> *Guided, hands-on. Understanding the data format is the skill.*

## Why today matters

Your advisor's instruction was specific: "take open datasets (KITTI, NuScenes), run inference, and
**understand the input/output data format**." That last part is where most people get stuck — not the
model, but *how the data is laid out*: where the images are, how labels are stored, and how camera and
sensor coordinates relate. Get the format right and everything else follows. These are also the closest
public analog to what openpilot's cameras produce.

## The two datasets at a glance

| | **KITTI** | **NuScenes** |
|---|-----------|--------------|
| Scale | Small, classic (2012) | Large, modern (2019), 1000 scenes |
| Sensors | 2 cameras, LiDAR, GPS/IMU | 6 cameras (360°), LiDAR, 5 radar, GPS/IMU |
| Labels | 2D + 3D boxes, per-frame `.txt` | 3D boxes + tracking, a relational database (JSON) |
| Best for | Learning the basics fast | Realistic, multi-sensor, temporal |

## KITTI label format (read one line)

Each object is a line in a per-image `.txt` file. The key fields:
```
type  truncated occluded alpha  x1 y1 x2 y2   h w l   x y z   ry
Car   0.00      0        1.85   387 181 423 203  1.4 1.6 3.8  2.3 1.4 8.4  -1.5
```
- `type` — class ("Car", "Pedestrian", "Cyclist", ...)
- `x1 y1 x2 y2` — the **2D box** in the image (the format from Days 42–43!)
- `h w l` + `x y z` + `ry` — the **3D box**: size, position in camera coords, rotation.

That 2D box is exactly what your Day 42 `iou` and a detector's output use — you already speak this language.

## NuScenes format (the mental model)

NuScenes is a set of JSON tables that point at each other by **token** (a unique id string — think of
it as a name tag). A `sample` (one moment in time) holds the tokens of its `sample_data` (each sensor's
file at that moment) and its `sample_annotation`s (each labeled 3D box); an `instance` links the same
physical object's annotations *across* moments (tracking). "Look up the record this token names" is all
a *join* means here. You query it with the `nuscenes-devkit`, not by reading files by hand.

In code that looks like **dicts inside dicts** (plain Python, nothing exotic):
`sample["data"]` is a dict mapping sensor names to tokens, so `sample["data"]["CAM_FRONT"]` reads
"from this sample, take the data dict, then take the front camera's token" — two `[...]` lookups in a
row, exactly like `d["a"]["b"]` on any nested dict.

## Do this

**1 · Explore without a full download.** Both provide **mini/sample** splits — get those first (a few GB),
not the full hundreds of GB.
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install nuscenes-devkit           # official loader
# Download the NuScenes 'v1.0-mini' split from nuscenes.org (free account), then:
```
```python
from nuscenes.nuscenes import NuScenes
nusc = NuScenes(version="v1.0-mini", dataroot="/path/to/nuscenes", verbose=True)
sample = nusc.sample[0]
cam = nusc.get("sample_data", sample["data"]["CAM_FRONT"])
print("front-camera image:", cam["filename"])         # the image file for this moment
ann = nusc.get("sample_annotation", sample["anns"][0])
print("first object:", ann["category_name"], "size", ann["size"])
```

**2 · Run a detector on a real frame.** Load a KITTI or NuScenes front-camera image and run Day 43's YOLO
on it. Compare YOLO's boxes to the dataset's ground-truth boxes — that comparison is the seed of *evaluation*.

## ✅ You did it right if…

- `NuScenes(version="v1.0-mini", ...)` loads without error and prints table counts (scenes, samples…).
- You can print a front-camera image filename for a sample, and the file exists on disk.
- For one annotation you can name its `category_name` (e.g. `vehicle.car`) and its `size` (3 numbers —
  width/length/height in meters, sane values like `[1.9, 4.5, 1.6]` for a car).
- Bonus: YOLO on that camera image finds the cars the ground truth says are there.

## No download yet? Do the format exercise anyway

The download (free account, a few GB for mini) is the friction point — the *format* is the lesson, and
you can practice it right now. Parse this real-shaped KITTI label line with `.split()` and name every
field from the table above:
```
Car 0.00 0 1.85 387.63 181.54 423.81 203.12 1.67 1.87 3.69 -16.53 2.39 58.49 1.57
```
Which four numbers are the 2D box? Would your Day 42 `iou` accept them as-is? (Yes — that's the point.)
Then grab the download when convenient and re-run step 1.

## Check yourself

- In a KITTI label line, which fields did you already build tools for in Day 42?
- Why start with the **mini/sample** split instead of the full dataset?
- KITTI stores labels as per-frame text files; NuScenes as a joined JSON database. Why does NuScenes need
  the more complex structure? (Hint: 6 cameras + tracking objects *across time*.)

## Where this shows up later

Week 8 turns "run a detector on a frame" into "evaluate it properly and compare experiments." Loading
real driving data and running perception on it is the concrete Stage-2 deliverable — and the closest
thing in academia to the problem openpilot solves on the road.

**Next:** Week 8 — Paper Reading & the Research On-ramp.
