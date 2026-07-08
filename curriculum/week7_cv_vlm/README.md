# Week 7 — Computer Vision & Vision-Language Models

> *Goal: the advisor's **Stage 2** — get hands-on with the open-source models the lab actually uses.
> Understand detection and segmentation, run YOLO / SAM / Grounding DINO, and load driving datasets
> (KITTI, NuScenes). This is where Weeks 5–6 pay off.*

You can now train small nets (Week 5) and run other people's code reproducibly (Week 6). This week you
put both to work on real computer-vision models — the exact ones in your advisor's roadmap.

| Day | Topic | Format | You'll be able to | Status |
|-----|-------|--------|-------------------|--------|
| 42 | **Detection fundamentals: boxes, IoU & NMS** | pytest | Compute IoU, run non-max suppression, convert box formats | ✅ |
| 43 | **YOLO: running a real detector** | guided | Clone YOLO, run inference, read its boxes/scores/labels | ✅ |
| 44 | **Segmentation with SAM 2 / SAM 3** | guided | Prompt SAM with a point/box, get masks, understand the output | ✅ |
| 45 | **Open-vocabulary detection: Grounding DINO** | guided | Detect objects from a *text prompt*, not a fixed class list | ✅ |
| 46 | **CLIP & vision-language models** | pytest | Match images to text with embeddings + cosine similarity | ✅ |
| 47 | **Datasets: KITTI & NuScenes** | guided | Understand driving-dataset formats; load labels for inference | ✅ |

**Format:** the *algorithms* every detector shares (IoU, NMS, cosine-similarity retrieval) are gradable
pure-numpy lessons — you build them (Days 42, 46). The *big pretrained models* (YOLO, SAM, Grounding
DINO) and *datasets* are **guided** hands-on lessons — you clone, set up, and run them, applying the
Day-40 reproduction checklist. Running these well is much easier with a GPU, but the concepts and
small-scale runs work on CPU too.

**The through-line to openpilot:** detection, segmentation, and multi-object tracking on driving data
*is* the perception problem openpilot solves. KITTI and NuScenes are the academic version of the data
a comma device sees.

**Next:** Week 8 — Paper Reading & the Research On-ramp.
