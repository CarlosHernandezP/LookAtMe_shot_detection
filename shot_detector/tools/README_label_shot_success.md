# Shot Success/Fail Labeling

Tool: `label_shot_success.py` — OpenCV window, binary success/fail labels per shot clip.

## Bundle layout (after tar -xzf)

```
shot_clips_2026-05-20/
├── clips/                              # MP4 shot clips (1s pose window + 1.5s tail)
├── annotations/                        # annotation_*.csv from shot_annotations/
├── ball_trajectories/                  # updated ball trajectories (new detection model)
├── rackets/                            # racket detections per match/cam
├── label_shot_success.py               # this labeling tool
└── README.md                           # this file
```

## Setup (your laptop)

```bash
pip install opencv-python
```

## Run

```bash
python label_shot_success.py --clips-dir ./clips --labels-csv ./labels.csv
```

By default, `_idle_*.mp4` clips are skipped (not real shots). Pass `--include-idle` if you want them too.

## Keys

| Key | Action |
|-----|--------|
| `y` / `1` | label success |
| `n` / `0` | label fail |
| `space` | skip (no label, revisit later) |
| `b` | back (drop last label, re-label previous clip) |
| `r` | restart current clip from frame 0 |
| `+` / `-` | speed up / slow down playback |
| `q` / `ESC` | quit (progress is already saved) |

## Output

`labels.csv`:

```
clip_filename,label,timestamp
0529b769-..._56849_serve_bottom.mp4,success,2026-05-20T19:42:11Z
0529b769-..._57147_serve_left.mp4,fail,2026-05-20T19:42:18Z
...
```

Labels are appended incrementally. Re-running the tool resumes from where you left off — already-labeled clips are skipped.

## Notes

- Clips are 75 frames @ 30 fps = 2.5s total. Pose CSV covers only the first 30 frames (shot window). Tail is for visual confirmation of outcome (ball lands in/out, opponent reaction).
- Clip filename schema: `<video_name>_<center_frame>_<shot_type>_<player_label>.mp4`. `center_frame` is the original annotated shot frame in the source video.
