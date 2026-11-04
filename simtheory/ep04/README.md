# SimTheory #04 — CAN FRICTION KILL CHAOS?

34 s vertical short (1440×2560, 60 fps). Two lanes, each with a pair of twin double pendulums 0.002° apart.
Left: no friction, chaos never ends. Right: joint friction drains the energy, the motion turns regular, and the twins re-merge.

## Pipeline (fully offline and deterministic)
```bash
python3 simulate.py                      # RK4 sim of all 4 pendulums + every measurement -> data/
python3 audio.py                         # synthesized soundtrack on the same clock -> data/audio.wav
python3 render.py /path/to/frames        # 2040 PNG frames (skia AA vectors + HDR additive + bloom + HUD)
./encode.sh /path/to/frames out          # main MP4 + Instagram MP4
python3 qa.py /path/to/frames out        # safe area, numbers vs log, loop frame, specs, loudness, stills
python3 make_log.py out                  # out/MEASUREMENT_LOG.md
python3 render.py --still 9.8 cover.png --cover   # cover source frame
```
Needs: numpy, scipy, opencv-python-headless, skia-python (+ libegl1), ffmpeg, Inter fonts.
`timeline.py` is the single video-time → sim-time map used by the renderer, HUD, captions and audio.
`physics.py` holds the equations of motion. See `out/MEASUREMENT_LOG.md` for the numbers and the note on why
dry joint friction and a labelled time-lapse were needed to fit the story into 34 s.

MP4s are not committed (over GitHub's 100 MB file limit); rebuild them with the commands above.
