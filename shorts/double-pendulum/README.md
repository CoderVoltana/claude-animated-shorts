# Double Pendulum Chaos (YouTube Short)

A 24 s vertical science short (1080x1920, 60 fps) on sensitive dependence on initial conditions.
It has a neon-on-black look and loops seamlessly.

## Timeline
| Time | Beat | On screen |
|---|---|---|
| 0–2 s | Hook | Pendulum released from rest. "CAN YOU PREDICT THIS?" |
| 2–6 s | The system | "THE DOUBLE PENDULUM": two arms, two hinges, only gravity, with θ₁/θ₂ angle overlays. "Zero randomness." |
| 6–17 s | Build and suspense | "This isn't one pendulum." / "It's 500." Each starts 0.000000001° apart. A magnifier follows the tip (zoom ×10⁴ → ×1), and a live log-scale graph shows the spread growing. "Still moving as one…" / "…for now." Heartbeat and riser. |
| 17–22 s | Reveal | Flash and shockwave. The faint cyan → blue → purple ensemble bursts apart while 5 tracked members (orange, yellow, green, pink, white) trace bright paths on top. "CHAOS. Deterministic. Yet unpredictable." A legend gives each tracked member's starting θ₂. |
| 22.2–24 s | Loop | Time runs backward (the equations are time-reversible). The 500 reunite into one pendulum, and the last frame leads into frame 0. |

## Science
- `sim.py` integrates the exact equations of motion for two point masses (1 kg) on massless rigid 1 m rods with frictionless hinges and g = 9.81 m/s². It uses classical RK4 in float64 with h = 1/2400 s.
- Start: θ₁ = 140°, θ₂ = 175° + i · 10⁻⁹° for i = 0…499, both at rest.
- Tracked members 50, 150, 250, 350 and 450 (θ₂ = 175.00000005° … 175.00000045°) are five different initial conditions under the same equations. They're drawn last, with thicker, brighter trails.
- Energy drift over the run is about 3·10⁻⁹ J.
- Everything shown is real output:
  - The motion, the separation, and the graph (spread of θ₂ across the ensemble).
  - The zoom factor (the magnification needed to see the tips apart).
- The graph shows roughly straight-line growth on a log scale. That is the signature of a positive Lyapunov exponent.
- `sim.bin` stores the reference member in float64 plus per-member offsets in float32. This keeps 10⁻⁹° differences exact through rendering.

## Build
```bash
python3 export.py                      # physics -> sim.bin + sim.json  (~25 s)
python3 audio.py                       # original synthesized score -> score.wav
node render.mjs frames <dir> 4         # 1,440 PNG frames via headless Chromium canvas
ffmpeg -framerate 60 -i <dir>/f%05d.png -i score.wav -af loudnorm=I=-14:TP=-1.5 \
  -c:v libx264 -preset slow -crf 15 -pix_fmt yuv420p -c:a aac -b:a 256k -shortest double_pendulum_chaos.mp4
```
Preview any moment in a browser: serve this folder and open `render.html?t=17.5`.
