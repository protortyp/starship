# Photoreal landing renders

Blender renders of a landing flown by a trained policy in the simulator: the exporter writes a
60 fps track, the scene builder turns it into a procedurally modelled Starship over a concrete
pad with volumetric plumes and dust, and Cycles renders it on a GPU.

## 1. Export a trajectory

```sh
uv run python -m starship.render.export --policy logs/final_run/policy_best.pt --stage 4 \
    --seed 0 --out out/landing.json --usd out/landing.usda
```

Flies one deterministic episode per seed on the CPU (`--tries` seeds until one lands), samples
every control step and resamples to `--fps` (cubic Hermite positions from the recorded
velocities, slerp attitudes, linear actuators). `out/landing.json` holds:

- `meta`: fps, frame count, duration, final status (`landed`), policy, stage, seed;
- `vehicle`: static geometry in body coordinates (hull profile, engine exits and gimbal pivots,
  flap hinges/axes/spans, pad radius) so the renderer needs nothing from the simulator;
- `frames`: columns, one row per frame: `time`, `position` (world position of the engine-plane
  origin), `quaternion` (x, y, z, w, body to world), `flaps` (4, rad), `spool`, `throttle`,
  `gimbal2`, `gimbal3`, `thrust` (3 each), `altitude`, `velocity`, `speed`, `status`.

Frames follow `starship.sim.config`: world z up with the pad at the origin; body b1 from the
engines to the nose, b2 starboard, b3 belly. The optional `.usda` is written without `pxr`
(time-sampled transforms, hull mesh, flap plates, engine anchors) and opens in usdview or Blender.

## 2. Render

```sh
blender -b -P tools/render/starship_scene.py -- --track out/landing.json \
    --mode still --time 41 --camera ground --resolution 3840x2160 --samples 512 \
    --out out/render/hero_ground.png
blender -b -P tools/render/starship_scene.py -- --track out/landing.json \
    --mode clip --start 38 --end 42 --clip-fps 30 --resolution 1920x1080 --samples 64 \
    --out out/render/clip
```

Options: `--camera ground|pad|aerial|wide`, `--camera-azimuth auto|deg` (`auto` places the
camera `--camera-offset` degrees from the belly direction so the tiled and steel halves both
show), `--sun-offset`, `--hdri <polyhaven id>`, `--engines sim|all` (`all` lights all three
Raptors at the commanded throttle; the policy lands on two), `--device OPTIX|CUDA|METAL|CPU`,
`--preview` (coarse volumes, no depth of field), `--exposure`, `--haze`, `--max-tiles`,
`--save-blend file.blend`, `--mode blend` (build and save only). Poly Haven assets download on
first use into `--assets` (default `out/render/assets`).

What is built: hull as a surface of revolution of the simulator's profile with ring welds,
plate seams, heat tint, oil-canning and soot in the shader; ~9.5k hexagonal heat tiles on the
windward sector of hull and nose plus ~1.3k on the flap faces (per-tile colour, tilt, a few
missing); flaps with bevelled slabs and hinge fairings; the open skirt with aft dome, three
gimballing sea-level Raptors (double-walled bells, regen ribs, powerheads, interior glow) and
three vacuum bells; volumetric plumes with shock diamonds and 4D-noise turbulence driven by the
per-engine thrust; an impingement splash volume and a dust ring that starts when a lit plume
gets within 42 m of the pad and grows with time; worn concrete pad with scorch and markings,
coastal terrain, ocean, light masts, tanks; HDRI sun matched to the camera, compositor fog glow
and mist haze; AgX view transform, motion blur (180 degree shutter at the output rate), depth of
field, adaptive sampling, OptiX/OIDN denoising.

Tested with Blender 4.5 LTS. A 1280x720 preview at 32 spp takes ~30 s on an Apple M-series
GPU; 4K at 512 spp takes minutes on an RTX 4090 (see the log in `out/render/`).

## 3. Render on vast.ai

```sh
just vast --name render up --gpu RTX_4090                 # rent, push code, install the project
just vast --name render exec bash tools/render/remote.sh install   # Blender 4.5 LTS + ffmpeg
scp -i ~/.ssh/id_ed25519 -o IdentitiesOnly=yes -o IdentityAgent=none -P <port> \
    out/landing.json root@<host>:/root/starship/out/        # out/ is not rsynced by push
just vast --name render exec bash tools/render/remote.sh render --track out/landing.json \
    --time 41 --camera ground --resolution 3840x2160 --samples 512 --out out/render/hero.png
just vast --name render exec bash tools/render/remote.sh encode out/render/clip out/render/clip.mp4 30
rsync -az -e "ssh -i ~/.ssh/id_ed25519 -o IdentitiesOnly=yes -o IdentityAgent=none -p <port>" \
    root@<host>:/root/starship/out/render/ out/render/
just vast --name render down
```

Host and port are in `.vast/render.json`. Long renders are better started inside `tmux` on the
instance so a dropped SSH connection cannot kill them.

## Asset licenses

All downloaded assets are CC0 1.0 (public domain) from [Poly Haven](https://polyhaven.com):
`belfast_sunset_puresky` and `qwantani_dusk_2_puresky` (HDRI skies),
`concrete_floor_worn_001` (pad), `coast_sand_rocks_02` (terrain). A `LICENSE.txt` is written
next to each download. Everything else in the scene is generated by the script. Blender is
GPL; the renders themselves carry no license obligations from it.
