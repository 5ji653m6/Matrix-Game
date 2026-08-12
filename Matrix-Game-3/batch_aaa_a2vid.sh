#!/bin/bash
# A2Vid coherent-audio regeneration of the AAA demo set.
# Same images/prompts/seed as batch_aaa_ltx.sh, but video is generated
# conditioned on the pre-generated per-scene soundtrack (--audio_track),
# so audio is one continuous track and motion syncs to it.
#
# Single-GPU A2Vid (no mgpu variant) -> 4-way parallel across GPUs 0-3,
# ~15-16 min per video, 3 rounds ~= 50 min wall-clock.
# aaa_forza is skipped: its a2vid output already exists (validation run,
# identical params) at output_ltx/aaa_forza_a2vid.mp4.
set -u
cd /data1/Matrix-Game/Matrix-Game-3

PY=/data1/ltx-world-model/.venv/bin/python
export LTX_ROOT=/data1/LTX-2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p output_ltx/batch_logs_aaa_a2vid

run() {
  local name="$1" image="$2" prompt="$3"
  echo "START $name gpu=$CUDA_VISIBLE_DEVICES $(date +%H:%M:%S)"
  "$PY" generate_ltx.py \
    --image "$image" \
    --prompt "$prompt" \
    --audio_track "output_ltx/${name}_soundtrack.wav" \
    --size 704*1280 --num_iterations 12 --seed 42 \
    --output_dir ./output_ltx --save_name "${name}_a2vid" \
    > "output_ltx/batch_logs_aaa_a2vid/${name}.log" 2>&1
  echo "DONE $name rc=$? $(date +%H:%M:%S)"
}

worker0() {
  run aaa_cyberpunk demo_images_aaa/cyberpunk2077/image.jpg \
    "Photorealistic cyberpunk garage workshop at night: a low wedge-shaped concept sports car in yellow, white and blue parked on an oil-stained concrete floor, a mechanic crouched by the rear wheel, tires and tool chests scattered around. The camera slowly dollies toward the car as dust drifts through harsh overhead work lights and steam curls from a floor grate. Moody neon-noir atmosphere, wet reflective floor, cinematic depth of field, AAA game cinematic."
  run aaa_rdr2 demo_images_aaa/rdr2/image.jpg \
    "Two weathered cowboys in heavy winter ponchos lead their horses on foot up a snowy mountain trail, snow-capped peaks and frosted pines fading into white haze behind them. The camera follows slowly behind the pair as snow falls and wind whips powder off the drifts. Soft overcast winter light, breath fogging in the cold air, photorealistic western frontier, cinematic and quiet."
  run aaa_elden_ring demo_images_aaa/elden_ring/image.jpg \
    "An armored knight on horseback stands on a grassy cliff edge overlooking vast misty ruins, while a colossal glowing golden tree of light dominates the sky, its luminous branches raining drifting golden sparks. The camera slowly pushes in past the knight toward the radiant tree as banners and leaves stir in the wind. Epic dark fantasy scale, volumetric golden-green light, painterly AAA game art."
}

worker1() {
  run aaa_wukong demo_images_aaa/blackmyth_wukong/image.jpg \
    "A monkey warrior in fur-trimmed robes faces a massive smoldering stone guardian inside a ruined mountain temple overgrown with moss and twisted roots, shafts of pale light cutting through the darkness. The camera pushes in slowly over the warrior's shoulder as embers and dust drift through the god rays. Mythic Chinese dark fantasy, dramatic chiaroscuro lighting, photorealistic detail."
  run aaa_gow demo_images_aaa/gow_ragnarok/image.jpg \
    "A muscular bald warrior with red tattoos seen from behind stands at a moss-covered cliff edge, looking across a misty Norse canyon toward a great wooden cage-lift suspended on chains between towering rock walls. The camera drifts slowly forward past his shoulder as mist rolls through the gorge and water drips from hanging moss. Cold overcast light, photorealistic mythic wilderness, cinematic atmosphere."
  run aaa_horizon demo_images_aaa/horizon_fw/image.jpg \
    "Inside a vast overgrown ruin of an ancient civilization: towering carved stone pillars wrapped in vines and roots, brilliant shafts of sunlight piercing the broken ceiling, tiny figures crossing a grand staircase below. The camera glides slowly forward into the hall as dust motes swirl in the god rays and leaves flutter down. Lush post-apocalyptic jungle temple, vibrant greens against warm stone, photorealistic AAA game render."
}

worker2() {
  run aaa_witcher demo_images_aaa/witcher3/image.jpg \
    "A white-haired swordsman on horseback pauses on a rocky hilltop trail overlooking a sweeping alpine valley: a lakeside village with red rooftops far below, pine forests, and a wall of snow-capped mountains under a bright sky with circling birds. The camera slowly pans left across the vista as grass and cloaks ripple in the mountain wind. Painterly medieval fantasy, crisp daylight, epic open-world scale."
  run aaa_tsushima demo_images_aaa/ghost_tsushima/image.jpg \
    "A lone samurai in a straw raincoat stands on a rocky bluff beneath a brilliant golden ginkgo tree, overlooking a mist-filled valley of forests and distant mountains, a thin column of smoke rising far away. The camera drifts slowly around the samurai as golden leaves swirl down through the sunlight and fog shifts in the valley below. Feudal Japan, breathtaking golden-hour light, cinematic AAA game beauty."
  run aaa_starfield demo_images_aaa/starfield/image.jpg \
    "An astronaut in a white and red spacesuit stands on the rocky ridge of an alien world, gazing at a colossal ringed planet hanging in a pale rose sky above jagged dark mountains and red crystal growths. The camera pushes in slowly past the astronaut's shoulder as fine dust drifts across the regolith. Hard sci-fi realism, serene and vast, cinematic planetary exploration."
}

worker3() {
  run aaa_deathstranding demo_images_aaa/death_stranding/image.jpg \
    "A lone porter in a blue-grey expedition suit with a towering stack of orange cargo cases strapped to his back climbs a windswept mossy hillside toward a colossal half-buried ring-shaped ruin overgrown with grass. The camera follows steadily behind him as clouds drift over the green slopes and grass ripples in waves. Melancholic photorealistic wilderness, soft diffused daylight, quiet determination."
  run aaa_gta5 demo_images_aaa/gta5/image.jpg \
    "An orange muscle car with glowing blue underglow speeds through a rain-soaked downtown intersection at night, neon signs and skyscraper lights blurring into bokeh, reflections streaking across the wet pavement. The chase camera swings low behind the car as it powers through a drift, tire spray catching the streetlights. Photorealistic open-world crime game footage, fast and cinematic."
}

echo "=== A2VID AAA BATCH START $(date +%H:%M:%S) ==="
CUDA_VISIBLE_DEVICES=0 worker0 &
CUDA_VISIBLE_DEVICES=1 worker1 &
CUDA_VISIBLE_DEVICES=2 worker2 &
CUDA_VISIBLE_DEVICES=3 worker3 &
wait
echo "=== A2VID AAA BATCH ALL DONE $(date +%H:%M:%S) ==="
