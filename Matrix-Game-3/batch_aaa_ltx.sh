#!/bin/bash
# Batch LTX generation over AAA game screenshots (Steam official, 1920x1080)
# with per-scene tailored cinematic prompts.
# Sequential runs on GPUs 0-3 (mgpu mode). ~7 min per video (model load + 12 iterations).
set -u
cd /data1/Matrix-Game/Matrix-Game-3

CKPT=/data1/models/Lightricks--LTX-2.3/snapshots/master/ltx-2.3-22b-distilled-1.1.safetensors
UPS=/data1/models/Lightricks--LTX-2.3/snapshots/master/ltx-2.3-spatial-upscaler-x2-1.1.safetensors
GEMMA=/data1/models/google--gemma-3-12b-it/snapshots/master
PY=/data1/ltx-world-model/.venv/bin/python

export CUDA_VISIBLE_DEVICES=0,1,2,3
export LTX_ROOT=/data1/LTX-2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p output_ltx/batch_logs_aaa

run() {
  local name="$1" image="$2" size="$3" prompt="$4"
  echo "START $name $(date +%H:%M:%S)"
  "$PY" generate_ltx.py --mgpu \
    --image "$image" \
    --prompt "$prompt" \
    --ltx_checkpoint "$CKPT" \
    --spatial_upsampler "$UPS" \
    --gemma_root "$GEMMA" \
    --size "$size" --num_iterations 12 --seed 42 \
    --output_dir ./output_ltx --save_name "$name" \
    > "output_ltx/batch_logs_aaa/$name.log" 2>&1
  echo "DONE $name rc=$? $(date +%H:%M:%S)"
}

run aaa_cyberpunk demo_images_aaa/cyberpunk2077/image.jpg "704*1280" \
  "Photorealistic cyberpunk garage workshop at night: a low wedge-shaped concept sports car in yellow, white and blue parked on an oil-stained concrete floor, a mechanic crouched by the rear wheel, tires and tool chests scattered around. The camera slowly dollies toward the car as dust drifts through harsh overhead work lights and steam curls from a floor grate. Moody neon-noir atmosphere, wet reflective floor, cinematic depth of field, AAA game cinematic."

run aaa_rdr2 demo_images_aaa/rdr2/image.jpg "704*1280" \
  "Two weathered cowboys in heavy winter ponchos lead their horses on foot up a snowy mountain trail, snow-capped peaks and frosted pines fading into white haze behind them. The camera follows slowly behind the pair as snow falls and wind whips powder off the drifts. Soft overcast winter light, breath fogging in the cold air, photorealistic western frontier, cinematic and quiet."

run aaa_elden_ring demo_images_aaa/elden_ring/image.jpg "704*1280" \
  "An armored knight on horseback stands on a grassy cliff edge overlooking vast misty ruins, while a colossal glowing golden tree of light dominates the sky, its luminous branches raining drifting golden sparks. The camera slowly pushes in past the knight toward the radiant tree as banners and leaves stir in the wind. Epic dark fantasy scale, volumetric golden-green light, painterly AAA game art."

run aaa_wukong demo_images_aaa/blackmyth_wukong/image.jpg "704*1280" \
  "A monkey warrior in fur-trimmed robes faces a massive smoldering stone guardian inside a ruined mountain temple overgrown with moss and twisted roots, shafts of pale light cutting through the darkness. The camera pushes in slowly over the warrior's shoulder as embers and dust drift through the god rays. Mythic Chinese dark fantasy, dramatic chiaroscuro lighting, photorealistic detail."

run aaa_gow demo_images_aaa/gow_ragnarok/image.jpg "704*1280" \
  "A muscular bald warrior with red tattoos seen from behind stands at a moss-covered cliff edge, looking across a misty Norse canyon toward a great wooden cage-lift suspended on chains between towering rock walls. The camera drifts slowly forward past his shoulder as mist rolls through the gorge and water drips from hanging moss. Cold overcast light, photorealistic mythic wilderness, cinematic atmosphere."

run aaa_horizon demo_images_aaa/horizon_fw/image.jpg "704*1280" \
  "Inside a vast overgrown ruin of an ancient civilization: towering carved stone pillars wrapped in vines and roots, brilliant shafts of sunlight piercing the broken ceiling, tiny figures crossing a grand staircase below. The camera glides slowly forward into the hall as dust motes swirl in the god rays and leaves flutter down. Lush post-apocalyptic jungle temple, vibrant greens against warm stone, photorealistic AAA game render."

run aaa_forza demo_images_aaa/forza5/image.jpg "704*1280" \
  "A silver supercar leads a night street race down a rain-slick desert highway, headlights carving through the storm, taillights and neon reflections smearing across the wet asphalt, giant saguaro cacti silhouetted against flashes of lightning. The chase camera hugs the road close behind the cars as spray kicks up from the tires. High-speed photorealistic racing game footage, dramatic storm lighting, strong motion blur."

run aaa_witcher demo_images_aaa/witcher3/image.jpg "704*1280" \
  "A white-haired swordsman on horseback pauses on a rocky hilltop trail overlooking a sweeping alpine valley: a lakeside village with red rooftops far below, pine forests, and a wall of snow-capped mountains under a bright sky with circling birds. The camera slowly pans left across the vista as grass and cloaks ripple in the mountain wind. Painterly medieval fantasy, crisp daylight, epic open-world scale."

run aaa_tsushima demo_images_aaa/ghost_tsushima/image.jpg "704*1280" \
  "A lone samurai in a straw raincoat stands on a rocky bluff beneath a brilliant golden ginkgo tree, overlooking a mist-filled valley of forests and distant mountains, a thin column of smoke rising far away. The camera drifts slowly around the samurai as golden leaves swirl down through the sunlight and fog shifts in the valley below. Feudal Japan, breathtaking golden-hour light, cinematic AAA game beauty."

run aaa_starfield demo_images_aaa/starfield/image.jpg "704*1280" \
  "An astronaut in a white and red spacesuit stands on the rocky ridge of an alien world, gazing at a colossal ringed planet hanging in a pale rose sky above jagged dark mountains and red crystal growths. The camera pushes in slowly past the astronaut's shoulder as fine dust drifts across the regolith. Hard sci-fi realism, serene and vast, cinematic planetary exploration."

run aaa_deathstranding demo_images_aaa/death_stranding/image.jpg "704*1280" \
  "A lone porter in a blue-grey expedition suit with a towering stack of orange cargo cases strapped to his back climbs a windswept mossy hillside toward a colossal half-buried ring-shaped ruin overgrown with grass. The camera follows steadily behind him as clouds drift over the green slopes and grass ripples in waves. Melancholic photorealistic wilderness, soft diffused daylight, quiet determination."

run aaa_gta5 demo_images_aaa/gta5/image.jpg "704*1280" \
  "An orange muscle car with glowing blue underglow speeds through a rain-soaked downtown intersection at night, neon signs and skyscraper lights blurring into bokeh, reflections streaking across the wet pavement. The chase camera swings low behind the car as it powers through a drift, tire spray catching the streetlights. Photorealistic open-world crime game footage, fast and cinematic."

echo "BATCH COMPLETE $(date +%H:%M:%S)"
