#!/bin/bash
# Batch LTX generation over MG3/MG2 demo images with per-scene tailored prompts.
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

mkdir -p output_ltx/batch_logs

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
    > "output_ltx/batch_logs/$name.log" 2>&1
  echo "DONE $name rc=$? $(date +%H:%M:%S)"
}

run mg3_002 demo_images/002/image.png "704*1280" \
  "A stylized low-poly desert canyon settlement built into red rock cliffs, with wooden rope bridges, ladders, palm trees and a tall green cactus. The camera dollies forward slowly along the sandy path toward the cliff dwellings. Bright midday sun, soft flat shadows, scattered clouds in a blue sky, vibrant flat-shaded game art style."

run mg3_003 demo_images/003/image.png "704*1280" \
  "A stylized low-poly tropical jungle clearing with a massive faceted boulder, palm and banana trees, and a small thatched hut beside a sandy path. The camera glides forward along the path, passing the boulder and heading into the dense foliage. Warm sunlight filters through the leaves, soft cloud shadows drift across the ground, purple-blue sky, vibrant low-poly game art."

run mg3_004 demo_images/004/image.png "704*1280" \
  "A stylized low-poly village graveyard on a grassy hill with weathered Celtic-cross gravestones under a large spreading tree, a rocky mountain with stone stairs rising behind. The camera pushes in slowly between the gravestones toward the mountain path. Crisp morning light, long soft shadows, clear blue sky, flat-shaded low-poly game aesthetic."

run mg3_005 demo_images/005/image.png "704*1280" \
  "A photorealistic cyberpunk alleyway at night: a concrete wall tagged with glowing pink graffiti, wet pavement reflecting neon light, steam pipes and a metal catwalk overhead. The camera tracks slowly sideways along the graffiti wall, revealing the dark rain-slick street beyond. Moody cinematic lighting, pink and teal neon palette, glistening wet surfaces, shallow depth of field."

run mg3_006 demo_images/006/image.png "704*1280" \
  "Ancient overgrown stone ruins bathed in warm sunlight: weathered masonry walls with arched niches, a tall tree growing through cracked flagstones, and a wide stone staircase. The camera moves slowly forward across the courtyard toward the stairs while leaves sway gently in the breeze. Dappled sunlight, drifting dust motes, photorealistic textures, serene atmosphere."

run mg3_007 demo_images/007/image.png "704*1280" \
  "A vibrant low-poly forest meadow with a dirt path curving past a red wooden fence, pine trees, gray boulders and scattered wildflowers. The camera dollies forward along the winding path into the colorful woods. Bright cheerful daylight, clear sky, saturated flat-shaded colors, stylized game environment."

run mg3_008 demo_images/008/image.png "704*1280" \
  "A stylized sci-fi spaceship corridor with beige paneled walls, glowing cyan strip lights, yellow-and-black hazard markings and an illuminated EXIT sign. The camera glides steadily forward down the corridor toward the far doorway. Clean flat-shaded game art, soft ambient lighting, subtly blinking indicator lights, quiet futuristic atmosphere."

run mg3_009 demo_images/009/image.png "704*1280" \
  "A photorealistic sunlit urban back alley: a yellow brick wall covered in colorful graffiti tags, a gray metal door, and overgrown weeds along cracked concrete slabs. The camera pushes in slowly toward the door while the tall grasses sway in a light breeze. Warm afternoon sunlight, sharp shadows, gritty realistic textures, cinematic composition."

run mg3_010 demo_images/010/image.png "704*1280" \
  "A stylized low-poly snowy landscape with snow-covered pine trees, a wooden fence and a stack of logs under a bright blue sky with a single puffy cloud. The camera dollies forward gently across the sparkling snowfield toward the forest. Crisp winter sunlight, long cool shadows, clean flat-shaded game art style."

run mg2_gta ../Matrix-Game-2/demo_images/gta_drive/0000.png "640*1280" \
  "Gameplay footage of a black supercar speeding down a sun-bleached desert highway past a liquor store and palm trees, dusty mountains shimmering in the heat haze. The chase camera follows close behind the car as it accelerates along the cracked asphalt. Hazy midday light, photorealistic open-world video game graphics, strong sense of speed."

run mg2_templerun ../Matrix-Game-2/demo_images/temple_run/0000.png "704*1152" \
  "Gameplay footage of an endless-runner adventure game: the camera sprints forward along a crumbling ancient stone walkway high above a sea of clouds, with mossy ruins and distant floating cliffs under an orange sunset sky. Fast continuous forward motion, dramatic golden light, mobile adventure game style."

run mg2_night_house ../Matrix-Game-2/demo_images/universal/0012.png "704*1280" \
  "A cinematic fantasy scene at night: a cozy timber-framed house with warm glowing windows beside a misty lake, under a starry sky with a huge full moon and the Milky Way. The camera slowly pans right across the lake toward the mountains as mist drifts over the water and fireflies float above the grass. Photorealistic, atmospheric, magical moonlit mood."

echo "BATCH COMPLETE $(date +%H:%M:%S)"
