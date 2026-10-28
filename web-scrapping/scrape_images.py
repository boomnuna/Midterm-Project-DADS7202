"""
scrape_images.py

Collects raw candidate images per class for the One Piece "Devil Fruits"
fine-grained classification project — Gomu Gomu no Mi, Mera Mera no Mi,
Ope Ope no Mi, Yami Yami no Mi.

WHY better_bing_image_downloader (not icrawler):
  icrawler's GoogleImageCrawler and BingImageCrawler are both broken as
  of testing — Google crashes outright (JS-only page, no crawler can
  read it without a real browser), and Bing's crawler runs without
  crashing but returns unrelated images because its HTML parser is out
  of date against Bing's current layout.

  This script uses `better_bing_image_downloader` instead — actively
  maintained, hits Bing's JSON API directly instead of scraping rendered
  HTML. Bing is used as the ONLY engine; it already yields near-full
  results on its own (often 85-90+ out of a 90+ limit per query).

*** READ THIS BEFORE RUNNING — THE #1 RISK FOR THIS DATASET ***
  This project classifies the FRUITS THEMSELVES — the swirled, spiky
  orb/fruit object shown in a jar, in someone's hand, or on a plate
  right before it's eaten — NOT the character using the resulting
  power. A screenshot of Luffy stretching his arm is NOT a Gomu Gomu
  no Mi image; it doesn't show the fruit at all. Mixing "power in use"
  screenshots into these folders defeats the whole point of the project
  and WILL need to be manually removed, so it's worth being deliberate
  about query wording now rather than cleaning it up later.

  Every query below is phrased to bias toward "the fruit object" (e.g.
  "devil fruit design", "fruit prop", "close up fruit"), but Bing's
  image search still cannot perfectly separate "fruit" from "power" —
  expect a real amount of manual filtering regardless of query wording.

WHY THIS SET IS TRICKY (harder than Iron Man armor or Luffy's Gears):
  - Unlike Luffy's Gears (1000+ episodes of footage) or Iron Man suits
    (major theatrical releases), each devil fruit's canon appearance AS
    A FRUIT is shown on-screen only a handful of times total across the
    whole series (usually just the "eating" flashback/scene). The bulk
    of what queries will surface is necessarily fan art, wiki reference
    renders, and merchandise/replica photos — NOT anime screenshots.
    Budget more time for manual review than either of those two prior
    projects, and don't be surprised if raw per-class counts end up
    lower even after adding more query variety.
  - Fan art of the same fruit tends to converge on the same wiki/official
    reference image as its source, so NEAR-DUPLICATE risk is higher
    here than in the character/armor/gear projects — lean harder on
    dedupe_and_clean.py's near-duplicate flagging step for this dataset.
  - The added "figure"/レプリカ (merchandise) queries pull from Japanese
    figure/toy retail sites (AmiAmi, Mandarake, etc.) — a different
    source pool than the English wiki/fan-art queries, which should
    raise the raw candidate count. But product photos get re-hosted
    near-identically across multiple retailer/reseller listings (same
    studio background, same angle, just re-cropped), so expect EVEN
    MORE near-duplicates from these queries specifically. Also watch
    for figures taking artistic liberties with color (chibi/keychain
    versions especially) — since color is the main discriminative cue
    here, a mis-colored figure photo can quietly contaminate a class.
  - Devil fruits share a common visual "template" (round fruit body +
    spiral/swirl pattern + stem leaves) — the four below were chosen
    specifically because they differ in COLOR, which is the main cue
    that will actually separate them:
      gomu_gomu: red/pink round fruit with dark swirls
      mera_mera: orange/red fruit, flame-associated color scheme
      ope_ope:   blue/purple fruit with heart-shaped swirl pattern
      yami_yami: black/dark purple fruit — the clearest outlier by color
    If ColorJitter augmentation is used heavily during training, revisit
    that choice for this dataset — color is doing more of the
    classification work here than it did for the armor or gear projects.
  - Manga (black-and-white) results will slip into searches sometimes —
    decide upfront whether you want a color-only dataset (recommended,
    since color is the primary discriminative cue here) and filter
    those out during manual review.
  - Official "canon" fruit designs occasionally get redrawn slightly
    between manga, anime, and games (Pirate Warriors, etc.) — minor
    version differences are expected and fine to keep; they're not the
    same problem as the power-vs-fruit mixup above.

IMPORTANT — this script only DOWNLOADS CANDIDATES. You must still:
  1. Manually review EVERY image — for this dataset specifically, the
     primary thing you're checking for is "is this actually a picture
     of the fruit itself, or did a power-in-use / character image slip
     through" (see warning above), not just relevance in general
  2. Decide on and enforce a manga-vs-anime-vs-fanart-vs-merch policy
     consistently across all four classes, not just some
  3. Run dedupe_and_clean.py afterward and pay close attention to the
     near-duplicate flags — expect more near-duplicates here than in
     prior scrapes, since fan art commonly copies the same reference image
  4. Keep a record of what search terms/sources you used per class,
     for the "Dataset description" section of your presentation
  5. If any class ends up noticeably smaller than the rest after
     cleaning (likely, given how little canon fruit footage exists),
     prioritize adding to it specifically rather than adding equally
     to every class

INSTALL:
    pip install better-bing-image-downloader

Usage:
    python scrape_images.py
"""

import os
import shutil
from better_bing_image_downloader import downloader

# ---- EDIT THIS: one or more search queries per class ----
CLASSES = {
    "gomu_gomu": [
        "Gomu Gomu no Mi devil fruit design",
        "Gomu Gomu no Mi fruit One Piece close up",
        "Luffy devil fruit before eating anime",
        "Gomu Gomu no Mi fruit prop replica",
        "Gomu Gomu no Mi fruit wiki reference art",
        "One Piece rubber fruit swirl pattern red",
        "Gomu Gomu no Mi devil fruit figure",
        "ゴムゴムの実 フィギュア",
        "Gomu Gomu devil fruit replica",
    ],
    "mera_mera": [
        "Mera Mera no Mi devil fruit design",
        "Mera Mera no Mi fruit One Piece close up",
        "Ace devil fruit before eating anime",
        "Mera Mera no Mi fruit prop replica",
        "Mera Mera no Mi fruit wiki reference art",
        "One Piece flame fruit swirl pattern orange",
        "Mera Mera no Mi devil fruit figure",
        "メラメラの実 フィギュア",
        "Mera Mera devil fruit replica",
    ],
    "ope_ope": [
        "Ope Ope no Mi devil fruit design",
        "Ope Ope no Mi fruit One Piece close up",
        "Law devil fruit before eating anime",
        "Ope Ope no Mi fruit prop replica",
        "Ope Ope no Mi fruit wiki reference art",
        "One Piece heart fruit swirl pattern blue",
        "Ope Ope no Mi devil fruit figure",
        "オペオペの実 フィギュア",
        "Ope Ope devil fruit replica",
    ],
    "yami_yami": [
        "Yami Yami no Mi devil fruit design",
        "Yami Yami no Mi fruit One Piece close up",
        "Blackbeard devil fruit before eating anime",
        "Yami Yami no Mi fruit prop replica",
        "Yami Yami no Mi fruit wiki reference art",
        "One Piece darkness fruit swirl pattern black",
        "Yami Yami no Mi devil fruit figure",
        "ヤミヤミの実 フィギュア",
        "Yami Yami devil fruit replica",
    ],
}

# Raised from a smaller default -> 130 to increase the raw pool per class.
# Bing has been yielding near-full results in testing (often 95%+ of the
# limit) for the character/armor/gear projects, but expect LOWER yield
# here specifically — canon "fruit object" images are much rarer than
# character or armor images, so a higher ceiling matters more this time.
IMAGES_PER_QUERY = 130           # per query
OUTPUT_ROOT = "raw_dataset"      # final images land in raw_dataset/<class_name>/
TEMP_ROOT = "_bbid_tmp"          # scratch space, cleaned up as we go
ENGINE = "bing"                  # Bing only — see note above


def scrape_class(class_name: str, queries: list[str]) -> int:
    class_dir = os.path.join(OUTPUT_ROOT, class_name)
    os.makedirs(class_dir, exist_ok=True)

    counter = 0
    for query in queries:
        print(f"[{class_name}] (bing) searching: '{query}'")
        try:
            downloader(
                query=query,
                limit=IMAGES_PER_QUERY,
                output_dir=TEMP_ROOT,
                engine=ENGINE,
                max_workers=8,
                verbose=False,
            )
        except Exception as e:
            # Don't let one bad query kill the whole run.
            print(f"  !! bing failed for '{query}': {e}")
            continue

        # better_bing_image_downloader saves into TEMP_ROOT/<query>/ —
        # move those files into the class folder with unique names,
        # then clean up the per-query temp folder.
        query_subdir = os.path.join(TEMP_ROOT, query)
        if not os.path.isdir(query_subdir):
            print(f"  (no results folder for '{query}' — skipping)")
            continue

        moved = 0
        for fname in os.listdir(query_subdir):
            src = os.path.join(query_subdir, fname)
            if not os.path.isfile(src):
                continue
            if fname.lower().endswith((".json", ".jsonl")):
                continue  # skip manifest files, keep only images
            ext = os.path.splitext(fname)[1] or ".jpg"
            dst = os.path.join(class_dir, f"{class_name}_{counter:05d}{ext}")
            shutil.move(src, dst)
            counter += 1
            moved += 1
        shutil.rmtree(query_subdir, ignore_errors=True)
        print(f"  -> moved {moved} images into {class_dir}")

    return counter


def main():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    os.makedirs(TEMP_ROOT, exist_ok=True)

    totals = {}
    for class_name, queries in CLASSES.items():
        totals[class_name] = scrape_class(class_name, queries)

    shutil.rmtree(TEMP_ROOT, ignore_errors=True)

    print("\nDone. Raw candidates (Bing) are in:", OUTPUT_ROOT)
    print("\nImages collected per class (before cleaning):")
    for class_name, count in totals.items():
        print(f"  {class_name}: {count}")

    print("\nNEXT STEPS (do not skip these):")
    print("  1. Manually review every image — the #1 thing to check is")
    print("     whether it actually shows the FRUIT ITSELF, not a")
    print("     character using the resulting power (a stretched arm,")
    print("     a flame attack, a black-goo attack, etc. is NOT a valid")
    print("     image for this project no matter how relevant it looks)")
    print("  2. Run dedupe_and_clean.py to remove exact/near duplicates")
    print("     — expect MORE near-duplicate flags here than in a")
    print("     character/armor dataset, since fan art of the same fruit")
    print("     commonly copies the same wiki/official reference image")
    print("  3. Decide on a manga-vs-anime-vs-fanart-vs-merch policy and")
    print("     apply it consistently across all FOUR classes, not just some")
    print("  4. If any class ends up noticeably smaller than the rest after")
    print("     cleaning (expected, since canon fruit footage is scarce),")
    print("     prioritize adding to it specifically rather than adding")
    print("     equally to every class")


if __name__ == "__main__":
    main()
