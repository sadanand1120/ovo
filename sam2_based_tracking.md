Q) how to add sam2 based tracking to our code? Currently i have that instance tracker class. Keep good properties from it.

A)

* **Seed source** on some frames:

  * `GT masks` if `--use-inst-gt`
  * otherwise `SAM AMG masks`
* **Tracker** on the other frames:

  * `SAM2.1 large`
* **Global identity memory**:

  * your 3D map / global point labels

## The simple cadence

This is the cadence I’d use:

* **Every frame**: run **SAM2.1 tracking** for the current active instances.
* **Every seed frame**: run **GT/AMG once on that frame**, then use those masks to:

  * recover / refresh lost tracks,
  * add truly new instances,
  * optionally merge duplicates.

A very natural choice is:

* **seed frames = mapping frames** (`t % map_every == 0`)
* **track-only frames = all other frames**

So:

* on frame `t`: always track active objects forward
* if `t` is also a seed frame: after tracking, run GT/AMG on that frame and use it to correct / add / merge

That gives you exactly one heavy per-frame mask stage only every `map_every` frames, while the in-between frames are just tracking.

## What gets tracked and when to drop it

Do **not** keep tracking every historical instance forever.

Keep only an **active set**:

* instances that were seen recently and are still plausibly in view

If an instance goes out of view or tracking is bad for a few frames, **drop it from the SAM2.1 active tracker**, but **do not delete its global id** if it was already a real mapped object.

So there are two notions:

* **track active/inactive**
* **global instance alive/dead**

That distinction keeps things clean.

My recommendation:

* if a **confirmed** object leaves view or misses tracking for `K_miss` frames:

  * remove it from SAM2.1 active tracking
  * mark it **inactive**
  * keep its global id and points in the map
* if it later reappears on a seed frame:

  * GT/AMG mask overlaps its old projected 3D support
  * reactivate that old global id instead of birthing a new one

For **tentative** newborns, if they don’t stabilize quickly, just kill them.

---

## The four phases

### 1) Birth

Birth happens **only on seed frames**.

Pipeline on a seed frame:

1. First, run SAM2.1 tracking for the already-active objects.
2. Then run **GT/AMG** on the current frame.
3. Throw away GT/AMG masks that are already well explained by a good tracked mask.
4. For each remaining seed mask:

   * if it matches an existing global instance strongly through projected old 3D points, **reuse that old gid**
   * otherwise, if it has enough new support, **create a new tentative gid**

So the birth rule is:

**new gid only if this seed mask is not already explained by an old instance and it owns enough new geometry**

That keeps birth conservative.

### 2) Track

Tracking happens **every frame**.

For each active gid:

1. SAM2.1 propagates its mask from frame `j` to frame `j+1`
2. validate that tracked mask against projected 3D support of that gid
3. if valid:

   * accept it
   * use it to label pixels / new points on this frame
4. if invalid:

   * increment miss counter
   * if too many misses, drop it from active tracking

The key point:

**SAM2.1 gives short-term mask continuity; the 3D map decides identity.**

### 3) Merge

Merge should happen **only on seed frames**, not every frame.

Why? Because seed frames are when you have a fresh GT/AMG mask that gives a stronger object-level proposal than a propagated track mask.

Merge rule:

* if two global ids repeatedly correspond to the **same seed object**
* and one of them contributes almost no unique support compared to the other
* then merge the smaller/newer one into the larger/older one

Parent selection should stay simple:

* prefer **confirmed** over tentative
* then **older**
* then **more points**

Most duplicates should actually be prevented earlier by the **attach instead of birth** rule. Merge is just cleanup.

### 4) Death

There are really two deaths.

#### Track death

A tracked object is removed from the SAM2.1 active set if:

* it is out of view for a while, or
* tracking quality is bad for `K_miss` frames

For a confirmed object, this is **not** global death. It just becomes inactive.

#### Instance death

Only **tentative** instances should really die.

Kill a tentative gid if:

* it never accumulates enough points,
* it keeps failing tracking validation,
* or on seed frames it keeps getting explained by another existing gid

That’s how you kill oversegmented / bogus fragments cleanly.

So the clean policy is:

* **tentative objects die easily**
* **confirmed objects do not die easily; they just go inactive**
* **duplicates are resolved by merge**

---

## The actual recipe

Here’s the simplest version I’d recommend:

1. Maintain:

   * `point_gid` for each global 3D point
   * `instances[gid]` with status and counters
   * `active_gids` being currently tracked by SAM2.1

2. For **every frame**:

   * project old map points into the frame
   * run SAM2.1 tracking for `active_gids`
   * validate tracked masks with projected 3D support
   * label the frame with accepted tracked masks

3. On **seed frames only**:

   * run `GT masks if --use-inst-gt else AMG masks`
   * ignore masks already explained by good tracks
   * for the rest:

     * attach to existing gid if strong old 3D support match
     * else birth a tentative gid if enough new support
   * also do duplicate merge here

4. When adding new 3D points:

   * assign gid from the final per-pixel instance image

5. Cleanup:

   * tentative ids confirm or die
   * confirmed ids can become inactive
   * inactive ids can be reactivated on later seed frames

---

## Pseudocode

```python
state:
    point_gid = []    # per global 3D point -> gid, -1 if unlabeled

    instances = {
        gid: {
            "status": "tentative" | "confirmed" | "inactive" | "dead",
            "n_points": int,
            "miss_count": int,
            "seed_hits": int,
            "last_seen": int,
        }
    }

    active_gids = set()
    next_gid = 0


def is_seed_frame(t):
    return (t % map_every == 0)


for each frame t:

    # ----------------------------------------
    # 1) project map into image
    # ----------------------------------------
    point_id_img, gid_img, new_geom_mask = project_map_to_frame(...)
    # gid_img[u,v] = gid of projected old 3D point at pixel, else -1

    inst_img = full_image_fill(-1)

    # ----------------------------------------
    # 2) TRACK every frame with SAM2.1
    # ----------------------------------------
    tracked_masks = sam2_track_step(frame_t, active_gids)

    good_tracked_masks = {}

    for gid, mask in tracked_masks.items():
        if valid_track(mask, gid, gid_img):
            good_tracked_masks[gid] = mask
            inst_img[mask] = gid
            instances[gid]["miss_count"] = 0
            instances[gid]["last_seen"] = t
        else:
            instances[gid]["miss_count"] += 1

    # drop bad / gone tracks from active set
    for gid in list(active_gids):
        if instances[gid]["miss_count"] > TRACK_MISS_MAX or out_of_view_too_long(gid, gid_img):
            active_gids.remove(gid)
            if instances[gid]["status"] == "tentative":
                instances[gid]["status"] = "dead"
            elif instances[gid]["status"] == "confirmed":
                instances[gid]["status"] = "inactive"

    # ----------------------------------------
    # 3) SEED pass only on seed frames
    # ----------------------------------------
    if is_seed_frame(t):

        if use_inst_gt:
            seed_masks = get_gt_masks(frame_t)
        else:
            seed_masks = get_amg_masks(frame_t)

        # throw away masks already explained by a good track
        seed_masks = remove_masks_explained_by_tracks(seed_masks, good_tracked_masks)

        for M in seed_masks:

            # try to attach/reactivate first
            gid_old, frac_old = dominant_old_gid_under_mask(M, gid_img)

            if gid_old is not None and frac_old >= ATTACH_TH:
                inst_img[M] = gid_old
                active_gids.add(gid_old)
                refresh_sam2_track(gid_old, M)

                if instances[gid_old]["status"] == "inactive":
                    instances[gid_old]["status"] = "confirmed"

                instances[gid_old]["seed_hits"] += 1
                instances[gid_old]["last_seen"] = t
                continue

            # otherwise consider birth
            n_new = count_new_support(M, new_geom_mask)

            if n_new >= BIRTH_MIN_POINTS:
                gid = next_gid
                next_gid += 1

                instances[gid] = {
                    "status": "tentative",
                    "n_points": 0,
                    "miss_count": 0,
                    "seed_hits": 1,
                    "last_seen": t,
                }

                active_gids.add(gid)
                refresh_sam2_track(gid, M)
                inst_img[M] = gid

        # merge duplicates only on seed frames
        for gid_a, gid_b in candidate_duplicate_pairs():
            if should_merge(gid_a, gid_b, gid_img, inst_img):
                parent, child = choose_parent(gid_a, gid_b)
                relabel_points(point_gid, child, parent)
                active_gids.discard(child)
                instances[child]["status"] = "dead"

    # ----------------------------------------
    # 4) add new 3D points and label them
    # ----------------------------------------
    new_points, new_pixels = add_new_points_from_frame(...)
    for pid, uv in zip(new_points.ids, new_pixels):
        gid = inst_img[uv]
        point_gid[pid] = gid

        if gid >= 0 and instances[gid]["status"] != "dead":
            instances[gid]["n_points"] += 1
            instances[gid]["last_seen"] = t

    # ----------------------------------------
    # 5) confirm or kill tentative ids
    # ----------------------------------------
    for gid, st in instances.items():
        if st["status"] != "tentative":
            continue

        if st["n_points"] >= CONFIRM_MIN_POINTS and st["seed_hits"] >= CONFIRM_MIN_SEED_HITS:
            st["status"] = "confirmed"
        elif too_old_without_confirmation(st, t):
            st["status"] = "dead"
            active_gids.discard(gid)
            relabel_points(point_gid, gid, -1)   # optional
```

---

## The important policy choices

The whole thing really boils down to these rules:

* **GT/AMG is only for seed frames**
* **SAM2.1 is for all in-between tracking**
* **Do not keep tracking all historical ids forever**
* **Drop only the track when a confirmed object leaves view**
* **Kill tentative junk quickly**
* **Merge only on seed frames**
* **Prefer attach/reactivate over birthing a new gid**

That is the cleanest version for your setup.

The only SAM2.1 facts I’m relying on here are that SAM2 supports image automatic mask generation, and that its video predictor is meant for multi-object promptable tracking with an inference state, with updated behavior allowing objects to be added after tracking has already started.
