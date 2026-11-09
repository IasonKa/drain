# The drain inside the body, with irrigation

Two pictures that show how the standoff drain would sit in the abdomen with an irrigation catheter whose tip is a little above it.

| File | What it shows |
|---|---|
| `in_body_setup.png` | **Left:** the whole set-up, seen from the side with the patient lying on the back. The drain lies in the lowest part of the abdominal cavity, under the bowel. The irrigation catheter comes in through its own skin exit, and its tip sits a few cm above the drain. Outside the body: the saline bag, a pump that sets the inflow rate, the collection canister and the wall suction regulator at −7 mmHg. **Top right:** a cut across the drain, with the liquid running down the cleft between two bowel loops, a thin film over the drain and a channel along each edge. **Bottom right:** one hole, with the organ resting on the ring tops and the liquid passing under it through the slots. |
| `in_body_3d_packed.png` | **The realistic version.** The same scene, but packed the way it is in the body: loops lying across the drain and touching each other, loops along both edges, a second layer of loops on top and the omentum draped over everything. There is no empty space. All tissue is drawn see-through so you can still see the drain, the catheter and the thin liquid films. |
| `in_body_3d.png` | Cut-away version: the front half of the loops is removed and only a few loops are drawn. The empty space around the drain comes only from that cut and isn't real. |

## What is real and what is drawn

- **Real:** the drain geometry in the 3D view, taken from your STL, and the dimensions in the close-ups: 17.85 × 9.67 mm section, 0.6 mm wall, Ø1.0 mm holes, rings 0.4 mm high with 0.35 mm slots, −7 mmHg.
- **Drawn here:** the organs, the catheter and the liquid. Their sizes are plausible (bowel about 28 mm across, catheter 4 mm), but the anatomy is simplified.
- **Not to scale:** the liquid film and the gaps are drawn much thicker than they really are. Real gaps are about 0.1–0.4 mm.
- The placement is only an example, not clinical advice. Where the drain and the catheter go is up to the surgical team.

## How to rebuild

Requirements:
- Python 3.12 with numpy, matplotlib and pyvista;
- the drain STL in `../../drainsponge/data/`. It isn't in git; copy it from `new_offset_desing.zip`.

```
py -3.12 make_in_body_setup.py     # in_body_setup.png, a few seconds
py -3.12 make_in_body_3d.py        # in_body_3d.png, under a minute
py -3.12 make_in_body_3d_packed.py # in_body_3d_packed.png (uses make_in_body_3d.py)
```

`make_in_body_3d.py` keeps a trimmed copy of the drain in `_drain_body_cache.vtp` (not committed). Delete that file after changing the STL. To move only the labels after a render, run `py -3.12 make_in_body_3d.py --labels`.
