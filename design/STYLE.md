# STYLE.md: the visual system

Taken from the DigitalHeat deck and site
([design/frontend](https://github.com/Mulaydm10/sprintd-basel-greentech-digitalheat/tree/main/design/frontend), MIT).
One look for the deck, the architecture diagram, the demo video's title cards and the report figures.

## Colours

| Token | Hex | Use |
| --- | --- | --- |
| `paper` | `#f2efe9` | slide and page background |
| `paper-2` | `#f8f6f1` | cards, table stripes |
| `white` | `#ffffff` | image frames, chart plot area |
| `ink` | `#141414` | headlines, body text, rules under eyebrows, the dark page behind slides |
| `ink-2` | `#2e2b27` | sub-headlines, long text |
| `muted` | `#55524c` | eyebrow line, captions, axis labels |
| `muted-2` | `#6e6a63` | secondary captions, footnotes |
| `rule` | `#dcd6cc` | grid lines, table borders |
| `rule-2` | `#c4bcae` | stronger dividers |
| `accent` | `#c93c05` | **us**: our method, the key figure, links, the one word to remember |
| `accent-2` | `#a86400` | link hover, our second variant in a chart |
| `blue` | `#1d6bb3` | the Galaxus Qwen 3.5 9B baseline |
| `blue-2` | `#5a93c9` | the Gemini baselines |
| `green` | `#157a43` | better than the baseline, a fixed error |
| `red` | `#d1173a` | worse, a confusion, a failure case |
| `amber` | `#e3a008` | warning, "unclear", pending |

One accent per slide. Baselines are always blue, we are always orange, so a judge can read any chart without its
legend.

## Type

- **Geist** for everything, **Geist Mono** for eyebrows, numbers, class names, file names and code
  (Google Fonts: `family=Geist:wght@100..900&family=Geist+Mono:wght@400..700`).
- Slide canvas 1920 × 1080. Scale on that canvas:

| Element | Font | Size | Weight / tracking |
| --- | --- | --- | --- |
| Title-slide word mark | Geist | 260 px | 760, letter-spacing −0.06em, line-height 0.85 |
| Slide headline | Geist | 96–120 px | 700, −0.04em |
| Lead sentence | Geist | 48 px | 400, line-height 1.25, max width 1400 px |
| Body / bullets | Geist | 32–36 px | 400 |
| Eyebrow line | Geist Mono | 24 px | uppercase, +0.08em, `muted` |
| Key-figure row | Geist Mono | 26 px | `ink`, 2 px `ink` rule above |
| Class names in text | Geist Mono | as body | any name from `contracts/classes.json`, never retyped |

## Slide anatomy

Every content slide has the same three bands (padding 96 px top, 110 px sides, 84 px bottom):

1. **Eyebrow row**: left "HACK APERTUS ZURICH · DIGITEC GALAXUS", right the section ("APPROACH", "EVIDENCE", …).
2. **Body**: one headline that is a claim, not a topic ("Perspective errors drop by a third", not "Perspective").
   One lead sentence. Then one visual: a chart, a table, an image grid or the diagram. Never two.
3. **Key-figure row** (title and result slides): three numbers in a 3-column grid, each with its `EXP-####`.

Section slides ("Approach", "Evidence", "Speed") are full `paper` with the section word at 200 px. Every slide
carries speaker notes (`data-speaker-notes`): what to say, in full sentences, with the source of each number.

## Charts and figures

- Bars for scores, ordered by value; our bar in `accent`, baselines in `blue` / `blue-2`, everything else `rule-2`.
- Per-dimension results: a small-multiples row of six, same y-axis, macro F1 written on each bar in Geist Mono.
- Confusion pairs (perspective): only the pairs that matter, before → after, `red` → `green`.
- Competition images: a few, in `white` frames with a 1 px `rule` border, the predicted labels below in Geist Mono.
  Only as many as challenge_description.md "The data stays private" allows; never commit the image files.
- No 3D, no gradients, no shadows. Grid lines `rule`, no chart border.

Report figures use the same look (model lane, matplotlib):

```python
import matplotlib as mpl

mpl.rcParams.update({
    "font.family": "Geist", "font.size": 11,
    "figure.facecolor": "#f2efe9", "axes.facecolor": "#ffffff",
    "axes.edgecolor": "#dcd6cc", "axes.grid": True, "grid.color": "#dcd6cc",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.labelcolor": "#55524c", "xtick.color": "#55524c", "ytick.color": "#55524c",
    "axes.prop_cycle": mpl.cycler(color=["#c93c05", "#1d6bb3", "#5a93c9", "#157a43", "#a86400", "#6e6a63"]),
})
```

## Deck runtime

The DigitalHeat deck is plain HTML on the `deck-stage` component (`deck-stage.js`, `support.js` in that repo): arrow
keys, fullscreen, speaker notes, and **print to PDF**, which gives the PDF deck the hand-in asks for (README.md
"Pitch deck"). The serve lane can reuse both files in `presentation/` with the MIT notice kept, one `<section>` per
slide with `data-label` and `data-speaker-notes`.
