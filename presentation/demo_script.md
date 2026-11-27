# Demo video: shot list (issue #53)

At most 2 minutes ([README.md](../README.md) "Demo video"). Recording is a person's job; this page says what to show
and what to say. Look: [design/STYLE.md](../design/STYLE.md): paper background, Geist, one orange accent for us,
title cards with the eyebrow line "HACK APERTUS ZURICH · DIGITEC GALAXUS".

**Numbers:** every `{…}` is filled from the generated files right before recording, never typed from memory
(AGENTS.md product rule 1):

```bash
python -m galaxus.serve.report --exp <final EXP> --kaggle --out docs/report_section7.md
```

`{EXP.mean}` and `{EXP.perspective}` come from its per-dimension table, `{img_s}` and `{cost_100k}` from its
"Resources" lines. A value still marked TODO there is cut from the video, not guessed.

| # | Time | On screen | Voice (plain, one breath per line) |
| --- | --- | --- | --- |
| 1 | 0:00–0:08 | Title card: "NoIdea · Apertus reads every Galaxus product image" | "Galaxus labels every product image six ways. We replaced their model with Apertus." |
| 2 | 0:08–0:22 | Four competition images in white frames, the six answers under each in Geist Mono (from the gallery, #31) | "Subject, background, perspective, people, text overlay and its language: six answers per image, decided by a Swiss open model." |
| 3 | 0:22–0:40 | `docs/architecture.svg`, slow zoom from left to right | "Every image is read once at 512 pixels. One runner sends holdout, dev and final through the same code. Only the predictor changes." |
| 4 | 0:40–0:58 | Terminal: `python -m galaxus.serve.runner … --workers 8`, the img/s counter climbing | "Apertus runs on our own GPU: {img_s} images a second, {cost_100k} dollars for a hundred thousand images." |
| 5 | 0:58–1:18 | The hero-shot gallery (`python -m galaxus.serve.demo`), one product, gallery reordering with the hero first | "This is what Galaxus does with the answers: the right picture first, images with foreign text hidden." |
| 6 | 1:18–1:36 | Results chart for the final EXP (from #57) next to the Galaxus baselines, ours in orange | "On our held-out images: {EXP.mean} mean macro F1, perspective {EXP.perspective}." |
| 7 | 1:36–1:50 | Terminal: `python -m galaxus.serve.preflight` ending in GO | "Saturday is rehearsed: one command checks the archive, the time and the submissions left." |
| 8 | 1:50–2:00 | Title card: repository URL, "Apertus v1.5 · on-premises · air-gapped" | "Open weights, open code, deployable where Galaxus wants it." |

## How to record (free tools only)

- Screen: Windows Game Bar (Win + Alt + R) or OBS Studio. 1920 × 1080, 30 fps.
- Title cards and the diagram: open `docs/architecture.svg` and the gallery HTML in Chrome, full screen (F11).
- Cut and voice: any free editor; the Edit with Ava credits from the Hack Apertus perks page are also fine.
- Only a few competition images on screen, as challenge_description.md "The data stays private" allows; the image
  files themselves never enter the repository.

## For design

- The video's URL goes into the **Demo** field at the top of technical_report.md (design owns that file).
- Shot 6 waits for the results chart (#57); shot 4 waits for a real Apertus run on a GPU (#59).
