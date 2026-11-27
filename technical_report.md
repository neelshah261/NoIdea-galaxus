# Technical report — `project name`, `team name`

A deeper write-up than the README: what you built, how it works, and what the
numbers say.

- **Challenge partner:** Digitec Galaxus
- **Challenge:** `challenge name`
- **Event:** On-site Zurich
- **Team:** `team name` — `member`, `member`, `member`
- **Demo:** `link to video`

## 1. Summary

The problem, your approach, and the headline result in one paragraph.

## 2. Architecture

Components, data flow, and where each one runs. Put diagrams in `docs/` and
reference them here.

### Target architecture (mandatory)

State which of the three architectures your project is deployable in, and how
it meets that constraint:

- **a) On-premise** — on the organisation's own infrastructure, under its own administration.
- **b) Air-gapped** — with no external network connection at runtime.
- **c) Sovereign Swiss cloud** — on a cloud platform operated in Switzerland, under Swiss jurisdiction, with Swiss data residency.

List any external dependencies, and separate build time from runtime.

## 3. Use of Apertus

- **Model:** `e.g. swiss-ai/Apertus-v1.5-8B`
- **How it is used:** inference | fine-tuning | evaluation | red-teaming | agents / tool use
- **Where it runs:** `local weights, hosted endpoint, ...`

Prompts, adapters, quantisation, serving stack — whatever a reader needs to
rebuild your setup.

## 4. Data

What you used, where it came from, and its licence. Flag anything personal or
non-redistributable, and keep it out of the repository (see `.gitignore`).
If data comes from human subjects or contains personal information, describe
how consent was obtained.

## 5. Limitations

Where it breaks, what you did not test, and known failure modes.

## 6. Reproducibility

What a judge needs to get your numbers back: hardware, runtime, seeds, and the
exact commit.

## 7. Galaxus challenge: results and resources

- **Kaggle:** team name and the public score (dev) of your selected submission; we take the private score from Kaggle.
- **Per dimension:** macro F1 of subject, background, perspective, individuals, text_classification and
  text_language (metric.py from the competition data computes them on your own validation split).
- **What mattered:** the changes with the biggest impact on the score (ablations welcome) and what did not work.
- **Models:** every model used in development and at runtime, with version and role.
- **Resources:** hardware (GPU type, VRAM), batch size, images per second, and the estimated cost of
  100,000 images per day.
- **Commit and weights:** the commit that produced your final submission, and where its model weights are published,
  with the revision.
- **Optional:** how many suspected label errors in train.csv you sent to the hosts as a csv (see challenge_description.md); the
  csv itself is not committed here.


## License

Creative Commons Attribution 4.0 (CC-BY-4.0). All HackApertus projects are open-sourced.

## References
