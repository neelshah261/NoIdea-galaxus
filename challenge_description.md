# Hack Apertus x Galaxus: product image classification

## Why this matters

Galaxus shows millions of product images. Knowing what each image shows decides which image comes first in the
gallery, which images are hidden, and where text overlays or packaging shots belong. Today a finetuned Qwen 3.5 9B
labels every new image. Build an Apertus v1.5 method that can replace it: as accurate, fast and cheap enough for
more than 100,000 images per day, and deployable on-premises, air-gapped or in a sovereign Swiss cloud.

## The task

For every image, answer one class per dimension, 41 classes in total:

| Dimension | Question |
| --- | --- |
| subject | What does the image show? |
| background | What is behind the product? |
| perspective | From where is the product seen? |
| individuals | Is a real person or animal in the photo? |
| text_classification | Which kind of text overlay does the image carry? |
| text_language | Which language dominates the text? |

Each image comes with product context from our catalog: title, product type, brand, category and subcategory. In
production this context can be missing or poor, so your method should still work without it. The classes, their
definitions and the labeling conventions are in labeling_guideline.md in the
[Kaggle data](https://www.kaggle.com/competitions/hackapertus-galaxus/data), and with example images from train.csv in
labeling_guideline.pdf.

## Where to find what

| Where | What |
| --- | --- |
| [FAQ.md](FAQ.md) | short answers to frequent questions |
| [reference/finetune_qwen35_reference.ipynb](reference/finetune_qwen35_reference.ipynb) | our Qwen 3.5 9B recipe as a notebook: download from Kaggle, finetune, predict with vLLM, write the submission |
| [Kaggle data](https://www.kaggle.com/competitions/hackapertus-galaxus/data) | images, labels, labeling guideline, scoring code, the encrypted final images |
| [Kaggle leaderboard](https://www.kaggle.com/competitions/hackapertus-galaxus/leaderboard) | public on the dev images, private on the final images |
| [Kaggle rules](https://www.kaggle.com/competitions/hackapertus-galaxus/rules) | the competition rules, binding for everyone who joins |
| Hack Apertus Discord, Galaxus challenge channel | questions and the passphrase of the final images. The Discord names of the Galaxus hosts, on site on 27 and 28 November, are pinned there |
| [Kaggle discussion](https://www.kaggle.com/competitions/hackapertus-galaxus/discussion) | the passphrase of the final images |
| [Resources & Tools](https://hackapertus.notion.site/zurich-resources-tools) | GPUs and the Apertus inference API of Hack Apertus; the compute credits are in the [FAQ](FAQ.md) |
| [sprintd](https://app.sprintd.org/events/hack-apertus-zurich) | your project submission |

## Models

The [model and infrastructure rules of Hack Apertus](https://hackapertus.notion.site/zurich-challenges-model-requirements)
apply. In short:

- Apertus v1.5 (8B or 70B), also finetuned or adapted, performs the classification.
- During development you may use any tools and models, including closed models, commercial APIs and coding
  assistants, for example to create training targets. Pseudo-labels on the unlabeled dev images are allowed.
- At runtime your method uses only Apertus and, for supporting tasks, other open-weights models (for example an
  image encoder or OCR). Apertus may run self-hosted or through the inference API that Hack Apertus provides.
  Closed models and proprietary AI APIs are not allowed at runtime.
- The solution is deployable on-premises, air-gapped or in a sovereign Swiss cloud, and your technical report lists
  every model you used.
- Not allowed: labeling dev or final images by hand, and looking up the products in the Galaxus shop or by reverse
  image search (Kaggle rules, Section B(6)).

## Timeline

| When (CET) | What |
| --- | --- |
| Fri 27 Nov, 09:00 | Challenge pitches. |
| Fri 27 Nov, 10:30 | Deep dive with the matched teams. |
| Fri 27 Nov, after the deep dive | Join the Kaggle competition with the link from the deep dive and accept its rules. Form your Kaggle team, named like your Hack Apertus team, before your first submission: Kaggle may refuse a later merge. Download the data and check the SHA-256 of final_encrypted.bin. |
| During the hack | Train, and submit dev predictions to the public leaderboard (20 per day). |
| Sat 28 Nov, 13:00 | We publish the passphrase of the final images in Discord and in the Discussion tab on Kaggle. Run your method on them, for inference only. Plan for at least 2 images per second: the 5,292 images then take about 45 minutes, and your file is on Kaggle by 14:15. |
| Sat 28 Nov, 14:30 | Deadline for the Kaggle submission and for your project submission on sprintd. |
| Sat 28 Nov, 15:00 | Team pitches, 7 minutes plus 2 minutes of questions. The top 2 go on to the Zurich finals at 18:30. |

How to decrypt the final images is described in the [Kaggle data](https://www.kaggle.com/competitions/hackapertus-galaxus/data),
how to build the final submission file under [Submission file](#submission-file) below.

## What you hand in

By 14:30 on 28 November:

1. Your Kaggle submission with the predictions for the final images, selected for the final ranking.
2. Your project on [sprintd](https://app.sprintd.org/events/hack-apertus-zurich), as Hack Apertus requires and the
   [README](README.md) describes: this repository (public), the demo video, the pitch deck and the technical report.

For this challenge the technical report also fills in section 7 of [technical_report.md](technical_report.md):
scores, what mattered, models, resources and weights. Your README or section 6 of the report explains setup, training
and inference with pinned dependencies, configs and seeds.

There is no code freeze: you may work on your method until 14:30. The final images are for inference only: do not
train or tune on them, and do not label them by hand. Section 7 of your technical report names the commit and the
revision of the weights that produced your final submission; publish the weights, for example on Hugging Face, by
14:30. If you use Apertus unchanged, the commit is enough. We may rerun any team's code on the final images from that
commit and those weights; a score that does not reproduce within its performance group may be disqualified
([Kaggle rules](https://www.kaggle.com/competitions/hackapertus-galaxus/rules), Section B(7)).

## Evaluation

Submissions are scored with the macro F1 per dimension, averaged over the six dimensions. The public leaderboard
scores the dev images, the private leaderboard the final images.

### Which cells are scored

Only human labels are scored, and only where the dimension is meaningful for the image:

- subject logo, placeholder or energy_label: background, perspective, individuals and text_classification are not scored
- subject infographic: background, perspective and individuals are not scored
- subject delivery_scope, detail_closeup or product_grouping: perspective is not scored
- background scene_lifestyle: subject and perspective are not scored
- text_language is scored only when a text overlay is present, or subject is infographic, packaging,
  package_and_product or full_product

The rules are applied to the human labels only: a model prelabel never triggers a rule. Your answer in a cell that
is not scored does not matter.

### The score

Per dimension we compute the F1 of every class and take the unweighted mean (macro F1). The classes are every
class in the labels plus every valid class you predicted, so predicting a class that never occurs costs you. The
leaderboard shows the mean of the six macro F1 scores, so rare classes count as much as frequent ones.

The Results page in the [Kaggle overview](https://www.kaggle.com/competitions/hackapertus-galaxus/overview) shows the macro F1, precision and recall
per dimension and the weighted F1: now for our baselines on the dev images, after the deadline for every team on
the final images. metric.py in the Kaggle data computes these and the F1, precision and recall per class locally.

### Classes

One class per dimension, 41 classes in total. labeling_guideline.md in the Kaggle data defines each of them.

| Dimension | Classes |
| --- | --- |
| subject | full_product, detail_closeup, accessories, packaging, package_and_product, multipack_trade_unit, delivery_scope, product_grouping, multiview, energy_label, infographic, logo, placeholder |
| background | white, color, scene_lifestyle |
| perspective | front_head_on, front_angle_left, front_angle_right, back_head_on, back_angle_left, back_angle_right, left_side, right_side, top_down, bottom_up |
| individuals | individual_present, none |
| text_classification | certification_seal, technical_specs, promotional_text, nutritional_info, other, none |
| text_language | German, English, French, Italian, Dutch, other, none |

text_classification is the overlay dimension. none is a real class and is scored like any other.

### Baselines

Our reference models on the dev images, scored like the public leaderboard (mean macro F1 over the six dimensions):

| Model | Dev score |
| --- | --- |
| Our production recipe, a finetuned Qwen 3.5 9B trained without the dev and final images | 0.917 |
| Gemini 4 Argon, zero-shot with our production prompt | 0.834 |
| Gemini 3.8 Flash, zero-shot with our production prompt | 0.806 |

All three read the same images as you. Qwen composites transparent backgrounds onto white, with the function on
the Kaggle data page, and scales each image to about 512 x 512 pixels in area; the Gemini models get the files
unchanged. The scores of Qwen 3.5 9B and
Gemini 3.8 Flash on the final images set the performance groups of the challenge assessment. We publish their final
scores with the private leaderboard.

[reference/finetune_qwen35_reference.ipynb](reference/finetune_qwen35_reference.ipynb) runs the Qwen recipe from start
to finish on the Kaggle files, with a short prompt of its own instead of our production prompt, and scores about as
well as our Qwen baseline. Your method classifies with Apertus, but most of the notebook carries over.

### Scoring locally

metric.py scores a solution that holds the human label of every scored cell and is empty in every other cell. To
validate on train.csv, hold out a part of it split by product_id:

```python
import pandas as pd
import metric

train = pd.read_csv("train.csv")
holdout_products = train.product_id.drop_duplicates().sample(frac=0.2, random_state=0)
holdout = train[train.product_id.isin(holdout_products)]
solution = holdout[["image_id", *metric.DIMENSIONS]].copy()
for dimension in metric.DIMENSIONS:
    unscored = (holdout[f"{dimension}_source"] == "model") | ~holdout[f"{dimension}_relevant"]
    solution.loc[unscored, dimension] = None
submission = pd.read_csv("holdout_predictions.csv")
print(metric.score(solution, submission, "image_id"))
```

`score` returns the leaderboard number, `metric.evaluate` with the same arguments the F1, precision and recall per
dimension and class. metric.py needs Python 3.10 and pandas 2.1 or newer.

### Submission file

One row per dev and final image, with a header:

    image_id,subject,background,perspective,individuals,text_classification,text_language
    img_00001,full_product,white,front_angle_left,none,none,none
    img_00002,infographic,white,front_head_on,none,technical_specs,German

- Every cell holds a class name exactly as listed under Classes above and in labeling_guideline.md, none
  lowercase: None with a capital N is read as an empty cell. Kaggle rejects a file with an empty cell or an unknown
  class name, with a message; locally, metric.py counts an empty cell as a wrong answer.
- Before 13:00 CET on 28 November, keep the final rows as in sample_submission.csv. They only affect the hidden
  private score.
- After 13:00 CET, predict the final images and upload one file with dev and final rows, before the deadline at
  14:30 CET (13:30 UTC). Kaggle scores the dev rows for the public and the final rows for the private
  leaderboard, separately: the dev rows of your final file do not change your final score, but they must be there
  and valid. The easiest way is your best dev file with the final rows replaced.
- Kaggle counts the 20 daily submissions per UTC day, so 28 November starts at 01:00 CET: keep some for the final
  90 minutes from 13:00 to 14:30.
- Upload by about 14:15 and select that submission for the final ranking: Kaggle must score a file before you can
  select it. A submission from before 13:00 CET still carries the constant answers of sample_submission.csv for the
  final images. If you select nothing, Kaggle takes your submission with the best public score, which may be one
  with the constant answers.
- If Kaggle is down around the deadline, send your final file by direct message to the Galaxus hosts on Discord
  before 14:30 CET. We score it with metric.py, which computes the same score as Kaggle, and rank it with the private
  leaderboard. This is only for an outage of Kaggle: a file that Kaggle rejects stays invalid.

## How we assess

The Kaggle ranking is the private leaderboard. For the assessment of this challenge, its score is the performance
part, counted in groups against our [baselines](#baselines): better than our
finetuned Qwen 3.5 9B, between Gemini 3.8 Flash and Qwen 3.5 9B, or below Gemini 3.8 Flash. Differences of a few
tenths of a percent are not decisive. You may focus on some dimensions, but the performance part uses the mean over
all six. Besides that score, we also look at the weighted F1 and the scores per dimension of your selected submission,
because the frequent classes make up most of the images in our shop.

| Part | Weight |
| --- | --- |
| Performance on the final images | about 30% |
| Innovation: how far the approach goes beyond training longer | 20% |
| Speed and resource usage: images per second, memory, cost per 100,000 images | 20% |
| Insights: clear evidence of what mattered and what did not | 15% |
| Code quality and practicality | 15% |

Very good perspective scores and a list of suspected label errors in train.csv earn a bonus. Send the list by 14:30
on 28 November as a csv with the columns image_id, dimension, label and suggested_label, one row per suspected
error, by direct message to the Galaxus hosts on Discord, and never commit it. The weights are a guideline: in close
cases we prefer the solution that is practical for us to adopt, and we explain our reasoning. The awards and the
advance to the Zurich finals follow the Hack Apertus process.

## The data stays private

The competition data and its labels are for this hackathon only: no commercial use, no passing them on, also not to
other teams, and delete them afterwards. Your code and model weights are open source under the Hack Apertus T&C, but
this repository and your published weights never contain image files, csv files, labels or the passphrase; the
.gitignore of this template keeps the data files out. You may show a few competition images in your pitch, pitch
deck, demo video and technical report.
