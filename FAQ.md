# FAQ: Hack Apertus x Galaxus

Short answers to the questions we expect. The full rules are in [challenge_description.md](challenge_description.md) and the
[Kaggle rules](https://www.kaggle.com/competitions/hackapertus-galaxus/rules); where they differ from this page, they
apply. Anything else: ask in the Galaxus challenge channel on Discord.

## Joining

**How do we join the Kaggle competition?**
With the link or QR code from the deep dive on Friday at 10:30. Every member joins with their own Kaggle account and
accepts the rules. Then form one Kaggle team, named like your Hack Apertus team, at most 5 people, before your first
submission: Kaggle may refuse a later merge.

**Do we need a Hugging Face account?**
Yes. The Apertus v1.5 models on Hugging Face are gated: log in and accept their terms before you download them. You
also publish your weights there or in a similar place.

**What compute do we get?**
Every member has a $10 Hugging Face coupon code in their sprintd account under Perks. On top, Hack Apertus gives each
team up to $20 of Hugging Face compute: the team lead joins the Hack Apertus organization on Hugging Face. It is one
pool for all teams, so please stay within your $20. Kaggle notebooks add 30 GPU hours per week for every account. An
L40S on Hugging Face costs about $1.80 an hour.

## Models

**Which Apertus?**
Apertus v1.5, 8B or 70B, also finetuned or adapted. It performs the classification itself.

**Which other models may we use?**
While you develop, any: closed models, commercial APIs and coding assistants, for example to create training targets
or pseudo-labels on the dev images. At runtime only Apertus and, for supporting tasks, open-weights models such as an
image encoder, embeddings or OCR. Closed models and proprietary AI APIs are not allowed at runtime. Unsure whether a
design still counts as Apertus classifying? Ask on Discord before you build it.

**Can we get your production prompt?**
No. labeling_guideline.md describes every class and convention our labelers followed. The
[reference notebook](reference/finetune_qwen35_reference.ipynb) finetunes Qwen 3.5 9B with a short prompt of its own
and scores about as well as our baseline.

**May Apertus run through the Hack Apertus inference API?**
Yes, at runtime Apertus may run self-hosted or through that API. GPUs and the API are described on the
[Resources & Tools](https://hackapertus.notion.site/zurich-resources-tools) page of Hack Apertus.

## Data and labels

**Why do some train labels have the source `model`?**
They are prelabels that no labeler checked, about a fifth of the train cells, mostly in cells that are not scored. Train on
them or not, as you like. Dev and final are scored on human labels only.

**May we correct the train labels?**
Yes, train is yours to clean, also by hand. Only the dev and final images may not be labeled by hand. A list of the
errors you found earns a bonus (see below).

**May we use other data?**
Yes, data that is available to all participants on equal terms, and pseudo-labels on the dev images. Not allowed: looking
up the products in the Galaxus shop or by reverse image search.

**Transparent PNGs turn black. Why?**
A plain `image.convert("RGB")` often turns the transparent area black. Our labelers saw them on white: use the function on
the [Kaggle data](https://www.kaggle.com/competitions/hackapertus-galaxus/data) page.

## Submissions

**What does a valid submission look like?**
One row per dev and final image, as in sample_submission.csv, with class names exactly as in labeling_guideline.md;
none is lowercase. Kaggle rejects a file with an empty cell or an unknown class name. Until 13:00 on Saturday keep the
final rows of sample_submission.csv.

**How many submissions do we get?**
20 per UTC day; Saturday starts at 01:00 CET. Keep some for the final 90 minutes.

**Which submission counts?**
The one you select. Upload by 14:15 and select it: Kaggle must score a file before you can select it. If you select
nothing, Kaggle takes your submission with the best public score, which may carry the constant answers for the final images.

## The final 90 minutes

**When do we get the final images?**
You download them encrypted on Friday. At 13:00 on Saturday we publish the passphrase in Discord and in the Kaggle
discussion. Test the download on Friday: the SHA-256 and that `7z x final_encrypted.bin` asks for a passphrase (on a
Mac `7zz`).

**How fast does our method need to be?**
Plan for at least 2 images per second: the 5,292 final images then take about 45 minutes, and your file is on Kaggle
by 14:15. Time a full dry run on the dev images before 13:00.

**May we keep working after 13:00?**
Yes, there is no code freeze until 14:30. The final images are for inference only: no training, tuning or labeling on
them.

**What if something breaks?**
If Kaggle is down, send your final file by direct message to the Galaxus hosts on Discord before 14:30. Otherwise the
deadline stands: upload what you have, and rows you could not predict keep the constant answers of sample_submission.csv.

## Hand-in and assessment

**What do we hand in by 14:30?**
Your selected Kaggle submission, and on sprintd your public repository, demo video, pitch deck and technical report.
Section 7 of the report names the commit and the revision of the weights that produced your final submission;
publish the weights by 14:30. If you use Apertus unchanged, the commit is enough.

**How are we judged?**
Performance on the final images is about 30%, counted in groups against our Qwen 3.5 9B and Gemini 3.8 Flash. The
rest is innovation, speed and resources, insights and code quality: see
[How we assess](challenge_description.md#how-we-assess).

**Is it worth focusing on a few dimensions?**
The performance part uses the mean over all six. Very good perspective scores earn a bonus.

**How do we send our list of label errors?**
As a csv with the columns image_id, dimension, label and suggested_label, by direct message to the Galaxus hosts on
Discord by 14:30. Never commit it: it contains labels.

**May we show competition images?**
Yes, a few, in your pitch, pitch deck, demo video and technical report. The image files, csv files and labels never
go into your repository or your weights.
