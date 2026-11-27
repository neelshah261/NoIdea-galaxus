# Hack Apertus Zurich — Challenge Partner: Digitec Galaxus

## 🎯 The Challenge

### Hack Apertus x Galaxus: product image classification
https://www.kaggle.com/competitions/hackapertus-galaxus

`NoIdea`

Template repository for [Hack Apertus](https://hackapertus.ch/) submissions.
Every project keeps the same layout, so organizers and judges find the same
things in the same place.

Submissions must use the v1.5 Apertus model family.
The submission must comply with the [HackApertus T&C](https://hackapertus.ch/terms-and-conditions) and the [model and challenge requirements](https://hackapertus.notion.site/zurich-challenges-model-requirements)

### What you build

An Apertus v1.5 method that classifies every Galaxus product image along six dimensions and can replace the
finetuned Qwen 3.5 9B that does this today.

- **The challenge** (task, model rules, timeline, evaluation and submission file, what you hand in, how we assess): [challenge_description.md](challenge_description.md)
- **Frequent questions:** [FAQ.md](FAQ.md)
- **Reference notebook:** our Qwen 3.5 9B finetune on the Kaggle files, [reference/finetune_qwen35_reference.ipynb](reference/finetune_qwen35_reference.ipynb)
- **On Kaggle:** [data](https://www.kaggle.com/competitions/hackapertus-galaxus/data), [leaderboard](https://www.kaggle.com/competitions/hackapertus-galaxus/leaderboard) and [competition rules](https://www.kaggle.com/competitions/hackapertus-galaxus/rules)

The competition data never goes into this repository: no image files, csv files, labels or the passphrase.


## 🔧 Resources & Tools

Check our resources & tools page for more information:
[Resources & Tools](https://hackapertus.notion.site/zurich-resources-tools)

| Models           | URL                                      |
|------------------|------------------------------------------|
| Apertus v1.5 8B  | [huggingface.co/swiss-ai/Apertus-v1.5-8B](https://huggingface.co/swiss-ai/Apertus-v1.5-8B)  |
| Apertus v1.5 70B | [huggingface.co/swiss-ai/Apertus-v1.5-70B](https://huggingface.co/swiss-ai/Apertus-v1.5-70B) |

## The structure

Keep the files and directories as shown below — don't rename or move them.

| Path | What it is |
| --- | --- |
| `README.md` | The challenge description and submission requirements |
| `challenge_description.md` | The complete challenge description incl. deliverables: task, model rules, evaluation, what you hand in |
| `FAQ.md` | Short answers to frequent questions about the challenge |
| `reference/` | Our Qwen 3.5 9B finetune as a reference notebook, with the outputs of our run |
| `technical_report.md` | The deeper write-up: architecture, evaluation, limitations |
| `LICENSE` | Apache-2.0 |
| `src/` | Your code |

## Environment

- Include instructions for running the project in your technical report.
- Use the following environment variables:
```
LLM_NAME — name and version of the model
LLM_BASE_URL — endpoint base URL
LLM_API_KEY — your API key
```

## 📦 Submission Requirements & Deliverables

The submission must:
1. follow the template repo and include all prerequisite files and definitions
2. run end-to-end when judges try to run it

### Git repo (URL)
- Create your repo from this template (**Use this template**).
- Keep the layout as it is: don't rename or move its files.
- Set the repo to PUBLIC (Settings --> Collaborators --> Manage Visibility)
- Submit the URL of YOUR Git repo.

### Technical Report (pdf)
- Update [technical_report.md](technical_report.md) in this repository with all the details for your submission.
- Upload a pdf of your technical report to the root of the repo, named as `TeamName_Report.pdf`.
- Format: pdf, max. 6 pages w/o references
- Submit the pdf of the technical report.

### Demo video (URL)
- Max. 2 min demo video of your prototype
- Add the link to the **Demo** field at the top of [technical_report.md](technical_report.md)
- Submit the URL of the video.

### Pitch deck (pdf)
- Max. 10 slides

## Getting started

1. Click **Use this template** to create your own repository.

## License

All Hack Apertus projects are open-sourced. Please check our Terms & Conditions for specific licensing details (6. What you build is open source): https://hackapertus.ch/terms-and-conditions

## Support

### FAQ
💡 https://hackapertus.ch/faq

### Contact

💬 In case you have questions, join the conversation on Discord. Alternatively, send an email to “hello@hackapertus.ch”.
