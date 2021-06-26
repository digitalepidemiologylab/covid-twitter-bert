---
language:
- en
thumbnail: "https://raw.githubusercontent.com/digitalepidemiologylab/covid-twitter-bert/master/images/COVID-Twitter-BERT_small.png"
tags:
- Twitter
- COVID-19
- text-classification
- pytorch
- tensorflow
- bert
license: MIT
datasets:
- mnli
pipeline_tag: zero-shot-classification
widget:
- text: "To stop the pandemic it is important that everyone turns up for their shots."
  candidate_labels: "health, sport, vaccine, guns"
---

# COVID-Twitter-BERT v2 MNLI

## Model description
This provides a zero-shot classifier to be used in cases where it is not possible to finetune the model on a specific task.

The technique is based on [Yin et al.](https://arxiv.org/abs/1909.00161). The article describes a very clever way of using pre-trained MNLI models as zero-shot sequence classifiers. The model is already finetuned on 400.000 generaic logical tasks. We can then use it as a zero-shot classifier by reformulating the classification task as a question.

Lets say you want to classify COVID-tweets as vaccine-related and not vaccine related. The typical way would be to collect a few hunder pre-annotated tweets and organise them in two classes. Then you would finetune the model on this.

With the zero-shot mnli-classifier, you can instead reformulate your question as "This text is about vaccines", and use this directly on inference. Without any training!
 
Find more info about the model on our [GitHub page](https://github.com/digitalepidemiologylab/covid-twitter-bert).

## Intended uses & limitations
Please note that how you formulate the question can give slightly different results. Collecting a training set and finetuning on this, will most likely give you better accuracy.

#### How to use
The easiest way to try this out is by using the Hugging Face pipeline. This uses the default Enlish template where it puts the text "This example is " in front of the text.  

```python
from transformers import pipeline
classifier = pipeline("zero-shot-classification", model="digitalepidemiologylab/covid-twitter-bert-v2-mnli")
```
You can then use this pipeline to classify sequences into any of the class names you specify.
```python
sequence_to_classify = 'To stop the pandemic it is important that everyone turns up for their shots.'
candidate_labels = ['health', 'sport', 'vaccine','guns']
hypothesis_template = 'This example is {}.'

classifier(sequence_to_classify, candidate_labels, hypothesis_template=hypothesis_template, multi_class=True)
``` 

## Training procedure
The model is finetuned on the 400k large [MNLI-task](https://cims.nyu.edu/~sbowman/multinli/) 


### BibTeX entry and citation info
bibtex
@article{muller2020covid,
  title={COVID-Twitter-BERT: A Natural Language Processing Model to Analyse COVID-19 Content on Twitter},
  author={M{\"u}ller, Martin and Salath{\'e}, Marcel and Kummervold, Per E},
  journal={arXiv preprint arXiv:2005.07503},
  year={2020}
}

or
Martin Müller, Marcel Salathé, and Per E. Kummervold.
COVID-Twitter-BERT: A Natural Language Processing Model to Analyse COVID-19 Content on Twitter.
arXiv preprint arXiv:2005.07503 (2020).
