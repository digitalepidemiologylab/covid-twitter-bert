# COVID Category dataset

This dataset is a subsample of the data used for training CT-BERT, specifically for the period between January 12 and February 24, 2020. Annotators on Amazon Turk (MTurk) were asked to categorise a given tweet text into
either being a personal narrative (33.3%) or news (66.7%). The annotation was performed using the [Crowdbreaks](crowdbreaks.org) platform.

## Usage
Download the file using [this link](https://raw.githubusercontent.com/digitalepidemiologylab/covid-twitter-bert/master/datasets/covid_category/covid_category.csv). 

We can only share the Tweet IDs. You can download the full tweet objects using the script provided [here](https://github.com/digitalepidemiologylab/crowdbreaks-paper).

If you end up using this dataset, please cite our pre-print:
```bibtex
@article{muller2020covid,
  title={COVID-Twitter-BERT: A Natural Language Processing Model to Analyse COVID-19 Content on Twitter},
  author={M{\"u}ller, Martin and Salath{\'e}, Marcel and Kummervold, Per E},
  journal={arXiv preprint arXiv:2005.07503},
  year={2020}
}
```
or
```
Martin Müller, Marcel Salathé, and Per E. Kummervold. 
COVID-Twitter-BERT: A Natural Language Processing Model to Analyse COVID-19 Content on Twitter.
arXiv preprint arXiv:2005.07503 (2020).
```

If you have questions, please get in touch martin.muller@epfl.ch.
