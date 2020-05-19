# Pretrained models configuration, add model configuration here

PRETRAINED_MODELS = {
        'bert_large_uncased': {
            'location': 'pretrained_models/bert/keras_bert/uncased_L-24_H-1024_A-16',
            'config': 'bert_config_large_uncased.json',
            'is_tfhub_model': False,
            'vocab_file': 'bert-large-uncased-vocab.txt',
            'lower_case': True,
            'do_whole_word_masking': False
            },
        'bert_large_uncased_wwm': {
            'location': 'pretrained_models/bert/keras_bert/wwm_uncased_L-24_H-1024_A-16',
            'config': 'bert_config_large_uncased_wwm.json',
            'is_tfhub_model': False,
            'vocab_file': 'bert-large-uncased-whole-word-masking-vocab.txt',
            'lower_case': True,
            'do_whole_word_masking': True
            },
        'covid-twitter-bert': {
            'location': 'digitalepidemiologylab/covid-twitter-bert/1',
            'is_tfhub_model': True,
            'config': 'bert_config_covid_twitter_bert.json',
            'vocab_file': 'bert-large-uncased-whole-word-masking-vocab.txt',
            'lower_case': True,
            'do_whole_word_masking': True
            }
        }
