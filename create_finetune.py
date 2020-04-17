"""Script which loads multiple datasets and prepares them for finetuning"""
import pandas as pd
from google.oauth2.service_account import Credentials
import gspread
import re
import unicodedata
import os
import logging
import html


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)-5.5s] [%(name)-12.12s]: %(message)s')


scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
credentials = Credentials.from_service_account_file('/home/martin/.config/gcloud/cb-tpu-projects-7eb849ddd1ba.json', scopes=scope)
gc = gspread.authorize(credentials)
sheet_handler = gc.open('Twitter Evaluation Datasets')

sheets = ['vaccine_sentiment_epfl', 
        'maternal_vaccine_stance_lshtm',
        'twitter_sentiment_semeval']
transl_table = dict([(ord(x), ord(y)) for x, y in zip( u"‘’´“”–-",  u"'''\"\"--")])
user_handle_regex = re.compile(r'(^|[^@\w])@(\w{1,15})\b')
control_char_regex = re.compile(r'[\r\n\t]+')


def read_data(sheet_name):
    # Read the vaccine_sentiment_epfl
    worksheet = sheet_handler.worksheet(sheet_name)
    rows = worksheet.get_all_values()
    # Get it into pandas
    df = pd.DataFrame.from_records(rows)
    df.columns = df.iloc[0]
    df  = df.reindex(df.index.drop(0))
    df['a'] = 'a'
    return df

def remove_control_characters(s):
    # replace \t, \n and \r characters by a whitespace
    s = re.sub(control_char_regex, ' ', s)
    # replace HTML codes for new line characters
    s = s.replace('&#13;', '').replace('&#10;', '')
    # removes all other control characters and the NULL byte (which causes issues when parsing with pandas)
    return "".join(ch for ch in s if unicodedata.category(ch)[0]!="C")

def clean_text(text):
    """Replace some non-standard characters such as ” or ’ with standard characters. """
    # remove anything non-printable
    text = remove_control_characters(text)
    # escape html (like &amp; -> & or &quot; -> ")
    text = html.unescape(text)
    # standardize quotation marks and apostrophes
    text = text.translate(transl_table)
    # replace multiple spaces with single space
    text = ' '.join(text.split())
    return text

def replace_user_handles(text):
    # Replace previously added filler user mentions
    text = text.replace('@twitteruser', '@<user>')
    # Replace any other remaining user mentions
    text = re.sub(user_handle_regex, '@<user>', text)
    return text

def replace_urls(text):
    # Replace previously added filler URL
    text = text.replace('http://anonymisedurl.com', '<url>')
    # Replace any other remaining urls
    return re.sub('((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))', '<url>', text)

def clean_data(df):
    """Replaces user mentions & standardize text"""
    df.loc[:, 'text'] = df.text.apply(clean_text)
    df.loc[:, 'text'] = df.text.apply(replace_user_handles)
    df.loc[:, 'text'] = df.text.apply(replace_urls)
    return df

def main():
    output_dir = os.path.join('output', 'finetune')
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    for s in sheets:
        logger.info(f'Reading sheet {s}...')
        df = read_data(s)
        logger.info('Cleaning data...')
        df = clean_data(df)
        # add dummy column
        df['a'] = 'a'
        # write train, dev and test set
        for _type in ['train', 'dev', 'test']:
            _df = df[df['dataset'] == _type]
            _df = _df.sample(frac=1)
            f_path = os.path.join(output_dir, f'{s}_{_type}.tsv')
            logging.info(f'Writing cleaned finetune data {f_path}')
            _df.to_csv(f_path, columns=['id', 'label', 'a', 'text'], header=False, index=False, sep='\t')

if __name__ == "__main__":
    main()
