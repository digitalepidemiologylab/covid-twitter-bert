import urllib.request
import os

download_locations = {
        'bert-base-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt",
        'bert-large-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-vocab.txt",
        'bert-base-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-vocab.txt",
        'bert-large-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-vocab.txt",
        'bert-base-multilingual-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-uncased-vocab.txt",
        'bert-base-multilingual-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased-vocab.txt",
        }

def main():
    output_folder = os.path.join('..', 'vocabs')
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    for _type, url in download_locations.items():
        f_path = os.path.join(output_folder, os.path.basename(url))
        print(f'Downloading type {_type} to {f_path}...')
        urllib.request.urlretrieve(url, f_path)

if __name__ == "__main__":
    main()
