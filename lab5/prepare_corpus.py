import os
import tarfile
import email
import re
import scipy.io as sio
from html import unescape
from email import parser
from email.policy import default
from collections import Counter
from six.moves import urllib
from lab5 import TextPreprocessor, convert_text_to_feature_vector
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))
preprocessor = TextPreprocessor()

DOWNLOAD_ROOT = "http://spamassassin.apache.org/old/publiccorpus/"
HAM_URL = DOWNLOAD_ROOT + "20030228_easy_ham.tar.bz2"
SPAM_URL = DOWNLOAD_ROOT + "20030228_spam.tar.bz2"
SPAM_PATH = os.path.join("data", "spam")
VOCAB_PATH = os.path.join("data", "vocab2.txt")
TRAIN_PATH = os.path.join("data", "customTrain.mat")
TEST_PATH = os.path.join("data", "customTest.mat")


def is_ascii(s):
    return all(ord(c) < 128 for c in s)


def download_spam_data():
    for filename, url in (("ham.tar.bz2", HAM_URL), ("spam.tar.bz2", SPAM_URL)):
        path = os.path.join(SPAM_PATH, filename)
        urllib.request.urlretrieve(url, path)
        with tarfile.open(path) as spam_file:
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(spam_file, path=SPAM_PATH)


def load_email(is_spam, filename, spam_path=SPAM_PATH):
    directory = "spam" if is_spam else "easy_ham"
    with open(os.path.join(spam_path, directory, filename), "rb") as f:
        return parser.BytesParser(policy=email.policy.default).parse(f)


def get_email_structure(email_object):
    if isinstance(email_object, str):
        return email_object
    payload = email_object.get_payload()
    if isinstance(payload, list):
        mail_parts = [get_email_structure(sub_email) for sub_email in payload]
        return "multipart({})".format(", ".join(mail_parts))
    else:
        return email_object.get_content_type()


def structures_counter(emails):
    structures = Counter()
    for mail in emails:
        structure = get_email_structure(mail)
        structures[structure] += 1
    return structures


def html_to_plain_text(html):
    text = re.sub('<head.*?>.*?</head>', '', html, flags=re.M | re.S | re.I)
    text = re.sub(r'<a\s.*?>', ' httpaddr ', text, flags=re.M | re.S | re.I)
    text = re.sub('<.*?>', '', text, flags=re.M | re.S)
    text = re.sub(r'(\s*\n)+', '\n', text, flags=re.M | re.S)
    return unescape(text)


def email_to_text(email_object):
    html = None
    for part in email_object.walk():
        content_type = part.get_content_type()
        if content_type not in ["text/plain", "text/html"]:
            continue
        try:
            content = part.get_content()
        except:  # in case of encoding issues
            content = str(part.get_payload())
        if content_type == "text/plain":
            return content
        else:
            html = content
    if html:
        return html_to_plain_text(html)


def prepare_dataset(spam_mails, ham_mails, mapping):
    x = []
    y = []

    for spam_mail in spam_mails:
        spam_text = email_to_text(spam_mail)
        if spam_text:
            spam_vector = convert_text_to_feature_vector(spam_text, mapping)
            x.append(spam_vector.flatten())
            y.append(1)

    for ham_mail in ham_mails:
        ham_text = email_to_text(ham_mail)
        if ham_text:
            x.append(convert_text_to_feature_vector(ham_text, mapping).flatten())
            y.append(0)
    return x, y


def download_and_prepare_data():
    download_spam_data()
    ham_dir = os.path.join(SPAM_PATH, "easy_ham")
    spam_dir = os.path.join(SPAM_PATH, "spam")
    ham_filenames = os.listdir(ham_dir)
    spam_filenames = os.listdir(spam_dir)
    ham_emails = [load_email(False, filename) for filename in ham_filenames]
    spam_emails = [load_email(True, filename) for filename in spam_filenames]

    vocab = []
    for spam_mail in spam_emails:
        email_text = email_to_text(spam_mail)
        if not email_text:
            continue
        text = preprocessor.preprocess_text(email_text)
        words = [w for w in text.split(' ') if w not in stop_words and is_ascii(w)]

        word_counts = Counter(words)
        vocab.append(word_counts)

    vocab = sum(vocab, Counter())

    most_common = vocab.most_common(1899)
    vocab = []
    for (k, v) in most_common:
        vocab.append(k)

    vocab = [(index + 1, word) for index, word in enumerate(sorted(vocab))]
    mapping = {}
    with open(VOCAB_PATH, 'w+') as f:
        for index, word in vocab:
            f.write("%s\t%s\n" % (index, word))
            mapping[word] = index

    ham_count, spam_count = len(ham_filenames), len(spam_filenames)

    x_train, y_train = prepare_dataset(
        spam_emails[:int(spam_count * 0.8)],
        ham_emails[:int(ham_count * 0.8)], mapping)
    sio.savemat(TRAIN_PATH, {'X': x_train, 'y': y_train})

    x_test, y_test = prepare_dataset(
        spam_emails[int(spam_count * 0.8):],
        ham_emails[int(ham_count * 0.8):], mapping)
    sio.savemat(TEST_PATH, {'Xtest': x_test, 'ytest': y_test})


if __name__ == "__main__":
    download_and_prepare_data()
