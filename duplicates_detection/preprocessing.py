import pandas as pd
import nltk
from tabulate import tabulate
import pycountry
import re
from tqdm import tqdm

# nltk.download('stopwords')
train = pd.read_csv('train.csv')
top_N = 10
stopwords = nltk.corpus.stopwords.words('english')

RE_stopwords = r'\b(?:{})\b'.format('|'.join(stopwords))

words = (train.name_1
         .str.lower()
         .replace([r'\|', RE_stopwords], [' ', ''], regex=True)
         .str.cat(sep=' ')
         .split())
train.name_1 = train.name_1.str.lower()
train.name_2 = train.name_2.str.lower()
train.name_1 = train.name_1.str.strip()
train.name_2 = train.name_2.str.strip()
countries = [country.name.lower() for country in pycountry.countries]

train.replace(re.compile(r"\s+\(.*\)"), "", inplace=True)
train.replace(re.compile(r"\s+\[.*\]"), "", inplace=True)
train.replace(re.compile(r"[^\w\s]"), "", inplace=True)


def multi_str_replace(strings, debug=True):
    re_str = r'\b(?:' + '|'.join(
        [re.escape(s) for s in strings]
    ) + r')(?!\S)'
    if debug:
        print(re_str)
    return re.compile(re_str, re.UNICODE)


legal_entities = ['ltd.', 'ltd', 'unltd', 'ultd', 'lp', 'llp', 'lllp', 'llc', 'l.l.c.',
                  'pllc', 'co.', 'inc.', 'inc', 'b.v.', 'corp.', 'p.c.', 's.c.r.l.',
                  'r.l.', 'pvt.', 's.p.a', 'c.a.', 's.a.', 's.l.', 's.l.n.e.', 's.l.l.',
                  's.c.', 's.c.p.', 's.a.d.', 'sociedad', 'sociedade', 'cooperativa',
                  's.r.o.', 's a', 'c.v.', 'ооо', 'зао', 'пао', 'ао', 'нко',
                  '有限公司', '股份有限公司', '无限责任公司', '有限责任股份公司', 'ltda', 'sro'
                                                                  'kgaa', 'gmbh', 'e.v.', 'r.v.', 'mbh', 'ag',
                  'societe', 'sep', 'cv', 'ltda',
                  'snc', 'scs', 'sca', 'sci', 'sa', 'sas', 'sarl', 'societa', 'pvt', 'private',
                  'imp.', 'exp.', 'sanayi', 'co', 'kg', 's.a.i.c.', 'co.,', 's.a.c.', 'sac',
                  'saic', 'ev', 'rv', 'bv', 'pc', 'rl', 'spa', 'ca', 'sa', 'sl', 'slne', 'sll',
                  'sc', 'scp', 'sad', 'sociedad', 'sociedade', 'cooperativa',
                  's.r.o.', 's a', 'c.v.']
pattern = r'\b' + '|'.join(legal_entities) + r'\b'
train['name1'] = train['name_1'].str.replace(pattern, ' ')
train['name2'] = train['name_2'].str.replace(pattern, ' ')

print(train.name1.sample(10))
