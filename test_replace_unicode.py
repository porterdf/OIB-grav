"""
some snippets to find and replace unicode characters in python
"""

### single instance
word = u'Buffalo,\xa0IL\xa060625'
word.replace(u'\xa0', ' ')
word.replace(u'\xa0', u' ') # replaced with space
word.replace(u'\xa0', u'0') # closest to what you were literally asking for
word.replace(u'\xa0', u'')  # removed completely
## decoding
word.encode('ascii','ignore')


### Pandas
## test data
t = "We've\xe5\xcabeen invited to attend TEDxTeen, an independently organized TED event focused on encouraging youth to find \x89\xdb\xcfsimply irresistible\x89\xdb\x9d solutions to the complex issues we face every day.,"
t2 = t.decode('unicode_escape').encode('ascii', 'ignore').strip()
import sys
sys.stdout.write(t2.strip('\n\r'))
df = pd.DataFrame([t,t,t],columns = ['text'])

## lambda
df['text'] = map(lambda x: x.decode('utf-8').replace(u'\xa0', ' '), df['text'].str)
df['text'] = df['text'].str.replace(u'\xa0', ' ')

df['text'] = df['text'].apply(lambda x: x.decode('unicode_escape').\
                                          encode('ascii', 'ignore').\
                                          strip())
## a function
def clean_text(row):
    # return the list of decoded cell in the Series instead 
    return [r.decode('unicode_escape').encode('ascii', 'ignore') for r in row]
df['text'] = df.apply(clean_text)
df["text"] = df.apply(clean_text, axis=1)

## regex
df["text"] =  df.text.str.replace('[^\x00-\x7F]','')


## basis for a search/replace script
import fileinput
import re
 
for line in fileinput.input(inplace=1, backup='.bak'):
    line = re.sub('foo','bar', line.rstrip())
    print(line)