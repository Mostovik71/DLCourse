import pandas as pd
import os
import json
from natsort import natsorted
# df = pd.DataFrame([])
# anns = os.listdir('OCRAUTO/anns')
# imgs = os.listdir('OCRAUTO/imgs')
imgsnew = os.listdir('OCRAUTO/imgsnew')
# # Переименование файлов
path1 = 'OCRAUTO/imgs/'
path2 = 'OCRAUTO/imgsnew/'
for i in range(len(imgs)):
     print(imgs[i])
     os.rename(os.path.join(path1, imgs[i]), os.path.join(path2, str(i) + '.jpg'))

for i in anns:
    with open('OCRAUTO/anns/' + i, "r") as read_file:
        data = json.load(read_file)
    df = df.append(pd.concat([pd.Series(i), pd.Series(data['description'])], axis=1))
df.columns = ['filename', 'words']
df['filename'] = df['filename'].str.replace('.json', '.png')
df.to_csv('labels.csv', index=False)
df = pd.read_csv('labels.csv')
df['filename'] = pd.Series(natsorted(imgsnew))
print(df.head())
df.to_csv('labelsnew.csv', index=False)
