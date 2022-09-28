imgs = os.listdir('./ann')
newimgs = [i for i in imgs if len(i)>13]
randomimgs = random.sample(newimgs, 500)
import random
import shutil
!cp -r ../input/nomeroff-russian-license-plates/autoriaNumberplateOcrRu-2021-09-01/train/ann ./
mkdir /kaggle/working/anns/
for i in randomimgs:
    print(i.split('.json'))
len(os.listdir('./imgs'))
for i in randomimgs:
    shutil.move('./img/' + i.split('.json')[0] + '.png', './imgs')
for i in randomimgs:
    shutil.move('./ann/' + i, './anns')