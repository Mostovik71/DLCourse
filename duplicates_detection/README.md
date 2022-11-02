<h1>Company duplicates detection</h1>

Bert only, full dataset

<img src="img/4.jpg">

Cosine sentence similarity model

<img src="img/1.jpg">

<h4>Cosine sentence similarity model + BERT full dataset

<img src="img/2.jpg">

<hr>

<h3> Метрики </h3>
<ul> <b> Cosine sentence similarity model + BERT full dataset </b>
  <li>Recall: 0.99</li>
  <li>Precicison: 0.89</li>
  <li>F1: 0.96</li>
</ul>

<h3> Гиперпараметры </h3>
<ul> <b> Гиперпараметры подобраны опытным путем </b>
  <li>Learning rate: 2e-5</li>
  <li>Epochs: 3</li>
  <li>Optimizer: AdamW (full model.params)</li>
  <li>Batch size: 16</li>
</ul>  
<hr>
<h3> Характеристики </h3>
  <li>CPU </li>
  <li>Первая итерация ± 50 comps/sec</li>
  <li>1 запрос: 7 sec </li>
<h3> Models </h3>
Модели здесь https://drive.google.com/drive/folders/1mRV56wwNSQTdkFlSU-Wd8Jt920c7uhh9?usp=share_link
<h3> Использование </h3> 
<li>Загрузить основной ноутбук (Cos_filter_...). Пользоваться лучше Google Colab, т.к. модели и данные на ходятся на гугл - диске.</li>
<li> Создать ярлык на своем гугл - диске, для того чтобы пользоваться обученными моделями.</li>
Когда эти действия выполнены, можно тестировать модель.
