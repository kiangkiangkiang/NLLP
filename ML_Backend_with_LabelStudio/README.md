# Machine Learning Backend on Label Studio

本篇將介紹如何透過ML模型，幫助Label Studio標記。

實作範例請參考「toturial.ipynb」。

## 0. 前言

Label Studio [[1]](https://labelstud.io/) 為目前機器學習常見的標記工具，能透過簡單的UI介面完成大部分機器學習的標記任務，包括Object Dectection, Classification, Named Entity Recognition (NER), Relation Extraction (RE) 等等。

然而標記任務需耗費大量人力成本，因此 Label Studio 也提供 **Machine Learning Backend**，使模型推論（預測）的結果能夠在 Label Studio 上呈現，以此降低人工標記成本，使人類只需負責判斷模型的推論結果。

### 應用場景

本篇則以「RE」任務為範例，應用場景是希望能夠在一篇判決書中，找到**精神慰撫金**的金額」，並且希望能正確對應到「是哪位原告提出的」，藉此示範如何結合Label Studio, ML Backend來處理NLP任務。

--- 

## 1. 環境/套件版本
- Python 3.8
- Ubuntu 20.04
- paddlenlp 2.5.1
- paddlepaddle 2.3.2
- label-studio 1.7.2 
- label-studio-ml 1.0.9 

---
## 2. 資料

資料來源可以參考[這裡](https://github.com/kiangkiangkiang/NLLP/tree/main/data)，主要來自**臺灣司法院的公開資料集內提供的判決書**，詳細資料來源、取得方式、涵蓋範圍等等皆可參考上述連結。

---
## 3. 模型介紹

由於本篇處理的NLP任務是以繁體中文為主，而中文的NLP任務的模型常見於以下：
1. 臺灣中研院CKIP [[2]](https://ckip.iis.sinica.edu.tw/)
2. 中國科大訊飛 iFLYTEK Research [[3]](https://github.com/ymcui/Chinese-BERT-wwm)
3. 中國百度 PaddleNLP [[4]](https://github.com/PaddlePaddle/PaddleNLP)
4. 中國 JioNLP [[5]](https://github.com/dongrixinyu/JioNLP)
5. 其他更多中文NLP的整理 [[6]](https://github.com/crownpku/Awesome-Chinese-NLP)

而本篇的範例中選用PaddleNLP中的 Universal Information Extraction (UIE) [[7]](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/model_zoo/uie) 模型作為ML Backend的模型。

---
## 4. 參考資料

- [1] https://labelstud.io/
- [2] https://ckip.iis.sinica.edu.tw/
- [3] https://github.com/ymcui/Chinese-BERT-wwm
- [4] https://github.com/PaddlePaddle/PaddleNLP
- [5] https://github.com/dongrixinyu/JioNLP
- [6] https://github.com/crownpku/Awesome-Chinese-NLP
- [7] https://github.com/PaddlePaddle/PaddleNLP/tree/develop/model_zoo/uie
