# NLP Bahasa Indonesia Resources

This repository provides link to useful dataset and another resources for NLP in Bahasa Indonesia. 

*Last Update: 15 Mar 2022*

## Table of contents
  * [Corpus](#corpus)
    * [Named Entity Recognition](#named-entity-recognition)
    * [POS-Tagging](#pos-tagging)
    * [Question and Answering](#question-and-answering)
    * [Paraphrasing](#paraphrasing)
    * [Text Summarization](#text-summarization)
    * [Hate-speech](#hate-speech)
    * [Word Analogy](#word-analogy)
    * [Formal-Informal](#formal-informal)
    * [Multilingual Parallel](#multilingual-parallel)
    * [Unsupervised Corpus](#unsupervised-corpus)
    * [Voice-Text](#voice-text)
    * [Puisi and Pantun](#puisi-and-pantun)
 * [Dictionary](#dictionary) 
    * [Synonym](#synonym)
    * [Sentiment](#sentiment)
    * [Position or Degree](#position-or-degree)
    * [Root Words](#root-words)
    * [Slang Words](#slang-words)
    * [Stop Words](#stop-words)
    * [Swear Words](#swear-words)
    * [Composite Words](#composite-words)
    * [Number Words](#number-words)
    * [Calendar Words](#calendar-words)
    * [Emoticon](#emoticon)
    * [Acronym](#acronym)
    * [Indonesia Region](#indonesia-region)
    * [Country](#country)
    * [Region](#region)
    * [Title of Name](#title-of-name)
    * [Gender by Name](#gender-by-name)
    * [Organization](#organization)
*  [Articles and Papers](#articles-and-papers)
    * [POS-Tagging](#pos-tagging)
    * [Word Embedding](#word-embedding)
    * [Topic Analysis](#topic-analysis)
    * [Text Classification](#text-classification)
*  [Pre-trained Models](#pre-trained-models)
*  [Usable Library](#usable-library)
*  [Spelling Correction](#spelling-correction)
*  [Twitter Scraping](#twitter-scrapping)
*  [Other Resources](#other-resourceS)


## [Corpus](corpus)

### [Named Entity Recognition](corpus/named-entity-recognition)
1) Product NER. https://github.com/dziem/proner-labeled-text
2) NER-grit. https://github.com/grit-id/nergrit-corpus

### [POS-Tagging](corpus/pos-tagging)
1) IDN Tagged Corpus. https://github.com/famrashel/idn-tagged-corpus
2) Indonesian Part-of-Speech (POS) Tagging. https://github.com/kmkurn/id-pos-tagging/blob/master/data/dataset.tar.gz

### [Question and Answering](corpus/question-and-answering)
1) TydiQA. https://github.com/google-research-datasets/tydiqa

### [Paraphrasing](corpus/paraphrasing)
1) Quora Paraphrasing. https://github.com/louisowen6/quora_paraphrasing_id
2) Paraphrase Adversaries from Word Scrambling. https://github.com/Wikidepia/indonesian_datasets/tree/master/paraphrase/paws

### [Text Summarization](corpus/text-summarization)
1) Indosum. https://github.com/kata-ai/indosum
2) Liputan6. https://huggingface.co/datasets/id_liputan6

### [Hate-speech](corpus/hate-speech)
1) ID Multi Label Hate Speech. https://github.com/okkyibrohim/id-multi-label-hate-speech-and-abusive-language-detection

### [Word Analogy](corpus/word-analogy)
1) KAWAT. https://github.com/kata-ai/kawat

### [Formal-Informal](corpus/formal-informal)
1) STIF-Indonesia. https://github.com/haryoa/stif-indonesia
2) IndoCollex. https://github.com/haryoa/indo-collex
3) https://github.com/okkyibrohim/id-multi-label-hate-speech-and-abusive-language-detection/blob/master/new_kamusalay.csv

### [Multilingual Parallel](corpus/multilingual-parallel)
1) https://huggingface.co/datasets/alt
2) https://opus.nlpl.eu/bible-uedin.php
3) http://www.statmt.org/cc-aligned/
4) https://huggingface.co/datasets/id_panl_bppt
5) https://huggingface.co/datasets/open_subtitles
6) https://huggingface.co/datasets/opus100
7) https://huggingface.co/datasets/tapaco
8) https://huggingface.co/datasets/wiki_lingua

### [Unsupervised Corpus](corpus/unsupervised-corpus)
1) OSCAR. https://oscar-corpus.com/
2) Online Newspaper. https://github.com/feryandi/Dataset-Artikel
3) IndoNLU. https://huggingface.co/datasets/indonlu
4) IndoNLG. https://github.com/indobenchmark/indonlg
5) IndoNLI. https://github.com/ir-nlp-csui/indonli
6) IndoBERTweet. https://github.com/indolem/IndoBERTweet
7) http://data.statmt.org/cc-100/
8) https://huggingface.co/datasets/id_clickbait
9) https://huggingface.co/datasets/id_newspapers_2018
10) https://opus.nlpl.eu/QED.php

### [Voice-Text](corpus/voice-text)
1) https://huggingface.co/datasets/common_voice
2) https://huggingface.co/datasets/covost2

### [Puisi and Pantun](corpus/puisi-and-pantun)
1) https://github.com/ilhamfp/puisi-pantun-generator


## [Dictionary](dictionary)

### [Synonym](dictionary/synonym)
1) https://github.com/victoriasovereigne/tesaurus

### [Sentiment](dictionary/sentiment)
1) (Negative) https://github.com/ramaprakoso/analisis-sentimen/blob/master/kamus/negatif_ta2.txt
2) (Negative) https://github.com/ramaprakoso/analisis-sentimen/blob/master/kamus/negative_add.txt
3) (Negative) https://github.com/ramaprakoso/analisis-sentimen/blob/master/kamus/negative_keyword.txt
4) (Negative) https://github.com/masdevid/ID-OpinionWords/blob/master/negative.txt
5) (Positive) https://github.com/ramaprakoso/analisis-sentimen/blob/master/kamus/positif_ta2.txt
6) (Positive) https://github.com/ramaprakoso/analisis-sentimen/blob/master/kamus/positive_add.txt
7) (Positive) https://github.com/ramaprakoso/analisis-sentimen/blob/master/kamus/positive_keyword.txt
8) (Positive) https://github.com/masdevid/ID-OpinionWords/blob/master/positive.txt
9) (Score) https://github.com/agusmakmun/SentiStrengthID/blob/master/id_dict/sentimentword.txt
10) (InSet Lexicon) https://github.com/fajri91/InSet [[Paper](https://www.researchgate.net/publication/321757985_InSet_Lexicon_Evaluation_of_a_Word_List_for_Indonesian_Sentiment_Analysis_in_Microblogs)]
11) (Twitter Labelled Sentiment) https://www.researchgate.net/profile/Ridi_Ferdiana/publication/339936724_Indonesian_Sentiment_Twitter_Dataset/data/5e6d64c6a6fdccf994ca18aa/Indonesian-Sentiment-Twitter-Dataset.zip?origin=publicationDetail_linkedData [[Paper](https://www.researchgate.net/publication/338409000_Dataset_Indonesia_untuk_Analisis_Sentimen)]
12) https://huggingface.co/datasets/senti_lex

### [Position or Degree](dictionary/position-or-degree)
1) https://github.com/panggi/pujangga/blob/master/resource/netagger/contextualfeature/psuf.txt
2) https://github.com/panggi/pujangga/blob/master/resource/netagger/contextualfeature/lldr.txt
3) https://github.com/panggi/pujangga/blob/master/resource/netagger/contextualfeature/opos.txt
4) https://github.com/panggi/pujangga/blob/master/resource/netagger/contextualfeature/ptit.txt

### [Root Words](dictionary/root-words)
1) https://github.com/agusmakmun/SentiStrengthID/blob/master/id_dict/rootword.txt
2) https://github.com/sastrawi/sastrawi/blob/master/data/kata-dasar.original.txt
3) https://github.com/sastrawi/sastrawi/blob/master/data/kata-dasar.txt
4) https://github.com/prasastoadi/serangkai/blob/master/serangkai/kamus/data/kamus-kata-dasar.csv

I have made the [combined root words list](https://github.com/louisowen6/NLP_bahasa_resources/blob/master/combined_root_words.txt) from all of the above repositories.
 
### [Slang Words](dictionary/slang-words)
1) https://github.com/ramaprakoso/analisis-sentimen/blob/master/kamus/kbba.txt
2) https://github.com/agusmakmun/SentiStrengthID/blob/master/id_dict/slangword.txt
3) https://github.com/panggi/pujangga/blob/master/resource/formalization/formalizationDict.txt

I have made the [combined slang words dictionary](https://github.com/louisowen6/NLP_bahasa_resources/blob/master/combined_slang_words.txt) from all of the above repositories.

### [Stop Words](dictionary/stop-words)
1) https://github.com/yasirutomo/python-sentianalysis-id/blob/master/data/feature_list/stopwordsID.txt
2) https://github.com/ramaprakoso/analisis-sentimen/blob/master/kamus/stopword.txt
3) https://github.com/abhimantramb/elang/tree/master/word2vec/utils/stopwords-list

I have made the [combined stop words list](https://github.com/louisowen6/NLP_bahasa_resources/blob/master/combined_stop_words.txt) from all of the above repositories.

### [Swear Words](dictionary/swear-words)
1) https://github.com/abhimantramb/elang/blob/master/word2vec/utils/swear-words.txt

### [Composite Words](dictionary/composite-words)
1) https://github.com/panggi/pujangga/blob/master/resource/tokenizer/compositewords.txt

### [Number Words](dictionary/number-words)
1) https://github.com/panggi/pujangga/blob/master/resource/netagger/morphologicalfeature/number.txt

### [Calendar Words](dictionary/calendar-words)
1) https://github.com/onlyphantom/elang/blob/master/build/lib/elang/word2vec/utils/negative/calendar-words.txt

### [Emoticon](dictionary/emoticon)
1) https://github.com/ramaprakoso/analisis-sentimen/blob/master/kamus/emoticon.txt
2) https://github.com/jolicode/emoji-search/blob/master/synonyms/cldr-emoji-annotation-synonyms-id.txt
3) https://github.com/agusmakmun/SentiStrengthID/blob/master/id_dict/emoticon.txt

### [Acronym](dictionary/acronym)
1) https://github.com/ramaprakoso/analisis-sentimen/blob/master/kamus/acronym.txt
2) https://github.com/panggi/pujangga/blob/master/resource/sentencedetector/acronym.txt
3) https://id.wiktionary.org/wiki/Lampiran:Daftar_singkatan_dan_akronim_dalam_bahasa_Indonesia#A

### [Indonesia Region](dictionary/indonesia-region)
1) https://github.com/abhimantramb/elang/blob/master/word2vec/utils/indonesian-region.txt
2) https://github.com/edwardsamuel/Wilayah-Administratif-Indonesia/tree/master/csv
3) https://github.com/pentagonal/Indonesia-Postal-Code/tree/master/Csv

### [Country](dictionary/country)
1) https://github.com/panggi/pujangga/blob/master/resource/netagger/contextualfeature/country.txt

### [Region](dictionary/region)
1) https://github.com/panggi/pujangga/blob/master/resource/netagger/contextualfeature/lpre.txt

### [Title of Name](dictionary/title-of-name)
1) https://github.com/panggi/pujangga/blob/master/resource/netagger/contextualfeature/ppre.txt

### [Gender by Name](dictionary/gender-by-name)
1) https://github.com/seuriously/genderprediction/blob/master/namatraining.txt

### [Organization](dictionary/organization)
1) https://github.com/panggi/pujangga/blob/master/resource/reference/opre.txt

## [Articles and Papers](articles-and-papers)

### [POS-Tagging](articles-and-papers/pos-tagging)
1) https://medium.com/@puspitakaban/pos-tagging-bahasa-indonesia-dengan-flair-nlp-c12e45542860
2) Manually Tagged Indonesian Corpus [[Paper](http://bahasa.cs.ui.ac.id/postag/downloads/Designing%20an%20Indonesian%20Part%20of%20speech%20Tagset.pdf)] [[GitHub](https://github.com/famrashel/idn-tagged-corpus)]

### [Word Embedding](articles-and-papers/word-embedding)
1) (FastText). https://structilmy.com/2019/08/membuat-model-word-embedding-fasttext-bahasa-indonesia/
2) (Word2Vec). https://yudiwbs.wordpress.com/2018/03/31/word2vec-wikipedia-bahasa-indonesia-dengan-python-gensim/

### [Topic Analysis](articles-and-papers/topic-analysis)
1) (Introduction to LSA & LDA). https://monkeylearn.com/blog/introduction-to-topic-modeling/
2) (Introduction to LDA w/ Code & Tips). https://www.analyticsvidhya.com/blog/2016/08/beginners-guide-to-topic-modeling-in-python/
3) (Topic Modeling Methods Comparison Paper). https://thesai.org/Downloads/Volume6No1/Paper_21-A_Survey_of_Topic_Modeling_in_Text_Mining.pdf
4) (Original LDA Paper). http://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf
5) (LDA Python Library). https://pypi.org/project/lda/; https://radimrehurek.com/gensim/models/ldamodel.html; https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.LatentDirichletAllocation.html
6) (Original CTM Paper). http://people.ee.duke.edu/~lcarin/Blei2005CTM.pdf
7) (CTM Python Library). https://pypi.org/project/tomotopy/; https://github.com/kzhai/PyCTM
8) (Gaussian LDA Paper). https://www.aclweb.org/anthology/P15-1077.pdf
9) (Gaussian LDA Library). https://github.com/rajarshd/Gaussian_LDA
10) (Temporal Topic Modeling Comparison Paper). https://thesai.org/Downloads/Volume6No1/Paper_21-A_Survey_of_Topic_Modeling_in_Text_Mining.pdf
11) (TOT: A Non-Markov Continuous-Time Model of Topical Trends Paper). https://people.cs.umass.edu/~mccallum/papers/tot-kdd06s.pdf
12) (TOT Library). https://github.com/ahmaurya/topics_over_time  
13) (Example of LDA in Bahasa Project Code). https://github.com/kirralabs/text-clustering

### [Text Classification](articles-and-papers/text-classification)
#### Zero-shot Learning
1) (Benchmarking Zero-shot Text Classification: Datasets, Evaluation and Entailment Approach) https://arxiv.org/pdf/1909.00161.pdf | https://github.com/yinwenpeng/BenchmarkingZeroShot
2) (Integrating Semantic Knowledge to Tackle Zero-shot Text Classification) https://arxiv.org/abs/1903.12626 | https://github.com/JingqingZ/KG4ZeroShotText
3) (Train Once, Test Anywhere: Zero-Shot Learning for Text Classification) https://arxiv.org/abs/1712.05972 | https://amitness.com/2020/05/zero-shot-text-classification/
4) (Zero-shot Text Classification With Generative Language Models) https://arxiv.org/abs/1912.10165 | https://amitness.com/2020/06/zero-shot-classification-via-generation/
5) (Zero-shot User Intent Detection via Capsule Neural Networks) https://arxiv.org/abs/1809.00385 | https://github.com/congyingxia/ZeroShotCapsule

#### Few-shot Learning
1) (Few-shot Text Classification with Distributional Signatures) https://arxiv.org/pdf/1908.06039.pdf | https://github.com/YujiaBao/Distributional-Signatures
2) (Few Shot Text Classification with a Human in the Loop) https://katbailey.github.io/talks/Few-shot%20text%20classification.pdf | https://github.com/katbailey/few-shot-text-classification
3) (Induction Networks for Few-Shot Text Classification) https://arxiv.org/pdf/1902.10482v2.pdf | https://github.com/zhongyuchen/few-shot-learning

## [Pre-trained Models](pre-trained-models)
1) Indo-BERT. https://github.com/indobenchmark/indonlu & https://huggingface.co/indobenchmark/indobert-base-p1
2) Indo-BERTweet. https://github.com/indolem/IndoBERTweet & https://huggingface.co/indolem/indobertweet-base-uncased
3) Transformer-based Pre-trained Model in Bahasa. https://github.com/cahya-wirawan/indonesian-language-models/tree/master/Transformers
4) Generate Word-Embedding / Sentence-Embedding using pre-Trained Multilingual Bert model. (https://colab.research.google.com/drive/1yFphU6PW9Uo6lmDly_ud9a6c4RCYlwdX#scrollTo=Zn0n2S-FWZih). P.S: Just change the model using 'bert-base-multilingual-uncased'
5) https://github.com/meisaputri21/Indonesian-Twitter-Emotion-Dataset. [[Paper](https://www.researchgate.net/publication/330674171_Emotion_Classification_on_Indonesian_Twitter_Dataset/link/5c4ea13a458515a4c745850d/download)]
6) https://github.com/Kyubyong/wordvectors
7) https://drive.google.com/uc?id=0B5YTktu2dOKKNUY1OWJORlZTcUU&export=download
8) https://github.com/deryrahman/word2vec-bahasa-indonesia
9) https://sites.google.com/site/rmyeid/projects/polyglot

## [Usable Library](usable-library)
1) Pujangga: Indonesian Natural Language Processing REST API. https://github.com/panggi/pujangga 
2) Sastrawi Stemmer Bahasa Indonesia. https://github.com/sastrawi/sastrawi
3) NLP-ID. https://github.com/kumparan/nlp-id
4) MorphInd: Indonesian Morphological Analyzer. http://septinalarasati.com/morphind/
5) INDRA: Indonesian Resource Grammar. https://github.com/davidmoeljadi/INDRA
6) Typo Checker. https://github.com/mamat-rahmat/checker_id
7) Multilingual NLP Package. https://github.com/flairNLP/flair
9) spaCy [[GitHub](https://github.com/explosion/spaCy)] [[Tutorial](https://bagas.me/spacy-bahasa-indonesia.html)]
9) https://github.com/yohanesgultom/nlp-experiments
10) https://github.com/yasirutomo/python-sentianalysis-id
11) https://github.com/riochr17/Analisis-Sentimen-ID
12) https://github.com/yusufsyaifudin/indonesia-ner

## [Spelling Correction](spelling-correction)
You can adjust [this code](https://norvig.com/spell-correct.html?utm_medium=social&utm_source=linkedin&utm_campaign=postfity&utm_content=postfity50031) with Bahasa corpus to do the spelling correction

## [Twitter Scraping](twitter-scrapping)
1) GetOldTweets3. https://github.com/Mottl/GetOldTweets3

Usage:
```bash
import GetOldTweets3 as got
tweetCriteria=got.manager.TweetCriteria().setQuerySearch('#CoronaVirusIndonesia').setSince("2020-01-01").setUntil("2020-03-05").setNear("Jakarta, Indonesia").setLang("id")
tweets=got.manager.TweetManager.getTweets(tweetCriteria)
for tweet in tweets:
	print(tweet.username)
	print(tweet.text)
	print(tweet.date)
	print("tweet.to")
	print("tweet.retweets")
	print("tweet.favorites")
	print("tweet.mentions")
	print("tweet.hashtags")
	print("tweet.geo")
 ```

2) Tweepy. http://docs.tweepy.org/en/latest/

Step-by-step how to use Tweepy. https://towardsdatascience.com/how-to-scrape-tweets-from-twitter-59287e20f0f1

Sign in to Twitter Developer. https://developer.twitter.com/en

Full List of Tweets Object. https://developer.twitter.com/en/docs/tweets/data-dictionary/overview/tweet-object

Increasing Tweepy’s standard API search limit. https://bhaskarvk.github.io/2015/01/how-to-use-twitters-search-rest-api-most-effectively./

## [Other Resources](other-resourceS)
1) https://github.com/indonesian-nlp/nlp-resources
2) https://github.com/irfnrdh/Awesome-Indonesia-NLP
3) https://github.com/kirralabs/indonesian-NLP-resources
4) https://huggingface.co/datasets?filter=languages%3Aid&p=0
