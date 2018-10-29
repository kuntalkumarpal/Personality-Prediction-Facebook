# Personality-Prediction-Facebook
Predicting facebook-users' personality based on status and linguistic features via flexible regression analysis techniques

This contains the implementation of the paper [Predicting facebook-users' personality based on status and linguistic features via flexible regression analysis techniques](https://dl.acm.org/citation.cfm?id=3167166)

### Data :
* [MyPersonality](https://sites.google.com/michalkosinski.com/mypersonality)
* We used a cutdown version of the data which had 48701 records of unique userid

### Requirements
* Python 2.7
* Numpy


### Files :
* final_merged_status.csv - 48701 userids along with all their statuses merged together
* final_liwc.csv - 48701 userids along with their LIWC features
* final_big5.csv - 48701 userids along with their BIG5 peronality scores (Gold Standard for our work, calculated by the dataset creators based on answers to questionnaires)

* reg_input_<lda_alpha_param>.csv - Merged userid-topics(generated by LDA), gold-standard big5 scores(final_big5.csv), LIWC features (final_liwc.csv)

* getTopics.py - Uses LDA(gensim library) to create topics from Facebook statuses. Parameters are hardcoded. Can be changed to generate various files final_id_topics.csv (Filename hardcoded)
* Calc_BIG5_SvrPoly_only_Topics.py - The main code to generate the BIG5 scores based on only Topics (SVR Polynomial kernels)
* Calc_BIG5_SvrPoly_Topics_Liwc.py - The main code to generate the BIG5 scores based on Topics along with LIWC features (SVR Polynomial kernels)
* Calc_BIG5_SvrPoly_Topics_Liwc_DecTree.py - The main code to generate the BIG5 scores based on only Topics (Decision Trees)
* Calc_BIG5_SvrPoly_only_Topics_DecTree.py - The main code to generate the BIG5 scores based on Topics along with LIWC features (Decision Trees)

### Note :
* The code have various hardcoding.
* Parameter setup is present in the files itself.
* The code was written some time  back and was not at all maintained since then. Now its being arranged from the bits and pieces found. So there may be inconsistencies
* The code is not well structured and designed in some areas.
* No separate training, validations and testing modules all are done sequentially together in same function
* The naming of the files are inconsistent.