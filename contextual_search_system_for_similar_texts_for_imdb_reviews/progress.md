*polishing BoW model (5% dataset):*
- initital: best min dist = 9.11
- better text pre-proccesing: best min dist = 9.11 , voc size = 29740
- added text post-processing (removing rare words, removing stop words, and applying stemming) : best min dist = 5.29, voc size = 10086
- adjusted word min_frequency to 5: best min dist = 5.10, voc size = 5337
- bug fix + code optimization: best min dist = 6.78, voc size = 5337