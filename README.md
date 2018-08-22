# agaadha-nama: Yoga Asana Sanskrit Classification with Deep Learning in Keras / Tensorflow

- agaadha (अगाध) very deep/supportless/profound/unfathomed
- nama (नाम) name

### Project Description

My yoga teacher training mentor, Anirudh Shastri, once informed that I will never be respected as a yoga teacher if I don't learn the Sanskrit terms. 8 years later, I still don't know the asana terms for most of the poses I teach. But it's okay because "Hey you, over there!"-asana always lightens up a class atmosphere.

My goal with this project is to train a model to be able to classify a given photo as a certain Sanskrit yoga asana. 

### Obtaining Images

No database of asana photos exists. The first step si to scrape the first 1000 Google Image Search Results for a given asan. To do this, I used [image_search](https://github.com/rushilsrivastava/image_search). 

My intial attmpt to train a model (based on my work on [digit-recognizer](https://github.com/lubagloukhova/digit-recognizer) revealed that the scraped images need some QA -- we need to identify which images depict actual asanas and which just happened to make it into the sanskrit term search results. Two options exist: (1) leverage mechanical turk to leverage human labeling of these photos as asana or non-asana, or (2) leverage deep learning image similarity to cluster the asana photos together and weed out the unrelated search results. 

### Weeding out non-asana Images

Examining random images for a given asana revealed that a fairly hgih proportion of the images were of too poor a quality to contribute to the accuracy of a model as positive class samples. There are two ways to identify and weed out these poor-quality images:
    - 1. human effort (MTurk?)
    - 2. feture extraction & clustering
    

DB Scan provide a way to cluster images in a flexible enough way to support this use case. However, initial attempts to use the algorithm on a computed cosine similarity matrix of the extracted features failed to extract outliers and resulted in core and non-core samples that didn't differ from eachother in any noticeable way (ay variou slevels of eps and min_samples)


### Model Building

### Results

### Next Steps 

