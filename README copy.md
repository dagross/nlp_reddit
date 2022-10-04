# ![](https://ga-dash.s3.amazonaws.com/production/assets/logo-9f88ae6c9c3871690e33280fcf557f33.png) Project 3: Web APIs & NLP

### Problem Statement

Our marketing company is looking for a way to improve effectiveness of advertising campaignes running on Reddit. For that purpose stakeholders would like to know is there a significant difference on how people adress man and woman. If such difference exist, this could greatly improve targeting algorithms. As a data scientist at this company, I have been tasked to solve this issue by building a classifying models and find keywords, that are specific for each audince if there are so. Measure of success of this project will be resulting accuracy score that should be higher than 40% of baseline accuracy. This model will be evaluated using accuracy rate. Resulting score should be as high as possible, but not lower than 70, which is 20 points higher than baseline accuracy.

threshhold on model performance, ceiling, getting stuck

## Brief Summary of Analysis and Interpretations
For solving the stated problem, I've decided to parce and download data from to subreddits: r/AskWoman and r/AskMan, that complitely satisfies audiency requirements. After EDA and basic cleaning using RegExp, I've started building models.
Here are the results of first iteration of work:

|Model|Train Score|Test Score|
|:---:|:---:|:---:|
|Stacking with CountVectorizer |0.9902692182938696| <dev>0.6303501945525292<dev>|
|AdaBoost + TFIF               |0.7697048329549141| <dev>0.6313229571984436<dev>|
|Randon Forest +TFIDF          |0.957833279273435 | <dev>0.6575875486381323<dev>|
|Bagging+TFIDF                 |0.9938371715861174| <dev>0.6186770428015564<dev>|
|DecisionTree + TFIDF          |0.727538112228349 | <dev>0.6011673151750972<dev>|
|Naive Bayes + CountVectorizer |0.7745702238079792| <dev>0.5787937743190662<dev>|
|Logistic Regression           |0.9072332144015569| <dev>0.603112840466926<dev>|
Despite the fact, that training score looks good for some models, testing are low, concidering that baseline accuracy is 50 percent.  All the models struggling to generale to new data, that is why all models are very overfitted.
Second iteration was condacted after additional NLP manipulation on the data. I've used SpaCy for lemmetizing, finding StopWords and cleaning auxiliary words instead of WordNetLemmatizer and RegExp. All the modeling here was conducted on a BoW (Bag of Words).
<dev>Note<dev> After dropping cleaning and dropping auxiliary words, baseline accuracy score was changed to <dev>0.582527<dev>.

Second stage results:
|Model|Train Score|Test Score|
|:---:|:---:|:---:|
|Stacking with CountVectorizer |0.6386243914973804 |0.5950749464668095|
|AdaBoost + TFIF               |0.6461905237762138 |0.5838115631691649|
|Random Forest +TFIDF          |0.6461476966123714 |0.5862098501070664|
|Bagging+TFIDF                 |0.6449628117460635 |0.586252676659529|
|DecisionTree + TFIDF          |0.5889877086039772 |0.586509635974304|
|Naive Bayes + CountVectorizer |0.6217504889434539 |0.0.5877087794432548|
|Logistic Regression           |0.6037773558509044 |0.5842398286937901|
All results were very close to baseline, which shows, that the signal was in the word order and connection between them. Bag of Words in this case shows only noise and the weight of specific and uniq words is low, unlike some highly specilized, niche subreddits for example,Toyota or Marvel. A wide variance of models, from the other side, allows as to assert, that the probability of incorrect modeling is minimized.
Third stage:
My next step was to check the hypothesis, that resulting scores are so low, because questions can be written by both man and woman and unfortunately, Reddit doesn't allow to check users sex. So I came up with an idea, to check the comments in those subreddits to compare.  r/AskMan should have more men's answer and r/AskWoman more women's. For tha that purpose I've downloaded ~same amount of comments as for the first two stages (~ 2000 for each /r). In the resulting dataset, baseline score was <dev>0.544444<dev>. Here are the results of the modeling:

|Model|Train Score|Test Score|
|:---:|:---:|:---:|
|Stacking with CountVectorizer  |0.9828167115902965 |0.6363636363636364|
|AdaBoost + TFIF                |0.7240566037735849 |0.6050505050505051|
|Random Forest +CountVectorizer |0.851078167115903  |0.6050505050505051|
|Bagging+TFIDF                  |0.9740566037735849 |0.5898989898989899|
|DecisionTree + TFIDF           |0.6765498652291105 |0.5797979797979798|
|Naive Bayes + CountVectorizer  |0.7934636118598383 |0.6353535353535353|
|Logistic Regression            |0.8733153638814016 |0.6191919191919192|

Forth attempt with wider range of ngram:
Unfoturnately, results were close to the previcios approaches or even closer to baseline. 
My fourth approach I came up to would be try to catch the sygnal "between the words", instead words by themself. So, I'll try to make range of ngram wider up two 4. Baseline Accuracy is <dev>0.5<dev>

|Model|Train Score|Test Score|
|:---:|:---:|:---:|
|Stacking with CountVectorizer  |0.9828167115902965| 0.6363636363636364|
|AdaBoost + TFIF                |0.7064547518650665| 0.6108949416342413|
|Random Forest +CountVectorizer |0.9948102497567305| 0.6138132295719845|
|Bagging+TFIDF                  |0.9967564060979566| 0.6284046692607004|
|DecisionTree + TFIDF           |0.8799870256243918| 0.6050583657587548|
|Naive Bayes + CountVectorizer  |0.7833279273434965| 0.5933852140077821|
|Logistic Regression            |0.9292896529354525| 0.6206225680933852|
Didn't help.





1. neighborhood
2. exter_qual
3. bsmt_qual
4. kitchen_qual
5. 1st_flr_sf
6. year_built
7. total_bsmt_sf
8. garage_cars
9. garage_area
10. gr_liv_area
11. overall_qual

Each of those was examined for the distribution skewness and outliers. Needed corrections have been made.
After that, I've build a Linear regression model to predict the prices with wide usage of transformation technics.
To avoid overfitting, I've used Lasso and Ridge, that noticeably improved results.   


Ridge model showed a slightly better result, than Lasso at MAE and R2 scores. 
|Model|MAE|R2_Train|R2_Test|
|:---:|:---:|:---:|:---:|
|Lasso|16997.90|0.913|0.899|
|Ridge|16578.27|0.929|0.902|




### Description

In week four we've learned about a few different classifiers. In week five we learned about webscraping, APIs, and Natural Language Processing (NLP). This project will put those skills to the test.

For project 3, your goal is two-fold:
1. Using [Pushshift's](https://github.com/pushshift/api) API, you'll collect posts from two subreddits of your choosing.
2. You'll then use NLP to train a classifier on which subreddit a given post came from. This is a binary classification problem.


#### About the API

Pushshift's API is fairly straightforward. For example, if I want the posts from [`/r/boardgames`](https://www.reddit.com/r/boardgames), all I have to do is use the following url: https://api.pushshift.io/reddit/search/submission?subreddit=boardgames

To help you get started, we have a primer video on how to use the API: https://youtu.be/AcrjEWsMi_E

**NOTE:** Pushshift now limits you to 100 posts per request (no longer the 500 in the screencast).

---

### Requirements

- Gather and prepare your data using the `requests` library.
- **Create and compare two models**. Any two classifiers at least of your choosing: random forest, logistic regression, KNN, etc.
- A Jupyter Notebook with your analysis for a peer audience of data scientists.
- An executive summary of your results.
- A short presentation outlining your process and findings for a semi-technical audience.

**Pro Tip:** You can find a good example executive summary [here](https://www.proposify.biz/blog/executive-summary).

---

### Necessary Deliverables / Submission

- Code must be in at least one clearly commented Jupyter Notebook.
- A readme/executive summary in markdown.
- You must submit your slide deck as a PDF.
- Materials must be submitted by **9:30 AM (PST) on Friday, Oct. 7th**.

---

## Rubric
Your instructors will evaluate your project (for the most part) using the following criteria.  You should make sure that you consider and/or follow most if not all of the considerations/recommendations outlined below **while** working through your project.

For Project 3 the evaluation categories are as follows:<br>
**The Data Science Process**
- Problem Statement
- Data Collection
- Data Cleaning & EDA
- Preprocessing & Modeling
- Evaluation and Conceptual Understanding
- Conclusion and Recommendations

**Organization and Professionalism**
- Organization
- Visualizations
- Python Syntax and Control Flow
- Presentation

**Scores will be out of 30 points based on the 10 categories in the rubric.** <br>
*3 points per section*<br>

| Score | Interpretation |
| --- | --- |
| **0** | *Project fails to meet the minimum requirements for this item.* |
| **1** | *Project meets the minimum requirements for this item, but falls significantly short of portfolio-ready expectations.* |
| **2** | *Project exceeds the minimum requirements for this item, but falls short of portfolio-ready expectations.* |
| **3** | *Project meets or exceeds portfolio-ready expectations; demonstrates a thorough understanding of every outlined consideration.* |


### The Data Science Process

**Problem Statement**
- Is it clear what the goal of the project is?
- What type of model will be developed?
- How will success be evaluated?
- Is the scope of the project appropriate?
- Is it clear who cares about this or why this is important to investigate?
- Does the student consider the audience and the primary and secondary stakeholders?

**Data Collection**
- Was enough data gathered to generate a significant result?
- Was data collected that was useful and relevant to the project?
- Was data collection and storage optimized through custom functions, pipelines, and/or automation?
- Was thought given to the server receiving the requests such as considering number of requests per second?

**Data Cleaning and EDA**
- Are missing values imputed/handled appropriately?
- Are distributions examined and described?
- Are outliers identified and addressed?
- Are appropriate summary statistics provided?
- Are steps taken during data cleaning and EDA framed appropriately?
- Does the student address whether or not they are likely to be able to answer their problem statement with the provided data given what they've discovered during EDA?

**Preprocessing and Modeling**
- Is text data successfully converted to a matrix representation?
- Are methods such as stop words, stemming, and lemmatization explored?
- Does the student properly split and/or sample the data for validation/training purposes?
- Does the student test and evaluate a variety of models to identify a production algorithm (**AT MINIMUM:** two models)?
- Does the student defend their choice of production model relevant to the data at hand and the problem?
- Does the student explain how the model works and evaluate its performance successes/downfalls?

**Evaluation and Conceptual Understanding**
- Does the student accurately identify and explain the baseline score?
- Does the student select and use metrics relevant to the problem objective?
- Does the student interpret the results of their model for purposes of inference?
- Is domain knowledge demonstrated when interpreting results?
- Does the student provide appropriate interpretation with regards to descriptive and inferential statistics?

**Conclusion and Recommendations**
- Does the student provide appropriate context to connect individual steps back to the overall project?
- Is it clear how the final recommendations were reached?
- Are the conclusions/recommendations clearly stated?
- Does the conclusion answer the original problem statement?
- Does the student address how findings of this research can be applied for the benefit of stakeholders?
- Are future steps to move the project forward identified?


### Organization and Professionalism

**Project Organization**
- Are modules imported correctly (using appropriate aliases)?
- Are data imported/saved using relative paths?
- Does the README provide a good executive summary of the project?
- Is markdown formatting used appropriately to structure notebooks?
- Are there an appropriate amount of comments to support the code?
- Are files & directories organized correctly?
- Are there unnecessary files included?
- Do files and directories have well-structured, appropriate, consistent names?

**Visualizations**
- Are sufficient visualizations provided?
- Do plots accurately demonstrate valid relationships?
- Are plots labeled properly?
- Are plots interpreted appropriately?
- Are plots formatted and scaled appropriately for inclusion in a notebook-based technical report?

**Python Syntax and Control Flow**
- Is care taken to write human readable code?
- Is the code syntactically correct (no runtime errors)?
- Does the code generate desired results (logically correct)?
- Does the code follows general best practices and style guidelines?
- Are Pandas functions used appropriately?
- Are `sklearn` and `NLTK` methods used appropriately?

**Presentation**
- Is the problem statement clearly presented?
- Does a strong narrative run through the presentation building toward a final conclusion?
- Are the conclusions/recommendations clearly stated?
- Is the level of technicality appropriate for the intended audience?
- Is the student substantially over or under time?
- Does the student appropriately pace their presentation?
- Does the student deliver their message with clarity and volume?
- Are appropriate visualizations generated for the intended audience?
- Are visualizations necessary and useful for supporting conclusions/explaining findings?


---

### Why did we choose this project for you?
This project covers three of the biggest concepts we cover in the class: Classification Modeling, Natural Language Processing and Data Wrangling/Acquisition.

Part 1 of the project focuses on **Data wrangling/gathering/acquisition**. This is a very important skill as not all the data you will need will be in clean CSVs or a single table in SQL.  There is a good chance that wherever you land you will have to gather some data from some unstructured/semi-structured sources; when possible, requesting information from an API, but sometimes scraping it because they don't have an API (or it's terribly documented).

Part 2 of the project focuses on **Natural Language Processing** and converting standard text data (like Titles and Comments) into a format that allows us to analyze it and use it in modeling.

Part 3 of the project focuses on **Classification Modeling**.  Given that project 2 was a regression focused problem, we needed to give you a classification focused problem to practice the various models, means of assessment and preprocessing associated with classification.   
