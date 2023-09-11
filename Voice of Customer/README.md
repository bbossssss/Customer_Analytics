# Voice of Customer Analytics - Google Colab Project

This Google Colab notebook contains a Voice of Customer (VoC) analytics project aimed at analyzing and extracting insights from customer reviews and feedback for the "Copper Beyond Buffet" restaurant on Wongnai.com.

## Overview

The VoC analytics project utilizes natural language processing (NLP) techniques to gain a deeper understanding of customer sentiments and opinions based on the reviews available on Wongnai.com. The notebook includes steps for data loading, text preprocessing, topic modeling, and visualization.

## Project Notebook

- **Project Notebook**: [voice of customer.ipynb](voice%20of%20customer.ipynb)

## Data

The data used in this project consists of customer reviews for "Copper Beyond Buffet" from Wongnai.com. You can find the data source [here](https://www.wongnai.com/restaurants/copper?_st=cD01O2I9MjI2NjgwO2FkPWZhbHNlO3Q9MTY5NDQ0MjIyNTExNTtyaT0xWDdhcERITHExSVNKcEpmckowRVVrcmFZTFozb2Y7aT0xWDcwSVgyd0ExUEdYUm5iU3g4RGlBVFhYaExNc1I7d3JlZj1zcjs%3D).

## About the Code

The Python script in this project (`voice of customer.ipynb`) performs the following key tasks:

1. **Data Loading and Preprocessing**: The script loads customer review data from a CSV file and preprocesses it. This includes text tokenization and the removal of stopwords.
   ```python
   # Load the customer review dataset
   df = pd.read_csv('customer_reviews.csv')

   # Define a list of stopwords and words to remove
   stopwords = list(pythainlp.corpus.thai_stopwords())
   removed_words = [' ', '  ', '\n', 'ร้าน', '(',')']
   screening_words = stopwords + removed_words

   # Function to tokenize words and remove stopwords
   def tokenize_with_space(sentence):
       merged = ''
       words = pythainlp.word_tokenize(str(sentence), engine='newmm')
       for word in words:
           if word not in screening_words:
               merged = merged + ',' + word
       return merged[1:]

   # Apply tokenization to the 'Review' column and create a new 'Review_tokenized' column
   df['Review_tokenized'] = df['Review'].apply(lambda x: tokenize_with_space(x))

2. **Topic Modeling with LDA**: The code uses Latent Dirichlet Allocation (LDA) to perform topic modeling on the customer reviews. LDA is employed to discover hidden topics within the text data.
   ```python
   # Topic modeling with LDA
   num_topics = 10
   lda_model = gensim.models.LdaModel(corpus=gensim_corpus, id2word=id2word, 
                                   chunksize=chunksize, alpha='auto', eta='auto', 
                                   iterations=iterations, num_topics=num_topics, 
                                   passes=passes, eval_every=eval_every)


3. **Visualization of Results**: The script generates visualizations of topic distributions using pyLDAvis, making it easier to interpret and understand the identified topics.
   ```python
   # Visualizing results
   pyLDAvis.gensim.prepare(lda_model, gensim_corpus, dictionary)


4. **Prediction of Topics**: The code assigns topics and scores to each customer review, allowing for an in-depth analysis of customer sentiments and opinions.
   ```python
   # Predicting topics and scores for each review
   df['topic'] = df['Review_tokenized'].apply(lambda x: lda_model.get_document_topics(dictionary.doc2bow(x.split(',')))[0][0])
   df['score'] = df['Review_tokenized'].apply(lambda x: lda_model.get_document_topics(dictionary.doc2bow(x.split(',')))[0][1])


The Jupyter Notebook (`voice of customer.ipynb`) provides detailed explanations and comments throughout the code, making it accessible and informative for users who want to understand the analysis process.

Feel free to explore the code to gain insights into how customer feedback is processed and analyzed in this project.

## How to Run

1. Open the [Google Colab notebook](https://github.com/bbossssss/MADT8101_Customer_Analytics/blob/c7eaa31bd454cc455ef8ac11c2a5c4236484823e/Voice%20of%20Customer/Voice_of_customer.ipynb) in your browser.

2. Click on "Open in Colab" to run the notebook in your Google Colab environment.

3. Follow the steps outlined in the notebook to analyze and visualize customer sentiments and opinions.

## Results

The project results include visualizations of topic distributions, identification of dominant topics in customer reviews, and the assignment of topics and scores to each customer review.

![Visualizations](https://github.com/bbossssss/MADT8101_Customer_Analytics/blob/6e624bbddb9119de59484f4089200d3a68035789/Voice%20of%20Customer/image/Visualizing_results.png)

## License

This project is open-source and is available under the [MIT License](LICENSE). You are free to use and modify the code for educational and non-commercial purposes.

## Contact

If you have any questions or suggestions regarding this project, feel free to contact Satorn at Satornboss@gmail.com.

Happy analyzing the Voice of Customer data in Google Colab!
