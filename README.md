# Resume Analyzer:

The goal of this project was to use a CNN developed for text classification (sentimental analysis) and adjust it to classify sentences as strong or weak (in a resume).

## Some traits of "strong" sentences:
  * Sentences usually began with an action verb
  * Sentences avoided using personal pronouns
  * Sentences were generally longer than weak sentences
  * Sentences usually were very descriptive, and thus had many nouns
  * Sentences sometimes included some quantitative amount
  * Sentences used active voice
  
## Some traits of "weak" sentences
  * Sentences used personal pronouns
  * Sentences were very basic or short
  * Sentences did not use quantitive descriptions
  * Sentences included interjection or exclamation
  * Sentences used passive voice
  
## Step 1: Transforming Sentences into POS
Most CNNs performing sentiment analysis maps words in a sentence to a dictionary during the embedding layer. For resume analysis, this would be difficult to perform since many strong and weak sentences use common words. Thus, I incorporated the Natural Language Toolkit (NLTK) which could map words to a particle of speech (POS). Thus the input to the CNN is the POS of the sentence. 

## Step 2: Convolutional Neural Network
The embedding layer maps the POS of the sentence to a dictionary of all the POS present in the training file. After some padding to transform the 2D array into a 4D tensor, the convolutional layer applies a filter matrix and max-pooling to transform the result into a modified 4D tensor. A dropout layer is used to randomly drop neurons to minimize overfitting. A softmax function transforms the 4D tensor into an array of probabilities. The CNN then adjusts weights and biases to minimize the cross-entropy loss. 

## Step 3: Training Data
Data acquisition was difficult: 
* There were very few collections of resumes on the web (We used Indeed and Monster) 
* Classifying a sentence/resume as strong or weak had to be performed manually. 

Ultimately, we ended up with around 250 total data points for training and testing. 

## Step 4: Results
Ultimately, we were satisifed with the results given the small dataset. The accuracy of training was approximately 84% with a loss of 0.50 after around 1000 training cycles. We tweaked around by adjusting embedding layer dimensionality, filter size, number of filters, and batch size to optimize the results. 

## Step 5: Improvements
Many improvements can be made to improve accuracy of this project. Increasing the training dataset would be most beneficial. A dataset of 1,000 or 10,000 sentences would dramatically improve accuracy. Also, this project does not distinguish the difference between "strong words (ex: "initiated", "analyzed", "implemented") and "weak" words (ex: "did","made","helped"). It also does not account for resume formatting or quality of work. 
