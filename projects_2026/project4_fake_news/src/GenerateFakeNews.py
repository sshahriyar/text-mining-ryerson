#Alexander Eliseev, 501093338
#DS8008 Final Project
#This file uses the LLAMA 3 (Aqquired it from hugging face) to generate more fake news articles similar to the real collected fake news articles
#These generated fake news articles are used to augment the fake news portion of the training dataset so that there is a
#closer amount of them to the number of real news articles so that the classification model for real or fake news is more accurate.
#LDA is ran on the entire training set of real fake news articles, to extract topics
#Next spaCy is used to identify names and people, it was an available and very reliable model that can do that
#Finally the topics + people are randomly selected in combinations and given to LLAMA 3 as prompt 
#to have it generate a news article about them. Obviously, its fake. Lab 11 was used as the inspiration for how to implement the LDA.
#All fake news articles are saved to the Data/FakeNewsArticlesArtificiallyGeneratedTrainingSet folder.
#Imports
import pandas as pd
import nltk
import os
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import spacy #NER model that can very effectively identify names
from gensim.models import LdaModel
from gensim import corpora
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import random

#First loading the real fake news articles and setting output destination where the generated fake articles will be saved to
InputFolder = "DS8008FinalProject/Data/train/FakeNewsArticlesTrainingSet/"
OutputFolder = "DS8008FinalProject/Data/train/FakeNewsArticlesArtificiallyGeneratedTrainingSet/"

#Load files saves the article text or "data" of the file as bytes when loading, so now converting each article from bytes back to UTF 8
#Also saving each article into a new array
FakeNewsArticles = []
for File in os.listdir(InputFolder):
    FilePath = os.path.join(InputFolder,File)

    #Opening article from the training folder of fake news articles
    with open(FilePath,"r",encoding="utf-8",errors="ignore") as File:
        Text = File.read()
        Text = Text.replace("<br />"," ").replace("<br>"," ")
        FakeNewsArticles.append(Text)

#print(FakeNewsArticles[0])

#Before proceeding unto LDA and topic extraction from the fake news articles, using SPACY NER in order to extract all names
#and put them all together into a dictionairy
nlp = spacy.load("en_core_web_sm")
NamedPeople = set()
counter = 0
for Article in FakeNewsArticles:
    counter += 1
    print(counter)
    doc = nlp(Article) #Converting to doc that spacy can parse
    for Entity in doc.ents:
        if Entity.label_ == "PERSON":
            NamedPeople.add(Entity.text)
#Testing, end result was actually quite good, majority of words in the 
#print(NamedPeople)

#Named people has the people identified and saved as their First Name + Last Name becuase of this, i want to now take the set
#and split their first/last names apart and lower case them to effectively be able to filter them out from the documents for LDA
NamedPeopleLowerCaseAndSplit = set()
#Some words that double as common names may be nice topics and should not be removed also, some noise does get caught by the NER
CollateralWordsToKeep = {"love","subscribe","angel","imperfect","dream","tangerine","give","images","ray","dick","hell","declares","raoch","cannon"}
for Person in NamedPeople:
    for Word in Person.split():
        if Word in CollateralWordsToKeep: #Keep the word
            continue
        NamedPeopleLowerCaseAndSplit.add(Word.lower())


#Now cleaning up the array of articles for LDA topic identification and tokenizing each one
FakeNewsArticlesTokenized = []
Lemmatizer = WordNetLemmatizer() #Lemmatizer to lemmatize the words to restrict noise and improve topic identification
stopWords=nltk.corpus.stopwords.words('english') #Creating set of stopwords that should be removed and not be used as tokens
stopWords+=["''", "'s", "...", "``","--","*","-","/","?",".","..","<",">","<>","|","!","@","#","$","%","^","&","(",")","_","+","=",";",":"]
stopWords+=["like", "would", "year", "first", "two","said", "say", "also", "could", "one","n't", "still", "even", "know"]
stopWords = set(stopWords) #Converting to set for faster speed
for i in range(len(FakeNewsArticles)):
    Article = FakeNewsArticles[i]
    Article = Article.lower() #Lower casing the article
    #Now going through every work in the article, removing any names identified earlier, useless punctuation and lemmatizing each token/word
    TokenWords = nltk.word_tokenize(Article)
    GoodTokenWords = []

    for Word in TokenWords:
        #First removing any token that is stopword and is essentially garbage
        if Word in stopWords: #Removing obvious noise tokens like punctuation and so on, these are useless
            continue
        elif Word in NamedPeopleLowerCaseAndSplit: #Removing named people, some actual words will be lose here but not many
            continue
        elif len(Word) <= 2: #Words with a length of 2 are generally useless and unimportant unless they are maybe a name which ahve been saved already
            continue
        else:
            GoodTokenWords.append(Lemmatizer.lemmatize(Word))
    
    FakeNewsArticlesTokenized.append(GoodTokenWords)

#Now that the articles have been cleaned up and tokenized they can be finally fed into the LDA model so reccuring topics can be identified
#Basically copying code from lab 11 here
Dictionairy = corpora.Dictionary(FakeNewsArticlesTokenized)
#Converting into bag of words format, needed for LDA
Corpus = [Dictionairy.doc2bow(Article) for Article in FakeNewsArticlesTokenized]
Topics = 200 #Setting as baseline

#Training LDA model
LDAModel = LdaModel(
    corpus=Corpus,
    id2word=Dictionairy,
    num_topics=Topics,
    #Reccomended to make these automatic to scale better and not be restricted, can be technically tuned this makes it easier
    alpha="asymmetric",
    eta="auto",
    passes=10,
    random_state=0
)

DocumentTopics = []

#Extracting identified 50 topics
for Topic_ID, Topic_Words in LDAModel.show_topics(num_topics=Topics,formatted=False):
    Topics = [Topic for Topic, prob in Topic_Words]
    DocumentTopics.append(Topics)

#Displaying 50 identified topics
#print(DocumentTopics)

#Now, using the identified topics and the identified people in order to create prompts of topics + people based on which
#LLAMA 3 will generate fake articles.
#Loading llama, requires access on machine from token/from hugging face
ModelName = "meta-llama/Meta-Llama-3-8B-Instruct"
#Creating llama 3 tokenizer
Tokenizer = AutoTokenizer.from_pretrained(ModelName)
#Defining llama 3
Model = AutoModelForCausalLM.from_pretrained(
    ModelName,
    torch_dtype=torch.float16
).to("cuda")
#Generates however many random fake articles randomly selecting from list of indeitifed people and topics for the prompt.
#And saves the generated fake news article to the train/FakeNewsArticlesArtificiallyGeneratedTraining
for i in range(150,4000):
    #Randomly selecting 2-5 people from the set of people identified in the fake news using NER
    NumPeople = random.randint(2,5)
    SelectedPeople = random.sample(list(NamedPeople),NumPeople)
    #Randomly selecting 1/200 identified topics
    RandomTopic = random.choice(DocumentTopics)

    #Generating fake news!
    #Prompt is passed to LLAMA 3 telling it/guiding it how to write the fake news article.
    #I kept changing number of words 
    RandomPrompt = f"""
    You are a news article generator. 

    Involving the following people:
    {SelectedPeople}

    And make up a logical story involving the listed people and these topic words:
    {RandomTopic}

    Avoid repeats.

    Make it between 300-600 words.

    Requirement: Only write the article, don't write anything else in your output.
    """

    #Final step is to generate the text by inputting the prompt into LLAMA 3
    ModelInput = Tokenizer(RandomPrompt,return_tensors="pt").to("cuda")
    with torch.inference_mode():
        Output = Model.generate(
            **ModelInput,
            max_new_tokens=575, #Setting this to something like 1200 is possible for maximum quality but, it trained 42 overnight in like 10 hours just not feasible.
            temperature=0.8,
            top_p=0.9,
            do_sample=True,
            num_return_sequences=2 #So once the context prompt has been loaded to LLAMA, there is little difference time wise between generating
            #Just 1 article or many. However changing contexts takes the bulk of the time. To PAD the number generated articles I have 10 be made that are all similar.
            ) #This does mean that they depreciate in training quality.
    
    #Extracting the 10 similarily generated fake news articles by LLAMA 3 and spliting them up
    CompletelyFakeArticles = []
    for output in Output:
        text = Tokenizer.decode(output, skip_special_tokens=True)
        text = text[len(RandomPrompt):].strip()
        CompletelyFakeArticles.append(text)       
    
    #Saving each one to an indivudal text file.
    for j, article in enumerate(CompletelyFakeArticles):
        FilePath = os.path.join(
            OutputFolder,
            f"FakeNewsArticleGenerated_{i}_{j}.txt"
        )

        with open(FilePath, "w", encoding="utf-8") as f:
            f.write(article)



















