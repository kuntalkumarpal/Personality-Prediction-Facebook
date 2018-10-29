from nltk.tokenize import RegexpTokenizer
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
import gensim
import csv
import sys
import cPickle
import time
import progressbar as pb

csv.field_size_limit(sys.maxsize)

tokenizer = RegexpTokenizer(r'\w+')

symbols = ['@','!','#','$','%','^','&','*','-','=','+','.',',','<','>',':',';','?','_']
#initialize widgets
widgets = ['Time for loop of 48701 iterations: ', pb.Percentage(), ' ',  
            pb.Bar(marker=pb.RotatingMarker()), ' ', pb.ETA()]
#initialize timer
timer = pb.ProgressBar(widgets=widgets, maxval=48710).start()

# create English stop words list
#en_stop = get_stop_words('en')

def createTopics(fname):
    
    ifile = open(fname, "rb")
    reader = csv.reader(ifile)

    f = open('final_id_topics.csv', 'w')
    writer = csv.writer(f)
    writer.writerow(("userid", "topics"))
    #next(reader)
    next(reader)
    line = 2
    #stores per user len of dict
    userDict = [] #to store uid with len of dictionary
    blank = [] #to store uid with no stemmed token
    
    for row in reader:
        print "Line : ",line
        line +=1
        doc_a = row[2]  

        doc_a=''.join([i if ord(i) < 128 else ' ' for i in doc_a])    

        #Filtering out the symbols. Issue where they are attached to words and the actual word is not being taken
        doc_a = ''.join([ i.replace("\r","") if i not in symbols else ' ' for i in doc_a])
       
        #compile sample documents into a list
        doc_set = [doc_a]
        # list for tokenized documents in loop
        texts = []

        
        # loop through document list
        for i in doc_set:
            stemmed_tokens = [word for word in gensim.utils.tokenize(i, lower=True) if word not in STOPWORDS and len(word) > 3]
            if stemmed_tokens:
                #print stemmed_tokens
                None
            else:
                noToken = 1
                print "NO STEMMED TOKEN"
                stemmed_tokens.append("00BLANK00")
                blank.append(row[1])

            texts.append(stemmed_tokens)

        #Finds the number of unique tokens in text supplied per user
        dictionary = corpora.Dictionary(texts)

        #Finds the occurance of each of the unique words
        corpus = [dictionary.doc2bow(text) for text in texts]

        #userid,length of dictionary
        userDict.append((row[1],len(dictionary)))
   
        # generate LDA model
        ldamodel = gensim.models.ldamodel.LdaModel(corpus, alpha=0.5, id2word = dictionary, passes=20)
        string = ""
        topics = ldamodel.print_topics()
        
        for topic in topics:
            ind_topic = topic[1]
            for x in ind_topic.split("+"):
                string = string + (x.split("*"))[1] + " "
                #print string		

        #print string		
        writer.writerow((row[1], string))
        timer.update(line)

        #raw_input("WAIT3:")


    cPickle.dump(userDict,open("DictStore.pkl","wb"))
    cPickle.dump(blank,open("blank.pkl","wb"))
    
    ifile.close()
    f.close()
    timer.finish()

#def evaluate


if __name__ == "__main__":

    start_time = time.time()
    fname = "final_merged_status.csv"
    createTopics(fname)
    end_time = time.time()
    print("--- %s seconds ---" % (time.time() - start_time))
