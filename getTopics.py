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

'''# Create p_stemmer of class PorterStemmer
p_stemmer = PorterStemmer()
'''


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
        '''print "******************************************************"
        print "******************************************************"
        print "******************************************************"
        print "******************************************************"
        print "LINE NUMBER DONE:"'''
        #print row,"\n\n"
        print "Line : ",line
        #print "ROW STRING : ",row
        #raw_input("WAIT:")	
        line +=1
        #doc_a : Status row[1], row[0] : id
        doc_a = row[2]  
        #print doc_a,"\n\n"

        doc_a=''.join([i if ord(i) < 128 else ' ' for i in doc_a])    
        #print doc_a,"\n\n"

        #print doc_a.replace("\r","")
        #raw_input("WAIT1:")

        #Filtering out the symbols. Issue where they are attached to words and the actual word is not being taken
        doc_a = ''.join([ i.replace("\r","") if i not in symbols else ' ' for i in doc_a])
        #print doc_a,"\n\n"

        #compile sample documents into a list
        #doc_set = [doc_a, doc_b, doc_c, doc_d, doc_e]
        doc_set = [doc_a]
        #print "\n\n:\n",doc_set
        # list for tokenized documents in loop
        texts = []

        
        # loop through document list
        for i in doc_set:
            #print "\n\n...\n",i
            #raw_input("WAIT1:")
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

        #raw_input("WAIT2:")

        #print("check TEXTS")
        #print texts 

        #Finds the number of unique tokens in text supplied per user
        dictionary = corpora.Dictionary(texts)

        #Finds the occurance of each of the unique words
        corpus = [dictionary.doc2bow(text) for text in texts]

        #print dictionary
        #print "Number of unique words :",len(dictionary)
        #userid,length of dictionary
        userDict.append((row[1],len(dictionary)))
        #print corpus
        #raw_input("WAIT:")
        #print dictionary
        #print (row[1],len(dictionary))
        #raw_input("WAIT2:")

        # generate LDA model
        ldamodel = gensim.models.ldamodel.LdaModel(corpus, alpha=0.5, id2word = dictionary, passes=20)
        #print(doc_a)
        #print(ldamodel.show_topics(num_topics=2, num_words=4, formatted = True))
        string = ""
        topics = ldamodel.print_topics()
        
        #print topics, "\n\n"
        #raw_input("WAIT:")
        #print ldamodel.show_topics(),"\n\n"
        
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
    #evaluate()
