#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pymorphy2

import pstats

from sklearn.model_selection import cross_val_score,ShuffleSplit
from sklearn.base import BaseEstimator, ClassifierMixin
import sys
import time

class Trainer(BaseEstimator,ClassifierMixin):
    """
    Trainer ver.2.0.0
    Create voc, first key - word1.
    voc[first_key] = voc2
    voc2 keys - word2_chunk
    voc2[word2_chunk] = voc3
    voc3 consists from word2 and value
    """
    def fit(self,data,target):
        self.phrase_stat_dict = {}
        self.word1_stat_dict = {}
        self.word2_stat_dict = {}
        linecounter = 0
        for line in target:
            linecounter = linecounter + 1
            if linecounter%100==0:
                sys.stdout.write("\rfit::1::"+str(linecounter))
                sys.stdout.flush()
            word1, word2 = line.split(' ')

            #fill all keys for word2 and word1
            for keyn in xrange(0,len(word2)+1):
                key = word2[:keyn]
                if key not in self.word2_stat_dict:
                    self.word2_stat_dict[key] = {}
                if word2 not in self.word2_stat_dict[key]:
                    self.word2_stat_dict[key][word2] = 0
                self.word2_stat_dict[key][word2] += 1

            for keyn in xrange(0,len(word1)+1):
                key = word1[:keyn]
                if key not in self.word1_stat_dict:
                    self.word1_stat_dict[key] = {}
                if word1 not in self.word1_stat_dict[key]:
                    self.word1_stat_dict[key][word1] = 0
                self.word1_stat_dict[key][word1] += 1

            """
            #fill word1 and word2 vocs
            if word2 not in self.word2_stat_dict:
                self.word2_stat_dict[word2] = 0
            self.word2_stat_dict[word2] += 1

            if word1 not in self.word1_stat_dict:
                self.word1_stat_dict[word1] = 0
            self.word1_stat_dict[word1] += 1 
            """

            if word1 not in self.phrase_stat_dict:
                self.phrase_stat_dict[word1] = {}
            voc = self.phrase_stat_dict[word1]
            for keyn in xrange(2,len(word2)+1):
                if keyn not in voc:
                    voc[keyn] = {}

                vockeyn = voc[keyn]

                key = word2[:keyn]
                if key not in vockeyn:
                    vockeyn[key] = {}
                if word2 not in vockeyn[key]:
                    vockeyn[key][word2] = 0
                vockeyn[key][word2] += 1

        self.most_freq_dict = {}

        linecounter = 0;
        for word1 in self.phrase_stat_dict:
            linecounter = linecounter + 1
            if linecounter%100==0:
                sys.stdout.write("\rfit::2::"+str(linecounter))
                sys.stdout.flush()
            self.most_freq_dict[word1] = {}
            voc2 = self.most_freq_dict[word1]
            voc = self.phrase_stat_dict[word1]
            for keyn in voc:
                voc2[keyn] = {}
                for key in voc[keyn]:
                    #find the word in voc[key] vocabulary with highest freq
                    #get function of dict returns the value for given key
                    voc2[keyn][key] = max(voc[keyn][key], key=voc[keyn][key].get)

        #add data from train

        trfile = open("../data/test.csv")
        trfile.readline()
        for line in trfile:
            line = unicode(line,encoding="utf8")
            Id, data_el = line.strip().split(',')
            word1,word2_part = data_el.split()
            for keyn in xrange(0,len(word1)):
                key = word1[:keyn]
                if key not in self.word1_stat_dict:
                    self.word1_stat_dict[key] = {}
                if word1 not in self.word1_stat_dict[key]:
                    self.word1_stat_dict[key][word1] = 0
                self.word1_stat_dict[key][word1] += 1

        trfile.close()

        self.most_freq_dict_word1 = {}
        self.most_freq_dict_word2 = {}
        for key in self.word1_stat_dict:
            voc = self.word1_stat_dict[key]
            self.most_freq_dict_word1[key] = max(voc,key=voc.get)
        for key in self.word2_stat_dict:
            voc = self.word2_stat_dict[key]
            self.most_freq_dict_word2[key] = max(voc,key=voc.get)

        return self

    def findmaxword(self,word2_chunk):
        #there was not phrase word1 pw2...
        #find among all word2
        
        if word2_chunk in self.most_freq_dict_word2:
            return self.most_freq_dict_word2[word2_chunk]
        if word2_chunk in self.most_freq_dict_word1:
            return self.most_freq_dict_word1[word2_chunk]
        
        """
        max_word2 = ''
        max_word2_value = -1
        for key in self.word2_stat_dict:
            if key[:len(word2_chunk)]==word2_chunk:
                if max_word2_value<self.word2_stat_dict[key]:
                    max_word2_value = self.word2_stat_dict[key]
                    max_word2 = key
        if max_word2_value == -1:
            #find among word1
            max_word1 = ''
            max_word1_value = -1
            for key in self.word1_stat_dict:
                if key[:len(word2_chunk)]==word2_chunk:
                    if max_word1_value<self.word1_stat_dict[key]:
                        max_word1_value = self.word1_stat_dict[key]
                        max_word1 = key
            if max_word1_value == -1:
                return u""
            else:
                return max_word1
        else:
            return max_word2
        """
        return u""

    def predict(self,data):
        newtarget = []

        self.word1_word2 = 0
        self.word1_maxword = 0
        self.word1_nomaxword = 0
        self.noword1_maxword = 0
        self.noword1_nomaxword = 0

        linecounter = 0
        for line in data:
            linecounter += 1
            if linecounter%200==0:
                sys.stdout.write("\rpredict::"+str(linecounter))
                sys.stdout.flush()
            line = line[0]
            word1, word2_chunk = line.split(' ')
            if word1 in self.most_freq_dict:
                voc2 = self.most_freq_dict[word1]
                findkey = False
                #find the biggest key we have
                if len(word2_chunk) in voc2:
                    if word2_chunk in voc2[len(word2_chunk)]:
                        newtarget.append(word1+u" "+voc2[len(word2_chunk)][word2_chunk])
                        findkey = True
                        self.word1_word2 += 1
                
                if not findkey:
                    max_word = self.findmaxword(word2_chunk)
                    if max_word==u"":
                        newtarget.append(u"что она")
                        self.word1_nomaxword += 1
                    else:
                        newtarget.append(word1+u" "+max_word)
                        self.word1_maxword += 1
            else:
                #there are no word1 on voc
                max_word = self.findmaxword(word2_chunk)
                if max_word==u"":
                    newtarget.append(u"что она")
                    self.noword1_nomaxword += 1
                else:
                    newtarget.append(word1+u" "+max_word)
                    self.noword1_maxword += 1

        print ""
        print 'word1 word2 ',self.word1_word2
        print 'word1 maxword ',self.word1_maxword
        print 'word1 nomaxword ',self.word1_nomaxword
        print 'noword1 maxword ',self.noword1_maxword
        print 'noword1 nomaxword ',self.noword1_nomaxword
        print ""

        return newtarget



# объявим где хранятся исходные данные
PATH_TRAIN = '../data/train.csv'

PATH_TEST = '../data/test.csv'

# объявим куда сохраним результат
PATH_PRED = '../data/mypred.csv'

## Из тренировочного набора собираем статистику о встречаемости слов

# создаем словарь для хранения статистики
word_stat_dict = {}

starttime = time.time()

# открываем файл на чтение в режиме текста
fl = open(PATH_TRAIN, 'rt')

# считываем первую строчку - заголовок (она нам не нужна)
fl.readline()

traindata = []
traintarget = []

testdata = []
testtarget = []

morph = pymorphy2.MorphAnalyzer()

linecounter = 0
#read all data
for line in fl:
    linecounter += 1
    if linecounter%200 == 0:
        sys.stdout.write("\rread file:"+str(linecounter))

    line = unicode(line,encoding="utf8")
    Id, data_el, target_el = line.strip().split(',')

    """
    change word form
    """
    #word1,word2 = data_el.split()
    #word1 = morph.parse(word1)[0].normal_form
    #data_el = word1+u" "+word2

    #word1,word2 = target_el.split()
    #word1 = morph.parse(word1)[0].normal_form
    #word2 = morph.parse(word2)[0].normal_form
    #target_el = word1+u" "+word2

    traindata.append([data_el])
    traintarget.append(target_el)

fl.close()

## Выполняем предсказание

# открываем файл на чтение в режиме текста
fl = open(PATH_TEST, 'rt')

# считываем первую строчку - заголовок (она нам не нужна)
fl.readline()

# открываем файл на запись в режиме текста
out_fl = open(PATH_PRED, 'wt')

# записываем заголовок таблицы
out_fl.write('Id,Prediction\n')

#read train data
for line in fl:
    line = unicode(line,encoding="utf8")
    Id, data_el = line.strip().split(',')
    testdata.append([data_el])


fl.seek(0)
fl.readline()

trainer = Trainer()

cv = ShuffleSplit(n_splits=4,test_size=0.5,random_state=0)
scores = cross_val_score(trainer,traindata,traintarget,cv=cv)
av_score = reduce(lambda x,y:x+y,scores)/len(scores)
print 'average score: ',av_score
print 'scores: ',scores

trainer.fit(traindata,traintarget)

score = trainer.score(traindata,traintarget)
print 'train score: ',score

cal_train = False
if cal_train:
    testtarget = trainer.predict(traindata)
else:
    testtarget = trainer.predict(testdata)

counter = 0

if cal_train:
    #write train target
    #for line in fl:
    for ind in xrange(0,len(testtarget)):
        #line = unicode(line,encoding="utf8")
        #Id, data_el = line.strip().split(',')
        #out_fl.write("%s,%s\n" % (Id.encode("utf-8"),testtarget[counter].encode("utf-8")))
    
        if testtarget[ind]!=traintarget[ind]:
            out_fl.write("%s,%s,%s,%s\n" % (ind,traindata[counter][0].encode("utf-8"),\
                    testtarget[counter].encode("utf-8"),\
                    traintarget[counter].encode("utf-8")))
        counter = counter + 1
else:
    #write train target
    for line in fl:
        line = unicode(line,encoding="utf8")
        Id, data_el = line.strip().split(',')
        out_fl.write("%s,%s\n" % (Id.encode("utf-8"),testtarget[counter].encode("utf-8")))
    
        counter = counter + 1

fl.close()
out_fl.close()

endtime = time.time()
eltime = int(endtime-starttime)

print ""
print "TIME: "+str(eltime/60/60)+":"+str(eltime/60%60)+":"+str(eltime%60)
