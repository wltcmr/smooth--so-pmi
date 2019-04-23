#!/usr/bin/env python3
# coding: utf-8
# File: so-pmi.py
# Author: lhy<lhy_in_blcu@126.com,https://huangyong.github.io>
# Date: 18-4-4
import jieba.posseg as pseg
import jieba
import re
import jieba.analyse
import math,time
jieba.load_userdict(r'D:\python\Lib\site-packages\jieba\dict2.txt')
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
train_path = 'E:/kaggle_and_paper_coding/SentimentWordExpansion-master/data/meituan.txt'
candipos_path = 'E:/kaggle_and_paper_coding/SentimentWordExpansion-master/data/candi_pos.txt'
candineg_path = 'E:/kaggle_and_paper_coding/SentimentWordExpansion-master/data/candi_neg.txt'
stop_words_path = r'E:\kaggle_and_paper_coding\SentimentWordExpansion-master\data\stop_words.txt'
sentiment_path = r'E:\kaggle_and_paper_coding\SentimentWordExpansion-master\data\sentiment_words.txt'
level_path = r'E:\kaggle_and_paper_coding\SentimentWordExpansion-master\data\levelDict.txt'
test_path = r'E:\kaggle_and_paper_coding\SentimentWordExpansion-master\data\test200.txt'
'''分词'''
def seg_corpus(train_path, sentiment_path,stop_words_path,level_path,test_path):
    #将情感词加入到用户词典当中，保证分词能够将种子情感词切开
    sentiment_words = [line.strip().split('\t')[0] for line in open(sentiment_path,encoding = 'utf_8')]
    levelwords = [line.strip().split(',')[0] for line in open(level_path,encoding = 'utf_8')]
    stop_words = [line.strip().split('\t')[0] for line in open(stop_words_path, encoding='utf_8')]
    test_words = [line.strip().split('\t')[0] for line in open(test_path, encoding='utf_8')]
    seg_data = list()
    count = 0
    for line in open(train_path,encoding = 'utf_8'):
        line = line.strip()
        re.sub(r"#[^%]*#", "", line)
        count += 1
        if line:
            segments = [word for word in jieba.cut(line,cut_all = True) if word not in stop_words ]
            seg_data.append(segments)
        #keywords = jieba.analyse.extract_tags(content, topK=5, withWeight=True, allowPOS=())
        #考虑关键词分解
        else:
            continue
    return seg_data
'''统计搭配次数'''
def collect_cowords(sentiment_path, seg_data):
    def check_words(sent):
        if set(sentiment_words).intersection(set(sent)):#如果去重的情感词集与每一条分好词去重的评论（看做一个文本）存在着交集，必然会存在一个共现
            return True
        else:
            return False
    cowords_list = list()
    window_size = 5
    count = 0
    sentiment_words = [line.strip().split('\t')[0] for line in open(sentiment_path,encoding = 'utf_8')]#将所有情感词存入到一个列表中
    for sent in seg_data:#遍历每一条评论
        count += 1
        if check_words(sent):#如果存在交集执行
            for index, word in enumerate(sent):#遍历分词后的单条评论
                if index < window_size:
                    left = sent[:index]#[0:1]
                else:
                    left = sent[index - window_size: index]
                if index + window_size > len(sent):
                    right = sent[index + 1:]
                else:
                    right = sent[index + 1: index + window_size + 1]#应该修改成[index+1:index+1+window_size]
                context = left + right + [word]#将原句拆分
                if check_words(context):
                    for index_pre in range(0, len(context)):
                        if check_words([context[index_pre]]):#判断是否是情感词根
                            for index_post in range(index_pre + 1, len(context)):
                                cowords_list.append(context[index_pre] + '@' + context[index_post])
                                #['i','love','you','very','much']如果‘i’是情感词那么i@love，l@you，i@very等等都作为候选共现组合
                                #同理在此前面的处理将该句切分成以下['love','you','very','much','i']、['i','you','very','much','love']……
                                #可识别出词根中出现的词与该文本的所有共现关系
    return cowords_list#二元共现词集，不考虑同一个文本中词出现的顺序
'''计算So-Pmi值'''
def collect_candiwords(seg_data, test_path,cowords_list, sentiment_path):
    '''good_turning平滑的SO—PMI算法'''
    def good_turing(word_dict):
        valuelist = []
        for k,v in word_dict.items():
            valuelist.append(v)
        valuelist = list(set(valuelist))
        turing={}
        for key in valuelist:
            if key not in turing:
                turing[key] = 0
        for k,v in word_dict.items():
            if v in turing:
                turing[v] += 1
        return turing
    '''互信息计算公式'''
    def compute_mi(p1, p2, p12):
        return math.log2(p12) - math.log2(p1) - math.log2(p2)#公式
    '''统计词频，这里的词频是出现该词的评论条数'''
    def collect_worddict(seg_data,test_path):
        test_words = [line.strip().split('\t')[0] for line in open(test_path, encoding='utf_8')]
        word_dict = dict()
        all = 0
        for each in test_words:
            if each not in word_dict:
                word_dict[each] = 1
        for line in seg_data:
            for word in line:
                if word not in word_dict:
                    word_dict[word] = 1#此处应该是平滑
                else:
                    word_dict[word] += 1
        all = sum(word_dict.values())
        return word_dict, all
    '''统计词共现次数'''
    def collect_cowordsdict(cowords_list):
        co_dict = dict()
        candi_words = list()
        for co_words in cowords_list:
            #candi_words.extend(co_words.split('@'))#列表末端添加
            if co_words not in co_dict:
                co_dict[co_words] = 1
            else:
                co_dict[co_words] += 1
        return co_dict #co_dict是共现词频的字典，candi_words是共现词分解后添加的列表此处candi_words应该存在很大的冗余，建议Set
    '''收集种子情感词'''
    def collect_sentiwords(sentiment_path, word_dict):
        pos_words = set([line.strip().split('\t')[0] for line in open(sentiment_path,encoding = 'utf_8') if#
                         line.strip().split('\t')[1] == 'pos']).intersection(set(word_dict.keys()))
        neg_words = set([line.strip().split('\t')[0] for line in open(sentiment_path,encoding ='utf_8') if
                         line.strip().split('\t')[1] == 'neg']).intersection(set(word_dict.keys()))
        return pos_words, neg_words
    '''计算sopmi值'''
    def compute_sopmi(test_path,pos_words, neg_words, word_dict, co_dict, all,n):
        test_words = [line.strip().split('\t')[0] for line in open(test_path, encoding='utf_8')]
        #candi_words共现词拆分后列表、pos_words褒义词表、neg_words贬义词表、word_dict单个词词频、co_dict是共现词频的字典、all总词数不去重sum(word_dict.values())、n为取前n个大的pmi值
        pmi_dict = dict()#储存pmi值
        for test_word in set(test_words):#如果只分析需要的新词则将candi_words替换成新词列表
            pos_pmi = []
            neg_pmi = []
            turing = good_turing(word_dict)
            for pos_word in pos_words:
                p1 = word_dict[pos_word] / all#p1概率，p1为基础词库中的词根
                p2 = word_dict[test_word] / all#p2概率
                pair = pos_word + '@' + test_word
                if pair not in co_dict:#'''没有考虑，颠倒情况例如，love@you 和you@love'''
                    print(test_word,'没有情感词共现')
                else:
                    count12 = co_dict[pair]
                    #p12 = count12 / all#共现概率…………………………………
                    #p12 = (count12 + 1) / (all + 2)  # Laplace平滑共现概率…………………………………

                    N = turing.get(count12,1)#get方法在于若存在key值则取key，若不存在则取则取默认值
                    p = count12 +1
                    N1 = turing.get(p,1)
                    p12 = (p*N1/N)/all

                    pos_pmi_each = compute_mi(p1,p2,p12)
                    pos_pmi.append(pos_pmi_each)
            #pos_pmi.sort(reverse = True)#从大到小排序
            #pos_sum = sum( pos_pmi[0:n] )#此处修改成概率必须大于某个阈值，或者设定一个参数n必须取相似度最大的前n个词
            for neg_word in neg_words:
                p1 = word_dict[neg_word] / all
                p2 = word_dict[test_word] / all
                pair = neg_word + '@' + test_word
                if pair not in co_dict:
                    print(test_word,'没有情感词共现')
                else:
                    count12 = co_dict[pair]
                    #p12 = count12 / all#不平滑共现概率…………………………………
                    #p12 = (count12 +1 )/ (all+2) # Laplace平滑共现概率…………………………………

                    N = turing.get(count12,1)  # get方法在于若存在key值则取key，若不存在则取则取默认值
                    p = count12 + 1
                    N1 = turing.get(p, 1)
                    p12 = (p * N1/N) / all

                    neg_pmi_each = float(-1*compute_mi(p1,p2,p12))
                    neg_pmi.append(neg_pmi_each)
            sum_list = pos_pmi + neg_pmi
            sum_list = sorted(sum_list,key = abs,reverse = True)#从大到小排序
            print(sum_list[0:n])
            so_pmi = sum(sum_list[0:n])#此处修改成概率必须大于某个阈值，或者设定一个参数n必须取相似度最大的前n个词
            #o_pmi = pos_sum - neg_sum
            pmi_dict[test_word] = so_pmi#so_pmi值褒贬的差值
        return pmi_dict
    word_dict, all = collect_worddict(seg_data,test_path)
    co_dict  = collect_cowordsdict(cowords_list)
    pos_words, neg_words = collect_sentiwords(sentiment_path, word_dict)
    pmi_dict = compute_sopmi(test_path, pos_words, neg_words, word_dict, co_dict, all,10)
    #返回一个字典，对应每个新词的pmi值
    return pmi_dict
'''保存结果'''
def save_candiwords(pmi_dict, candipos_path, candineg_path,test_path):
    def get_tag(word):
        if word:
            return [item.flag for item in pseg.cut(word)][0]
        else:
            return 'x'
    pos_dict = dict()
    neg_dict = dict()
    f_neg = open(candineg_path, 'w+',encoding='utf_8')
    f_pos = open(candipos_path, 'w+',encoding='utf_8')

    for word, word_score in pmi_dict.items():
        if word_score > 0:
            pos_dict[word] = word_score
        if word_score < 0:
            neg_dict[word] = abs(word_score)
    for word, pmi in sorted(pos_dict.items(), key=lambda asd:asd[1], reverse=True):
        f_pos.write(word + ',' + str(pmi) + ',' + 'pos'+get_tag(word) +'\n')
    for word, pmi in sorted(neg_dict.items(), key=lambda asd:asd[1], reverse=True):
        f_neg.write(word + ',' + str(pmi) + ',' + 'neg'+get_tag(word) +'\n')
    TP = 0; TN =0;FP = 0;FN = 0;
    test_label = [line.strip().split('\t')[1] for line in open(test_path, encoding='utf_8')]
    test_words = [line.strip().split('\t')[0] for line in open(test_path, encoding='utf_8')]
    for word,word_score in pmi_dict.items():
        index_w = test_words.index(word)
        if test_label[index_w] == 'pos':
            if word_score > 0 :
                TP+=1
            if word_score < 0 :
                FP+=1
        if test_label[index_w] == 'neg':
            if word_score > 0 :
                FN+=1
            if word_score < 0 :
                TN+=1
    ACC = (TP+TN)/(TP+TN+FP+FN)
    print('TP: ', TP)
    print('TN: ', TN)
    print('FP: ', FP)
    print('FN: ', FN)
    print("ACC值为：",str(ACC))
    f_neg.close()
    f_pos.close()
    return

def sopmi(train_path,sentiment_path,candipos_path,candineg_path,level_path,test_path):
    print('step 1/4:...seg corpus ...')
    start_time  = time.time()
    seg_data = seg_corpus(train_path, sentiment_path,stop_words_path,level_path,test_path)
    end_time1 = time.time()
    print('step 1/4 finished:...cost {0}...'.format((end_time1 - start_time)))
    print('step 2/4:...collect cowords ...')
    cowords_list = collect_cowords(sentiment_path, seg_data)
    end_time2 = time.time()
    print('step 2/4 finished:...cost {0}...'.format((end_time2 - end_time1)))
    print('step 3/4:...compute sopmi ...')
    pmi_dict = collect_candiwords(seg_data,test_path,cowords_list, sentiment_path)
    print(pmi_dict)
    end_time3 = time.time()
    print('step 1/4 finished:...cost {0}...'.format((end_time3 - end_time2)))
    print('step 4/4:...save candiwords ...')
    save_candiwords(pmi_dict, candipos_path, candineg_path,test_path)
    end_time = time.time()
    print('finished! cost {0}'.format(end_time - start_time))

if __name__ == '__main__':
    sopmi(train_path,sentiment_path,candipos_path,candineg_path,level_path,test_path)
    '''
    seg_data = seg_corpus(train_path, sentiment_path)
    cowords_list = collect_cowords(sentiment_path, seg_data)
    pmi_dict = collect_candiwords(seg_data, cowords_list, sentiment_path)
    save_candiwords(pmi_dict, candipos_path, candineg_path)
    '''



    
 
