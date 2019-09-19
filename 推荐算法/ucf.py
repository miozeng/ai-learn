# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import numpy as np
import datetime
import codecs
import csv
import sys
import time
import math
from scipy.stats import pearsonr

class UcfRecommend:
  
   def __init__(self, file):
      self.user_data =  pd.read_csv(file)
      self.user_item_feture = self.item_feture()
   
   def date_cal(self,time):
      action_time = datetime.datetime.strptime(time,"%Y-%m-%d %H")
      end_date = datetime.datetime(2014,12,19,0,0,0)
      self.cdate = end_date - action_time 
      return self.cdate.days 
  
   def item_feture(self):
        self.user_data['days'] = list(map(self.date_cal, self.user_data['time']))
        # 获取一天内得数据
        before1day = self.user_data[self.user_data['days']<=0]
        # 获取1-5天内得数据
        before5day_ = self.user_data[self.user_data['days']<=5 ]
        before5day = before5day_[before5day_['days']>0]
        # 获取5天以前得数据
        before30day = self.user_data[self.user_data['days']>5]
        
        #处理user_id 和item_category 按时间统计的数据
        user_item_count = pd.crosstab([self.user_data.user_id,self.user_data.item_id],self.user_data.behavior_type)
        user_cate_count_1 = pd.crosstab([before1day.user_id,before1day.item_id],before1day.behavior_type)
        user_cate_count_2 = pd.crosstab([before5day.user_id,before5day.item_id],before5day.behavior_type)
        user_cate_count_3 = pd.crosstab([before30day.user_id,before30day.item_id],before30day.behavior_type)
        
        #合并数据
        user_cate_feture = pd.merge(user_item_count,user_cate_count_1,how='left',right_index=True,left_index=True)
        user_cate_feture = pd.merge(user_cate_feture,user_cate_count_2,how='left',right_index=True,left_index=True)
        user_cate_feture = pd.merge(user_cate_feture,user_cate_count_3,how='left',right_index=True,left_index=True)
        #空值用0代替
        user_cate_feture.fillna(0,inplace=True)
        
        #重设表格
        user_cate_feture = user_cate_feture.reset_index()
        
        #columns 重新赋值
        user_cate_feture.columns = [ 'user_id', 'item_id', '1_total', '2_total', '3_total', '4_total',
                  '1_oneday', '2_oneday', '3_oneday', '4_oneday', '1_fiveday', '2_fiveday', '3_fiveday', '4_fiveday',
                               '1_fivedayafter', '2_fivedayafter', '3_fivedayafter', '4_fivedayafter']
        
        
        user_cate_feture_2=user_cate_feture.copy()
        
        #给他们打分 一天内的浏览、收藏、加购物车、购买 分别是4，8，12，16分  1-5天分别是2，4，6，8   5天以上的 1，2，3，4 
        user_cate_feture_2['1_oneday'] = user_cate_feture['1_oneday']*4
        user_cate_feture_2['2_oneday'] = user_cate_feture['2_oneday']*8
        user_cate_feture_2['3_oneday'] = user_cate_feture['3_oneday']*12
        user_cate_feture_2['4_oneday'] = user_cate_feture['4_oneday']*16
        user_cate_feture_2['1_fiveday'] = user_cate_feture['1_fiveday']*2
        user_cate_feture_2['2_fiveday'] = user_cate_feture['2_fiveday']*4
        user_cate_feture_2['3_fiveday'] = user_cate_feture['3_fiveday']*6
        user_cate_feture_2['4_fiveday'] = user_cate_feture['4_fiveday']*8
        user_cate_feture_2['1_fivedayafter'] = user_cate_feture['1_fivedayafter']*1
        user_cate_feture_2['2_fivedayafter'] = user_cate_feture['2_fivedayafter']*2
        user_cate_feture_2['3_fivedayafter'] = user_cate_feture['3_fivedayafter']*3
        user_cate_feture_2['4_fivedayafter'] = user_cate_feture['4_fivedayafter']*4
        #user_cate_feture_2 = user_cate_feture_2.drop(['1_total'])
        user_cate_feture_2['point'] = user_cate_feture_2[['1_oneday', '2_oneday', '3_oneday', '4_oneday', '1_fiveday', '2_fiveday', '3_fiveday', '4_fiveday',
                               '1_fivedayafter', '2_fivedayafter', '3_fivedayafter', '4_fivedayafter']].apply(lambda x : x.sum(),axis=1 ) 
        
        
        col_list = ['user_id', 'item_id','point']
        self.user_cate_feture_3 = user_cate_feture_2[col_list]
        
        
        return self.user_cate_feture_3
    
   def user_dict(self):
        user_item_dict = dict()
        for index,row in self.user_item_feture.iterrows():
            user_item_dict.setdefault(row['user_id'], {})
            user_item_dict[row['user_id']][row['item_id']] =row['point']
        self.user_item_pd = pd.DataFrame.from_dict(user_item_dict)
        self.user_item_pd.fillna(0,inplace=True)
        return self.user_item_pd
    
   def UserSimilarity(self):
        self.C = {} #User 和User的相似矩阵
        for indexs in self.user_item_pd.columns:
            self.C.setdefault(indexs, {})
            for indexsc in self.user_item_pd.columns:
                if indexs ==indexsc:
                    continue
                pccs = np.corrcoef(self.user_item_pd[indexs], self.user_item_pd[indexsc])
                #print(pccs[0][1])
                self. C[indexs].setdefault(indexsc,0)
                self.C[indexs][indexsc] =np.abs(pccs[0][1])
        self.C_pd = pd.DataFrame.from_dict(self.C)
        self.C_pd.fillna(0,inplace=True)
        return self.C_pd 

   def UCF_Recommend(self,user_id, K=3,N=10):
        # user_id,需要推荐的用户  user_item_pd 用户商品二维矩阵 C，用户关系矩阵  K 选取相关用户的前K个商品，N推荐N个商品
        self.rank = {}
        action_item = self.user_item_pd[user_id]
        #根据user_id对应的列排序，得到排序高的前K行对应用户的数据
        top_C = self.C_pd.sort_values(by=user_id , ascending=False)
        top_C = top_C.iloc[0:K]
        #print(top_C)
        #遍历排名高的前K个用户
        for index,row in top_C.iterrows():
            # 因为top_C是用户相关系数矩阵所以index是相关用户的userid
        #     print(index)
            #得到用户user_id 与index 的分数
            uscore = row[user_id]
            #print("uscore"+str(uscore))
            #得到当前相似用户打分高的前K条记录
            ui = self.user_item_pd.sort_values(by=index , ascending=False).iloc[0:K]
            #user_item_pd 是用户-物品打分相关矩阵，所以index2是item_id
            for index2,row2 in ui.iterrows():
        #         print(index2)
        #         print(row2[index])
                #得到此用户index 对当前商品的打分
                uiscore = row2[index]
                
                if  action_item[index2] >0:
                    #print('yes')
                    continue
                #print('no')
                if index2 not in self.rank.keys():
                    self.rank.setdefault(index2, 0)
                #根据当前用户index对此商品的评分以及用户相关系数的乘积，得到需推荐用户user_id对应的item得分
                self.rank[index2] += uscore * uiscore
        if  self.rank:
            self.rank = dict(sorted(self.rank.items(),key=lambda x:x[1], reverse=True)[0:N])
        return self.rank
    
   def RecommendAll(self):
       self.info=[]
       for index,row in self.user_item_pd.iteritems():
           dt=[]
           dt.append(str(index))
           #print(index)
           rank = ucf.UCF_Recommend(index)
           reitem = list(rank.keys())[0]
           dt.append(str(reitem))
           #print(reitem)
           self.info.append(dt)
         
   def my_write_csv(self):
        with open("C:/Users/Coffee/ai/ai-learn/data/user/Recommend_info.csv","w",newline='') as csvfile: ##“ ab+ ”去除空白行，又叫换行！
            #csvfile.write("utf-8")  ##存入表内的文字格式
            writer = csv.writer(csvfile)  #存入表时所使用的格式
            title1 = 'user_id'
            title2 = 'item_id'
            writer.writerow([title1,title2])
           # print(self.info)
            writer.writerows(self.info) #写入表
            
            
#创建对象tianchi_fresh_comp_train_user test_user.csv
print('start read')
ucf = UcfRecommend("C:/Users/Coffee/ai/ai-learn/data/user/test_user.csv")
#print(ucf.user_item_feture.head())
ucf.user_dict()
print('start simi')
ucf.UserSimilarity()
#rank = ucf.UCF_Recommend(10001082.0,N=1)
#print(list(rank.keys())[0])
#print(type(rank.keys()) )
print('start RecommendAll')
ucf.RecommendAll()
print('start my_write_csv')
ucf.my_write_csv()



   


