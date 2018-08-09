#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 11:40:46 2018

@author: yangyang
"""

import data_utils as utils

test=utils.load_test(['../data/dataset-E2E-goal-oriented-test-v1.0/tst1/dialog-task1API-kb1_atmosphere-test1.json'],FLAGS)
cui=['italian', 'british', 'indian', 'french', 'spanish']
atm=['business', 'casual', 'romantic']
loc=['rome', 'london', 'bombay', 'paris', 'madrid']
num=['two', 'four', 'six', 'eight']
pri=['cheap', 'moderate', 'expensive']
c=[]
a=[]
l=[]
n=[]
p=[]
for s in test:
    cui_p=[]
    cui_p=[j for i in s['utter_list'] for j in i if j in cui]
    if cui_p[0] == s['a'][1]:
        c.append(1)
    else:
        c.append(0)
    
    atm_p=[]
    atm_p=[j for i in s['utter_list'] for j in i if j in atm]
    if atm_p[0] == s['a'][5]:
        a.append(1)
    else:
        a.append(0)
        
    loc_p=[]
    loc_p=[j for i in s['utter_list'] for j in i if j in loc]
    if loc_p[0] == s['a'][2]:
        l.append(1)
    else:
        l.append(0)
        
        
    num_p=[]
    num_p=[j for i in s['utter_list'] for j in i if j in num]
    if num_p[0] == s['a'][3]:
        n.append(1)
    else:
        n.append(0)
        
    pri_p=[]
    pri_p=[j for i in s['utter_list'] for j in i if j in pri]
    if pri_p[0] == s['a'][4]:
        p.append(1)
    else:
        p.append(0)

count=0
for i in range(len(test)):
    if c[i]==1 & a[i]==1 & l[i]==1 & n[i]==1 & p[i]==1 :
        count+=1
    
    
    