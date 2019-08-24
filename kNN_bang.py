#!/usr/bin/env python
# coding: utf-8

"""
# ##### Для отладки алгоритма и сравнения результатов использовалась учебная выборка с kaggle

# Задача на kaggle: https://www.kaggle.com/c/bioresponse
# Данные: https://www.kaggle.com/c/bioresponse/data
# По данным характеристикам молекулы требуется определить, будет ли дан биологический ответ (biological response). Признаки нормализованы.

# ##### Вызов алгоритма
# from kNN_bang import KNN

# ##### Ниже примеры использования синтаксических конструкций алгоритма. Имена принимаемых аргументов аналогичны используемым в стандартном.  

# запуск алгоритма при базовых настройках, то есть при 
# 5 соседях, без применения весов, и расчете дистанции по Эвклидовой метрике
#test_base = KNN()			# данный метод создает объект класса с базовыми настройками 
#test_base.fit(X_train, y_train)	# данный метод загружатет обучающую выборку и целевой вектор
#test_base.predict(X_test)		# данный метод создает вектор предсказаний целевых значений
#test_base.accuracy(y_test) 			# данный метод считает точность предсказаний

# запуск алгоритма при 3 соседях, без применения весов, и расчете дистанции
# по Эвклидовой метрике
#test_uniform = KNN(n_neighbors=3, weights='uniform', p=2)
#test_uniform.fit(X_train, y_train)
#test_uniform.predict(X_test)
#test_uniform.accuracy(y_test)


# запуск алгоритма при 5 соседях, применении весов, и расчете дистанции
# по Эвклидовой метрике выше аналогичного расчета при использовании стандартного модуля
#test_distance = KNN(n_neighbors=5, weights='distance', p=2)
#test_distance.fit(X_train, y_train)
#test_distance.predict(X_test)
#test_distance.accuracy(y_test)

# запуск алгоритма при 5 соседях, применении весов, и расчете дистанции
# по Манхэттенской метрике
#test_distance_man = KNN(n_neighbors=5, weights='distance', p=1)
#test_distance_man.fit(X_train_50, y_train_50)
#test_distance_man.predict(X_test_25)
#test_distance_man.accuracy(y_test)

"""

import pandas as pd
import numpy as np
from collections import defaultdict, Counter
import operator

class KNN:
 
    def __init__(self, n_neighbors=5, weights='uniform', p=2):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.p = p

    def fit(self, X_train, y_train):
        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)
 
    def predict(self, X_test):
        p = self.p
        n_neighbors = self.n_neighbors
        weights = self.weights
        X_train = self.X_train
        y_train = self.y_train
        X_test = np.array(X_test)
        pediction_vector = []
        for i in range(len(X_test)): 
            dictance_between_objects_array = []
            if p == 2:
                for j in range(len(X_train)):
                    e_dist = np.linalg.norm(X_test[i] - X_train[j])
                    dictance_between_objects_array.append(e_dist)
            elif p == 1:
                for j in range(len(np.array(X_train))):
                    m_dist = map(lambda x,y: math.fabs(float(x)-float(y)),
                                 np.array(X_test)[i],np.array(X_train)[j]) 
                    m_dist = sum(m_dist)/100.
                    dictance_between_objects_array.append(m_dist)
            neighbors_dictionary = {}
            for i, j in zip(dictance_between_objects_array, y_train):
                neighbors_dictionary.setdefault(i,[]).append(j)
            sorted_distances = sorted(neighbors_dictionary.keys())
            self.sorted_distances = sorted_distances
            
            neighbors_list = []
            if weights == 'uniform':
                for g in range(n_neighbors):
                    neighbors_list.append(neighbors_dictionary.get(sorted_distances[g]))
                    
                neighbor = []
                for h in neighbors_list:    
                    neighbor.append(h[0]) 
                counter = Counter(neighbor)
                prediction = max(counter.iteritems(), key=operator.itemgetter(1))[0]
                pediction_vector.append(prediction)
                self.pediction_vector = pediction_vector    
            
            elif weights == 'distance':
                short_neighbors_dictionary = {}
                short_key = []
                short_values = []
                for g in range(n_neighbors):
                    short_key.append(1/float(g+1))
                    short_values.append(neighbors_dictionary.get(sorted_distances[g])[0])
                for i, j in zip(short_key, short_values):
                    short_neighbors_dictionary.setdefault(j,[]).append(i) 
                
                sum_dict = {}
                sum_key = []
                sum_value = []
                for i in range(len(short_neighbors_dictionary.items())):
                    sum_value.append(short_neighbors_dictionary.items()[i][0])
                    sum_key.append(sum(short_neighbors_dictionary.items()[i][1]))
                for i, j in zip(sum_key, sum_value):
                    sum_dict.setdefault(i,[]).append(j)
                sorted_weights = sorted(sum_dict.items())
                pediction_vector.append(sorted_weights[-1][1])
                self.pediction_vector = pediction_vector        
    def accuracy(self, y_test):
	pediction_vector = self.pediction_vector 
	self.y_test = np.array(y_test)
        
        right_prediction = 0
        for i in range(len(pediction_vector)):
            if pediction_vector[i] == y_test[i]:
                right_prediction += 1
            else:
                pass
        print 'KNN accuracy ', float(right_prediction)/len(pediction_vector)


