"""
2016.6.24
Algorithms for building and using a decision tree for classification.
"""
import numpy
import math
import random
from collections import defaultdict


class DecisionTreeClassifier:
	def __init__(self, **kwargs):
		# criterion:string, optional (default='gini')
		# the function to measure the quality of a split. 'gini' or 'entropy',
		self.criterion = kwargs.get('criterion', 'entropy')
		assert isinstance(self.criterion, (str, None))
		# max_features: int, float, string or None, optional (default=None)
		# The number of features to consider when looking for the best split.
		# if int, then
		# consider max_features features at each split.
		# if float, then
		# max_features is a percentage and int(max_features * n_features) features are considered at each split.
		# if 'auto', then
		# max_features=sqrt(n_features).
		# if 'sqrt', then
		# max_features=sqrt(n_features).
		# if 'log2', then
		# max_features=log2(n_features).
		# if None, then
		# max_features=n_features
		self.max_features = kwargs.get('max_features', None)
		assert self.max_features is None or isinstance(self.max_features, (int, float, str))
		# max_depth: int or None, optional (default=None)
		# The maximum depth of the tree. if None, then nodes are expanded until all leaves are pure or
		# until all leaves contain less than min_samples_split samples. ignored if max_leaf_nodes if not None
		self.max_depth = kwargs.get('max_depth', None)
		assert self.max_depth is None or isinstance(self.max_depth, int)
		# min_samples_split : int or None, optional (default=2)
		# The minimum of samples required to split an internal node.
		self.min_samples_split = kwargs.get('min_samples_split', 2)
		assert self.min_samples_split is None or isinstance(self.min_samples_split, int)
		# min_samples_leaf : int or None, optional (default=1)
		# The minimum number of samples required to be at a leaf node.
		self.min_samples_leaf = kwargs.get('min_samples_leaf', 1)
		assert self.min_samples_leaf is None or isinstance(self.min_samples_leaf, int)
		# classes_ : array of shape =[n_classes]
		# The classes labels.
		self.classes_ = None
		# feature_importance_ : array of shape=[n_features]
		# The feature importances. The higher, the more important the feature. The importance of a feature is
		# computed as the (normalized) total reduction of the criterion brought by that feature. it is also known
		# as the Gini importance.
		self.feature_importance_ = None
		# max_features_ : int or None
		# The inferred value of max_features
		self.max_features_ = None
		# n_classes_ : int or None
		# The number of classes.
		self.n_classes_ = None
		# n_features_ : int or None
		# The number of features when fit is performed
		self.n_features_ = None
		# tree_ : Tree object
		# The underlying Tree object
		self.tree_ = None
		# _random_columns_ : list or None
		# random select n_features when fit is performed
		self.__random_columns_ = None
		# training_samples_ : numpy.ndarray
		self.training_samples_ = None
		# targets_ : numpy.ndarray
		self.targets_ = None
		# feature_values_ : numpy.ndarray
		# all feature values of all features
		self.feature_values_ = None
		# sample_counts_ : int or None
		# The number oa samples
		self.samples_counts_ = None
		pass

	def fit(self, X, y):
		"""
		Build a decision tree from the training set(X,y)
		:param X: The training input samples.
		:param y: The target values.
		:return: Return self.
		"""
		assert isinstance(X, numpy.ndarray)
		assert isinstance(y, numpy.ndarray)
		assert len(X) == len(y)
		if len(X) < 1:
			raise ValueError("No training data!")
		self.classes_ = list(set(y))
		self.n_classes_ = len(self.classes_)
		features_total = len(X[0])
		# Calculate the inferred value of max_features
		if self.max_features is None:
			self.max_features_ = features_total
		elif isinstance(self.max_features, int):
			self.max_features_ = min(self.max_features, features_total)
		elif isinstance(self.max_features, float):
			if self.max_features > 1 or self.max_features < 0:
				raise ValueError("max_features(float) should be between 0 and 1.")
			self.max_features_ = max(int(self.max_features * features_total), 1)
		elif self.max_features == 'auto' or self.max_features == 'sqrt':
			self.max_features_ = int(math.sqrt(features_total))
		elif self.max_features == 'log2':
			self.max_features_ = max(int(math.log2(features_total)), 1)
		else:
			raise ValueError("max_features: %s undefined", self.max_features)
		self.n_features_ = self.max_features_
		self.feature_importance_ = numpy.zeros(self.n_features_)
		# filter features
		if self.n_features_ != features_total:
			self.__random_columns_ = list()
			for i in range(self.n_features_):
				self.__random_columns_.append(random.randint(0, features_total - 1))
			self.training_samples_ = X[:, self.__random_columns_]
		else:
			self.training_samples_ = X
		self.targets_ = y
		self.samples_counts_ = len(self.training_samples_)
		# assign the criterion function
		if self.criterion == 'entropy':
			criterion_func = DecisionTreeClassifier.__information_gain
		elif self.criterion == 'gini':
			criterion_func = DecisionTreeClassifier.__information_gain
		else:
			raise
		self.feature_importance_ = criterion_func(X, y)
		# Calculate the inferred value of feature_values
		self.feature_values_ = list()
		for i in range(len(self.training_samples_[0])):
			self.feature_values_.append(set(self.training_samples_[:, i]))
		self.feature_values_ = numpy.array(self.feature_values_)

		# create a decision tree
		self.tree_ = Tree(
			self.create_decision_tree(
				rows=[i for i in range(self.samples_counts_)],
				cols=[i for i in range(self.n_features_)],
				criterion_func=criterion_func,
			)
		)
		return self

	def create_decision_tree(self, rows, cols, criterion_func):
		data = self.training_samples_[rows, :]
		data = data[:, cols]
		targets = self.targets_[rows]
		features = self.feature_values_[cols]
		# if all samples belong to ONE class. Return a leaf node
		unique_classes = list(set(targets))
		if len(unique_classes) == 1:
			return Node(
				result=unique_classes[0],
			)
		if len(rows) == 0 or DecisionTreeClassifier.__is_single_value(data):
			# return a leaf node labeled by the class having the most samples

			return Node(
				result=DecisionTreeClassifier.__get_mode(targets),
			)
		# split a leaf node
		# select the best attribute to split the node
		feature_importance = criterion_func(data, targets)
		feature_selected = feature_importance.index(max(feature_importance))
		node = Node()
		node.feature = cols[feature_selected]
		node.result = None
		# each value of the feature selected
		for value in features[feature_selected]:
			index = list(filter(lambda i: data[i, feature_selected] == value, [i for i in range(len(rows))]))
			if len(index) == 0:
				node.add_child(
					value=value,
					node=Node(
						result=DecisionTreeClassifier.__get_mode(targets),
					)
				)
			else:
				rows_next = [rows[i] for i in index]
				cols_next = list(cols)
				del cols_next[feature_selected]
				node.add_child(
					value=value,
					node=self.create_decision_tree(
						rows=rows_next,
						cols=cols,
						criterion_func=criterion_func,
					)
				)
		return node

	@staticmethod
	def __entropy(data):
		"""
		Calculates the entropy of the attribute attr in given data.

		:return: the entropy value.
		"""
		assert isinstance(data, (numpy.ndarray, list))
		data = list(data)
		counts = dict((i, data.count(i)) for i in data)
		len_data = len(data)
		return -sum((count / len_data) * math.log2(count / len_data) for count in counts.values() if count)

	@staticmethod
	def __information_gain(data, target):
		"""
		Calculates the information gain (reduction in entropy) that would result by splitting
		the data on the chosen attribute.

		:param data: training samples
		:param target: target values
		:return: the information gain
		"""
		result = list()
		len_data = len(data)
		result_entropy = DecisionTreeClassifier.__entropy(target)
		for i in range(len(data[0])):
			feature_values_set = set(data[:, i])
			feature_entropy = 0.0
			for each_feature_value in feature_values_set:
				index = list(filter(lambda j: data[j, i] == each_feature_value, [j for j in range(len(data))]))
				feature_entropy += float(len(index)) / len_data * DecisionTreeClassifier.__entropy(target[index])
			result.append(result_entropy - feature_entropy)
		return result

	@staticmethod
	def __get_mode(l):
		appear = defaultdict(int)
		for item in l:
			appear[item] += 1
		max_times = max(appear.values())
		for key, value in appear.items():
			if value == max_times:
				return key

	@staticmethod
	def __is_single_value(data):
		"""
		Judge whether each feature only has single value
		:param data: training data
		:return: bool
		"""
		for i in range(len(data[0])):
			if len(set(data[:, i])) > 1:
				return False
		return True


class Tree:
	def __init__(self, root_node=None):
		self.root_node = root_node
		# The number of nodes
		self.number_ = 1
		# The number of leaf nodes
		self.leaf_number_ = 1
		pass

	def __len__(self):
		return self.number_


class Node:
	def __init__(self, **kwargs):
		# The feature to split to subtree
		self.__feature = kwargs.get('feature', None)
		# saved the target result only in leaf nodes.
		self.__result = kwargs.get('result', None)
		# A dict of Node, each value is a subtree when the feature equals the key
		self.__children = kwargs.get('children', None)

	def add_child(self, value, node):
		assert isinstance(node, Node)
		if self.__children is None:
			self.__children = dict()
		self.__children[value] = node

	@property
	def feature(self):
		return self.__feature

	@feature.setter
	def feature(self, value):
		assert value is None or isinstance(value, int)
		self.__feature = value

	@property
	def result(self):
		return self.__result

	@result.setter
	def result(self, value):
		assert value is None or isinstance(value, int)
		self.__result = value


if __name__ == '__main__':
	import csv

	lines = csv.reader(open('cdata6.csv'))
	cdata = list()
	for line in lines:
		cdata.append(line)
	cdata = numpy.array(cdata)
	X = numpy.array([[1, 1, 1], [0, 0, 0], [1, 0, 0], [0, 1, 1], [0, 1, 0]])
	y = numpy.array([1, 0, 0, 1, 1])
	model = DecisionTreeClassifier(criterion='entropy')
	model.fit(cdata[:, 0:-1], cdata[:, -1])
	pass
