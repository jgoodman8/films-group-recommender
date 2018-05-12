import numpy


class Group:
    def __init__(self, user_ids, items, ratings):
        self.user_ids = sorted(user_ids)
        self.items = items
        self.ratings = ratings

        self.false_positives = []
        self.temporal_recommendations = []

        self.precision = 0
        self.recall = 0

        self.bias = 0
        self.factors = []

        self.weighted_matrix = []
        self.recommendation_list = []

    @staticmethod
    def get_ratings_subset(ratings, user_ids):
        if len(user_ids) == 0:
            return []

        ratings_subset = numpy.argwhere(ratings[user_ids[0]] == 0)
        for user_id in user_ids:
            user_ratings_subset = numpy.argwhere(ratings[user_id] == 0)
            ratings_subset = numpy.intersect1d(ratings_subset, user_ratings_subset)

        return ratings_subset

    def non_testable_items(self, ratings):
        non_eval_items = numpy.argwhere(ratings[self.user_ids[0]] == 0)

        for user_id in self.user_ids:
            cur_non_eval_items = numpy.argwhere(ratings[user_id] == 0)
            non_eval_items = numpy.intersect1d(non_eval_items, cur_non_eval_items)

        return non_eval_items

    def generate_actual_recommendations(self, ratings, threshold):
        non_eval_items = self.non_testable_items(ratings)

        items = numpy.argwhere(
            numpy.logical_or(ratings[self.user_ids[0]] >= threshold, ratings[self.user_ids[0]] == 0)).flatten()
        fp = numpy.argwhere(
            numpy.logical_and(ratings[self.user_ids[0]] > 0, ratings[self.user_ids[0]] < threshold)).flatten()

        for user_id in self.user_ids:
            cur_items = numpy.argwhere(numpy.logical_or(ratings[user_id] >= threshold, ratings[user_id] == 0)).flatten()
            fp = numpy.union1d(fp, numpy.argwhere(
                numpy.logical_and(ratings[user_id] > 0, ratings[user_id] < threshold)).flatten())
            items = numpy.intersect1d(items, cur_items)

        items = numpy.setdiff1d(items, non_eval_items)

        self.false_positives = fp
        self.temporal_recommendations = items

    def evaluate(self):
        true_positives = float(numpy.intersect1d(self.temporal_recommendations, self.recommendation_list).size)
        false_positives = float(numpy.intersect1d(self.false_positives, self.recommendation_list).size)

        try:
            self.precision = true_positives / (true_positives + false_positives)
        except ZeroDivisionError:
            self.precision = numpy.NaN

        try:
            self.recall = true_positives / self.temporal_recommendations.size
        except ZeroDivisionError:
            self.recall = numpy.NaN

        return self.precision, self.recall, true_positives, false_positives
