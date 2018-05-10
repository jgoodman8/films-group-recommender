import numpy


class Group:
    def __init__(self, user_ids, items, ratings):
        self.user_ids = sorted(user_ids)
        self.items = items
        self.ratings = ratings

        self.factors = []
        self.bias = 0
        self.precision_wbf = 0
        self.recall_wbf = 0
        self.weight_matrix_wbf = []
        self.recommendation_list = []

    @staticmethod
    def find_members_subset(ratings, members):
        if len(members) == 0:
            return []

        unwatched_items = numpy.argwhere(ratings[members[0]] == 0)
        for member in members:
            unwatched_by_member = numpy.argwhere(ratings[member] == 0)
            unwatched_items = numpy.intersect1d(unwatched_items, unwatched_by_member)

        return unwatched_items
