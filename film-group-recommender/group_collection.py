import numpy
from group import Group


class GroupCollection:
    def __init__(self, train_ratings, test_ratings):
        self.collection_size = 20
        self.groups_size = 50

        self.train_ratings = train_ratings
        self.test_ratings = test_ratings

        self.groups = []

    def create(self, users_number, disjoint=True):
        print('Creating', self.collection_size, 'groups of size', self.groups_size)
        groups = []
        subset_threshold = 50
        users_ids = [i for i in range(users_number)]

        group_count = 0
        while group_count in range(self.collection_size):
            user_id_fold = numpy.random.choice(users_ids, size=self.groups_size, replace=False)

            train_subset = Group.get_ratings_subset(self.train_ratings, user_id_fold)
            test_subset = Group.get_ratings_subset(self.test_ratings, user_id_fold)

            subset_intersection = numpy.setdiff1d(train_subset, test_subset)

            if len(train_subset) != 0 and len(subset_intersection) >= subset_threshold:
                groups += [Group(user_id_fold, train_subset, self.train_ratings)]

                if disjoint:
                    users_ids = numpy.setdiff1d(users_ids, user_id_fold)
                group_count += 1

        self.groups = groups
