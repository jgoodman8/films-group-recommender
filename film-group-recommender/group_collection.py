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
        avbl_users = [i for i in range(users_number)]
        groups = []
        testable_threshold = 50

        iter_idx = 0
        while iter_idx in range(self.collection_size):
            group_members = numpy.random.choice(avbl_users, size=self.groups_size, replace=False)

            candidate_items = Group.find_members_subset(self.train_ratings, group_members)
            non_evaluable_items = Group.find_members_subset(self.test_ratings, group_members)
            testable_items = numpy.setdiff1d(candidate_items, non_evaluable_items)

            if len(candidate_items) != 0 and len(testable_items) >= testable_threshold:

                groups += [Group(group_members, candidate_items, self.train_ratings)]

                if disjoint:
                    avbl_users = numpy.setdiff1d(avbl_users, group_members)
                iter_idx += 1

        self.groups = groups
