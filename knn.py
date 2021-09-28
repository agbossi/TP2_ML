import math

def euclidean_distance_from(x1):
    def euclidean_distance(x2):
        distance = 0
        for i in range(len(x1)):
            distance += (x1[i] - x2[i]) ** 2
        return math.sqrt(distance)
    return euclidean_distance


class knn:

    def __init__(self, x, f_x, classes):
        self.x = x
        self.f_X = f_x
        self.classes = classes
        self.k = 5

    # def classify(self, data):
    #     distances = list(map(euclidian_distance(data, self.x), self.f_X))

