import sys
import random

from pyspark import SparkContext, SparkConf
from functools import partial
import numpy as np
import logging

def word_count_users(line):
    split_list = line.replace(' ', '').split(',')
    user_id = int(split_list[0])
    return user_id

def word_count_movies(line):
    split_list = line.replace(' ', '').split(',')
    movie_id = int(split_list[1])
    return movie_id

def parse_data_set(line, user_map, movie_map):
    user_map = user_map.value
    movie_map = movie_map.value

    split_list = line.replace(' ', '').split(',')
    user_id = split_list[0]
    movie_id = split_list[1]
    rating = split_list[2]

    return user_map[int(user_id)], movie_map[int(movie_id)], float(rating)

def tuplize(mat):
    res = []

    args = mat.nonzero()
    for row, col in zip(args[0], args[1]):
        res.append((row, col, mat[row, col]))

    return res

def tuplize_w(w, num_factors):
    res = []

    for key in w:
        arr = w[key]
        for i in xrange(num_factors):
            res.append((key, i, arr[i]))

    return res

def tuplize_h(h, num_factors):
    res = []

    for key in h:
        arr = h[key]
        for i in xrange(num_factors):
            res.append((i, key, arr[i]))

    return res

# Map each user_id to serial numbers.
# Map each movie_id to a serial list.
def map_partition_function(block, nw, nh, lambda_value, num_factors, beta_value, num_iteration):

    nw = nw.value
    nh = nh.value
    lambda_value = lambda_value.value
    num_factors = num_factors.value
    beta_value = beta_value.value
    num_iteration = num_iteration.value

    if block:
        for tuple in block:
            v_b = tuple[1][0]
            w_b = tuple[1][1]
            h_b = tuple[1][2]

            w_b_matrix = {}
            h_b_matrix = {}

            for factor_id, movie_id, value in h_b:
                if movie_id not in h_b_matrix:
                    h_b_matrix[movie_id] = np.zeros(num_factors)

                h_b_matrix[movie_id][factor_id] = value

            for user_id, factor_id, value in w_b:
                if user_id not in w_b_matrix:
                    w_b_matrix[user_id] = np.zeros(num_factors)

                w_b_matrix[user_id][factor_id] = value

            total_loss = 0.0
            for user_id, movie_id, rating in v_b:
                eps = (num_iteration + 100) ** (-1.0 * beta_value)

                movie = np.copy(h_b_matrix[movie_id])
                user = np.copy(w_b_matrix[user_id])

                dot_product = np.inner(movie, user)

                v = rating - dot_product
                total_loss += v * v

                update = 2.0 * (rating - dot_product)

                for factor in xrange(0, num_factors):
                    w_b_matrix[user_id][factor] += eps * update * movie[factor] -\
                                                   2.0 * eps * lambda_value / float(nw[user_id]) * user[factor]
                    h_b_matrix[movie_id][factor] += eps * update * user[factor] -\
                                                    2.0 * eps * lambda_value / float(nh[movie_id]) * movie[factor]

            return (0, tuplize_w(w_b_matrix, num_factors)), (1, tuplize_h(h_b_matrix, num_factors)), (2, total_loss)

    return (0, []), (1, [])

# Map each user_id to serial numbers.
# Map each movie_id to a serial list.
def compute_loss(block, num_factors):

    num_factors = num_factors.value


    if block:
        for tuple in block:
            v_b = tuple[1][0]
            w_b = tuple[1][1]
            h_b = tuple[1][2]

            w_b_matrix = {}
            h_b_matrix = {}

            for factor_id, movie_id, value in h_b:
                if movie_id not in h_b_matrix:
                    h_b_matrix[movie_id] = np.zeros(num_factors)

                h_b_matrix[movie_id][factor_id] = value

            for user_id, factor_id, value in w_b:
                if user_id not in w_b_matrix:
                    w_b_matrix[user_id] = np.zeros(num_factors)

                w_b_matrix[user_id][factor_id] = value

            total_loss = 0.0

            for user_id, movie_id, rating in v_b:

                movie = np.copy(h_b_matrix[movie_id])
                user = np.copy(w_b_matrix[user_id])

                dot_product = np.inner(movie, user)

                v = rating - dot_product
                total_loss += v * v

            return [total_loss]

    return (0, []), (1, [])



def print_rdd(W):
    for rdd in W.collect():
        print rdd

def dump_rdd(mat, filepath, dimx, dimy):
    """
    Dumps an rdd of (row, col, value) into a text file using numpy arrays.
    """
    shape = (dimx, dimy)
    arr = np.zeros(shape)

    for row, col, rating in mat.collect():
        arr[row][col] = rating

    np.savetxt(filepath, arr, delimiter=',')

def should_choose_cells(row, col, iteration, num_workers):
    row = row % num_workers
    col = col % num_workers
    iteration = iteration % num_workers

    if col > row:
        return ((col - row) % num_workers) == iteration
    else:
        return ((num_workers - row + col) % num_workers) == iteration

def main():
    """
    Spark driver program.
    """
    num_factors = int(sys.argv[1])
    num_workers = int(sys.argv[2])
    num_iterations = int(sys.argv[3])
    beta_value = float(sys.argv[4])
    lambda_value = float(sys.argv[5])
    inputV_filepath = sys.argv[6]
    outputW_filepath = sys.argv[7]
    outputH_filepath = sys.argv[8]

    master = "local[{0}]".format(num_workers)
    print "Using master {0}".format(master)

    conf = SparkConf().setAppName("dsgd").setMaster(master)
    logging.getLogger("dsgd").setLevel(logging.ERROR)
    sc = SparkContext(conf=conf)

    rdd_lines = sc.textFile(inputV_filepath)

    user_count = 0
    movie_count = 0

    sorted_users = rdd_lines.map(word_count_users).distinct().sortBy(lambda x: x)
    sorted_movies = rdd_lines.map(word_count_movies).distinct().sortBy(lambda x: x)

    user_map = {}
    movie_map = {}

    for user in sorted_users.collect():
        user_map[user] = user_count
        user_count += 1

    for movie in sorted_movies.collect():
        movie_map[movie] = movie_count
        movie_count += 1

    num_users = user_count
    num_movies = movie_count
    W = []
    H = []

    bc_user_map = sc.broadcast(user_map)
    bc_movie_map = sc.broadcast(movie_map)

    V = rdd_lines.map(partial(parse_data_set, user_map=bc_user_map, movie_map=bc_movie_map))

    nw = V.countByKey()
    nh = V.map(lambda x: (x[1], x[2])).countByKey()

    bc_nw = sc.broadcast(nw)
    bc_nh = sc.broadcast(nh)
    bc_lambda_value = sc.broadcast(lambda_value)
    bc_num_factors = sc.broadcast(num_factors)
    bc_beta_value = sc.broadcast(beta_value)

    LOWER_BOUND = 0.0
    UPPER_BOUND = 1.0

    for user in xrange(0, num_users):
        for i in xrange(0, num_factors):
            W.append((user, i, random.uniform(LOWER_BOUND, UPPER_BOUND)))
    W = sc.parallelize(W)

    for i in xrange(0, num_factors):
        for movie in xrange(0, num_movies):
            H.append((i, movie, random.uniform(LOWER_BOUND, UPPER_BOUND)))

    H = sc.parallelize(H)

    for iteration in xrange(0, num_iterations):

        w = W.keyBy(lambda x: x[0] % num_workers)

        h = H.keyBy(lambda x: ((x[1] - iteration) % num_workers))

        # Time to partition.
        v = V.keyBy(lambda x: x[0] % num_workers) \
            .filter(lambda x: should_choose_cells(x[0], x[1][1], iteration, num_workers))

        bc_num_iteration = sc.broadcast(iteration)

        rdd = v.groupWith(w, h).partitionBy(num_workers).mapPartitions(
            partial(map_partition_function,
                    nw=bc_nw,
                    nh=bc_nh,
                    lambda_value=bc_lambda_value,
                    num_factors=bc_num_factors,
                    beta_value=bc_beta_value,
                    num_iteration=bc_num_iteration))

        W = rdd.filter(lambda x: x[0] == 0).map(lambda x: x[1]).flatMap(lambda x: x)
        H = rdd.filter(lambda x: x[0] == 1).map(lambda x: x[1]).flatMap(lambda x: x)
        L = rdd.filter(lambda x: x[0] == 2).map(lambda x: x[1]).reduce(lambda a, b: a + b)

        if iteration % num_workers == 0:
            t = 0
            loss = 0.0

            while t < num_workers:
                w = W.keyBy(lambda x: x[0] % num_workers)
                h = H.keyBy(lambda x: ((x[1] - iteration - t) % num_workers))

                # Time to partition.
                v = V.keyBy(lambda x: x[0] % num_workers) \
                    .filter(lambda x: should_choose_cells(x[0], x[1][1], iteration + t, num_workers))

                loss += v.groupWith(w, h).partitionBy(num_workers).mapPartitions(
                    partial(compute_loss, num_factors=bc_num_factors)).reduce(lambda a, b: a + b)

                t += 1

            print "Iteration{0}, Total Loss {1}".format(int(iteration/num_workers) + 1, loss)



    dump_rdd(W, outputW_filepath, user_count, num_factors)
    dump_rdd(H, outputH_filepath, num_factors, movie_count)

if __name__ == "__main__":
    main()