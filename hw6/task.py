import sys
import numpy as np
from pyspark import SparkContext, StorageLevel
from sklearn.cluster import KMeans
from random import shuffle, seed
from itertools import combinations
from collections import defaultdict
from math import ceil
from time import perf_counter

def read_dataset(sc, filepath):
    return sc.textFile(filepath).map(lambda x: x.split(',')).persist(StorageLevel.MEMORY_AND_DISK)

def convert_to_numeric(rdd):
    return rdd.map(lambda x: [int(val) if i <= 1 else float(val) for i, val in enumerate(x)])

class BFR():

    def __init__(self, n_cluster, ds, cs, rs, summary_stats):
        self.n_cluster = n_cluster
        self.ds = ds
        self.cs = cs
        self.rs = rs
        self.summary_stats = summary_stats

    def _get_data(self, rdd):
        return rdd.map(lambda x: (x[0], x))

    def _get_keys(self, rdd):
        return rdd.map(lambda x: x[0]).collect()
    
    def _get_cluster(self, rdd):
        return rdd.map(lambda x: x[1]).collect()

    def _split_data_into_chunks(self, num_chunks, rdd, random_seed=1):
        seed(random_seed)
        keys = self._get_keys(rdd)
        shuffle(keys)
        len_chunks = ceil(len(keys)/num_chunks)
        chunks = []
        for i in range(num_chunks):
            chunk_set = set(keys[i*len_chunks:(i+1)*len_chunks])
            chunks.append(rdd.filter(lambda x: True if x[0] in chunk_set else False).collect())

        return chunks

    def _update_summary_stats(self, clust, set_type):
        if (clust[0], set_type) not in summary_stats:
            self.summary_stats[(clust[0], set_type)] = {}
            self.summary_stats[(clust[0], set_type)]["N"] = 0
            self.summary_stats[(clust[0], set_type)]["SUM"] = np.zeros(clust[1][2:].shape)
            self.summary_stats[(clust[0], set_type)]["SUMSQ"] = np.zeros(clust[1][2:].shape)
        else:
            self.summary_stats[(clust[0], set_type)]["N"] += 1
            self.summary_stats[(clust[0], set_type)]["SUM"] += clust[1][2:]
            self.summary_stats[(clust[0], set_type)]["SUMSQ"] += np.square(clust[1][2:])

    def _compute_mahalanobis(self, arr, clust_stats):
        return np.sqrt(np.sum(np.square((arr-clust_stats["SUM"]/clust_stats["N"])/np.sqrt(clust_stats["SUMSQ"]/clust_stats["N"]\
            -(clust_stats["SUM"]/clust_stats["N"])**2))))

    def fit(self, sc, rdd, random_seed=1):
        ##### INITIALIZATION ROUND #####
        data = self._get_data(rdd).collectAsMap()
        round_result = {}
        chunks = self._split_data_into_chunks(5, rdd, random_seed)
        start_sample_rdd = sc.parallelize(chunks[0])
        start_sample_arr = np.array(chunks[0])

        kmeans_first20 = KMeans(self.n_cluster*5, random_state=random_seed).fit(start_sample_arr[:, 2:])
        labeled_clusters = list(zip(kmeans_first20.labels_, start_sample_arr))
        singleton_clusters = sc.parallelize(labeled_clusters).groupByKey().mapValues(list) \
            .filter(lambda x: True if len(x[1]) == 1 else False).collect()
        for clust in singleton_clusters:
            self.rs.add((-1, int(clust[1][0][0])))

        start_sample_without_rs_rdd = start_sample_rdd.filter(lambda x: True if (-1, x[0]) not in self.rs else False)
        start_sample_without_rs_arr = np.array(start_sample_without_rs_rdd.collect())

        kmeans_first20_without_rs = KMeans(self.n_cluster, random_state=random_seed).fit(start_sample_without_rs_arr[:, 2:])
        labeled_clusters_without_rs = list(zip(kmeans_first20_without_rs.labels_, start_sample_without_rs_arr))

        for clust in labeled_clusters_without_rs:
            self.ds.add((clust[0], int(clust[1][0])))
            self._update_summary_stats(clust, "ds")

        if len(self.rs) > self.n_cluster*5:
            print('runs before')
            start_sample_rs_rdd = start_sample_rdd.filter(lambda x: True if (-1, x[0]) in self.rs else False)
            start_sample_rs_arr = np.array(start_sample_rs_rdd.collect())

            kmeans_first20_rs = KMeans(self.n_cluster*5, random_state=random_seed).fit(start_sample_rs_arr[:, 2:])
            labeled_clusters_rs = list(zip(kmeans_first20_rs.labels_, start_sample_rs_arr))
            singleton_clusters_rs = sc.parallelize(labeled_clusters_rs).groupByKey().mapValues(list) \
                .filter(lambda x: True if len(x[1]) == 1 else False).collect()
            non_singleton_clusters_rs = sc.parallelize(labeled_clusters_rs).groupByKey().mapValues(list) \
                .filter(lambda x: True if len(x[1]) > 1 else False).flatMapValues(lambda x: x).collect()

            self.rs = set([])
            for clust in singleton_clusters_rs:
                self.rs.add((-1, int(clust[1][0][0])))

            for clust in non_singleton_clusters_rs:
                self.cs.add((clust[0], int(clust[1][0])))
                self._update_summary_stats(clust, "cs")

        cs_summary_stats_check = {key: self.summary_stats[key] for key in self.summary_stats.keys() if "cs" in key}
        round_result[1] = [len(self.ds), len(cs_summary_stats_check), len(self.cs), len(self.rs)]

        d = start_sample_arr[:, 2:].shape[1]
        threshold = 2*np.sqrt(d)
        for i, chunk in enumerate(chunks[1:]):
            ##### SUBSEQUENT ROUNDS #####
            start = perf_counter()
            for arr in chunk:
                ### DS CHECK ###
                ds_summary_stats = {key: self.summary_stats[key] for key in self.summary_stats.keys() if "ds" in key}
                dists_ds = [(self._compute_mahalanobis(arr[2:], clust_stats), clust_num, arr) \
                    for clust_num, clust_stats in ds_summary_stats.items()]
                min_dist_clust_ds = min(sorted(dists_ds, key=lambda x: x[0]))

                clust_check_ds = (min_dist_clust_ds[1][0], min_dist_clust_ds[2])
                if clust_check_ds[0] < threshold:
                    self.ds.add((clust_check_ds[0], int(clust_check_ds[1][0])))
                    self._update_summary_stats(clust_check_ds, "ds")
                else:
                    if self.cs:
                        ### CS CHECK ###
                        print('runs cs update')
                        cs_summary_stats = {key: self.summary_stats[key] for key in self.summary_stats.keys() if "cs" in key}
                        dists_cs = [(self._compute_mahalanobis(arr[2:], clust_stats), clust_num, arr) \
                            for clust_num, clust_stats in cs_summary_stats.items()]
                        min_dist_clust_cs = min(sorted(dists_cs, key=lambda x: x[0]))

                        clust_check_cs = (min_dist_clust_cs[1][0], min_dist_clust_cs[2])
                        if clust_check_cs[0] < threshold:
                            self.cs.add((clust_check_cs[0], int(clust_check_cs[1][0])))
                            self._update_summary_stats(clust_check_cs, "cs")
                        else:
                            ### RS CHECK ###
                            self.rs.add((-1, int(clust_check_cs[1][0])))
                    else:
                        ### RS CHECK ###
                        self.rs.add((-1, int(clust_check_ds[1][0])))

            ### RS KMEANS ###
            if len(self.rs) > self.n_cluster*5:
                rs_rdd = rdd.filter(lambda x: True if (-1, x[0]) in self.rs else False)
                rs_arr = np.array(rs_rdd.collect())

                kmeans_rs_next = KMeans(self.n_cluster*5, random_state=random_seed).fit(rs_arr[:, 2:])
                labeled_clusters_rs_next = list(zip(kmeans_rs_next.labels_, rs_arr))
                singleton_clusters_rs_next = sc.parallelize(labeled_clusters_rs_next).groupByKey().mapValues(list) \
                    .filter(lambda x: True if len(x[1]) == 1 else False).collect()
                non_singleton_clusters_rs_next = sc.parallelize(labeled_clusters_rs_next).groupByKey().mapValues(list) \
                    .filter(lambda x: True if len(x[1]) > 1 else False).flatMapValues(lambda x: x).collect()
                
                self.rs = set([])
                for clust in singleton_clusters_rs_next:
                    self.rs.add((-1, int(clust[1][0][0])))

                for clust in non_singleton_clusters_rs_next:
                    self.cs.add((clust[0], int(clust[1][0])))
                    self._update_summary_stats(clust, "cs")

            ### MERGE CS ###
            cs_summary_stats_merge = {key: self.summary_stats[key] for key in self.summary_stats.keys() if "cs" in key}
            cs_dists_merge = defaultdict(list)
            available = set(cs_summary_stats_merge.keys())
            for clust1, clust2 in list(combinations(cs_summary_stats_merge.items(), 2)):
                if clust1[0] in available and clust2[0] in available:
                    dist = self._compute_mahalanobis(clust1[1]["SUM"]/clust1[1]["N"], clust2[1])
                    cs_dists_merge[clust1[0]].append((clust2[0], dist))
                    available.remove(clust1[0])
                    available.remove(clust2[0])

            cs_dists_merge_dict = dict(cs_dists_merge)
            merge_clusts = {clust: min(combo_dists, key=lambda x: x[1]) for clust, combo_dists in cs_dists_merge_dict.items() \
                if min(combo_dists, key=lambda x: x[1])[1] < threshold}
            
            cs_copy = self.cs.copy()
            for clust_to_merge, merge_clust in merge_clusts.items():
                for point in cs_copy:
                    if int(clust_to_merge[0]) == int(point[0]):
                        self.cs.remove(point)
                        self.cs.add((merge_clust[0][0], point[1]))
                        clust = (merge_clust[0][0], np.array(data[point[1]]))
                        self._update_summary_stats(clust, "cs")
                del self.summary_stats[(clust_to_merge[0], "cs")]

            if i+1 == len(chunks)-1:
                ### MERGE CS INTO DS ###
                cs_summary_stats_merge_last = {key: self.summary_stats[key] for key in self.summary_stats.keys() if "cs" in key}
                ds_summary_stats_merge = {key: self.summary_stats[key] for key in self.summary_stats.keys() if "ds" in key}
                cs_dists_to_merge_to_ds = defaultdict(list)
                for cs_clust, cs_stats in cs_summary_stats_merge_last.items():
                    for ds_clust, ds_stats in ds_summary_stats_merge.items():
                        dist = self._compute_mahalanobis(cs_stats["SUM"]/cs_stats["N"], ds_stats)
                        cs_dists_to_merge_to_ds[cs_clust].append((ds_clust, dist))

                cs_dists_to_merge_to_ds_dict = dict(cs_dists_to_merge_to_ds)
                merge_clusts_cs_to_ds = {clust: min(combo_dists, key=lambda x: x[1]) \
                    for clust, combo_dists in cs_dists_to_merge_to_ds_dict.items() \
                        if min(combo_dists, key=lambda x: x[1])[1] < threshold}
                
                cs_copy = self.cs.copy()
                for clust_to_merge, merge_clust in merge_clusts_cs_to_ds.items():
                    for point in cs_copy:
                        if int(clust_to_merge[0]) == int(point[0]):
                            self.cs.remove(point)
                            self.ds.add((merge_clust[0][0], point[1]))
                            clust = (merge_clust[0][0], np.array(data[point[1]]))
                            self._update_summary_stats(clust, "ds")
                    del self.summary_stats[(clust_to_merge[0], "cs")]

                cs_copy = self.cs.copy()
                for point in cs_copy:
                    self.cs.remove(point)
                    self.cs.add((-1, point[1]))

            cs_summary_stats_check = {key: self.summary_stats[key] for key in self.summary_stats.keys() if "cs" in key}
            round_result[i+2] = [len(self.ds), len(cs_summary_stats_check), len(self.cs), len(self.rs)]
            end = perf_counter()
            print(f'Round {i+2} time: {end-start}')

        return round_result, self.ds, self.cs, self.rs

if __name__ == "__main__":
    start = perf_counter()

    input_file = sys.argv[1]
    n_cluster = int(sys.argv[2])
    output_file = sys.argv[3]

    sc = SparkContext('local[*]', 'task')
    sc.setLogLevel('ERROR')

    input_rdd = read_dataset(sc, input_file)
    data_rdd = convert_to_numeric(input_rdd)
    ds = set([])
    cs = set([])
    rs = set([])
    summary_stats = {}
    bfr = BFR(n_cluster, ds, cs, rs, summary_stats)
    # GOOD VALS: 0, 1, 2, 9, 10
    round_result, ds, cs, rs = bfr.fit(sc, data_rdd, 1)
    ordered_points = sorted(ds.union(cs).union(rs), key=lambda x: x[1])

    with open(output_file, 'w') as file:
        file.write('The intermediate results:\n')
        for round_num, stats in round_result.items():
            file.write(f'Round {round_num}: {stats[0]},{stats[1]},{stats[2]},{stats[3]}\n')
        file.write('\nThe clustering results:\n')
        for i, point in enumerate(ordered_points):
            if i == len(ordered_points)-1:
                file.write(f'{point[1]},{point[0]}')
            else:
                file.write(f'{point[1]},{point[0]}\n')
    
    end = perf_counter()
    print(f'Time elapsed: {end-start}')