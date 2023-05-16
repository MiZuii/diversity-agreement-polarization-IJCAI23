import numpy as np
import itertools
import mapel.elections.features.diversity as div
import matplotlib.pyplot as plt
import os


def restore_order(x):
    for i in range(len(x)):
        for j in range(len(x) - i, len(x)):
            if x[j] >= x[-i - 1]:
                x[j] += 1
    return x


def distances_to_rankings(rankings, distances):
    dists = distances[rankings]
    return np.sum(dists.min(axis=0))


def find_improvement(distances, d, starting, rest, n, k, l):
    for cut in itertools.combinations(range(k), l):
        # print(cut)
        for paste in itertools.combinations(rest, l):
            ranks = []
            j = 0
            for i in range(k):
                if i in cut:
                    ranks.append(paste[j])
                    j = j + 1
                else:
                    ranks.append(starting[i])
            # check if unique
            if len(set(ranks)) == len(ranks):
                # check if better
                d_new = distances_to_rankings(ranks, distances)
                if d > d_new:
                    return ranks, d_new, True
    return starting, d, False


def local_search_kKemeny_single_k(election, k, l, starting=None):
    if starting is None:
        starting = list(range(k))
    distances = div.calculate_vote_swap_dist(election)

    n = election.num_voters

    d = distances_to_rankings(starting, distances)
    iter = 0
    check = True
    while (check):
        # print(iter)
        # print(starting)
        # print(d)
        # print()
        iter = iter + 1
        rest = [i for i in range(n) if i not in starting]
        for j in range(l):
            starting, d, check = find_improvement(distances, d, starting, rest, n, k, j + 1)
            if check:
                break
        # print()
    return d


def local_search_kKemeny(election, l, starting=None):
    max_dist = election.num_candidates * (election.num_candidates - 1) / 2
    res = []
    for k in range(1, election.num_voters):
        # print(k)
        if starting is None:
            d = local_search_kKemeny_single_k(election, k, l)
        else:
            d = local_search_kKemeny_single_k(election, k, l, starting[:k])
        d = d
        if d > 0:
            res.append(d)
        else:
            break
    for k in range(len(res), election.num_voters):
        res.append(0)

    return res


def diversity_index(election):
    if election.fake:
        return 'None'
    max_dist = election.num_candidates * (election.num_candidates - 1) / 2
    res = [0] * election.num_voters
    chosen_votes = []
    distances = div.calculate_vote_swap_dist(election)
    best = np.argmin(distances.sum(axis=1))
    chosen_votes.append(best)
    best_vec = distances[best]
    res[0] = best_vec.sum()
    distances = np.vstack((distances[:best], distances[best + 1:]))

    for i in range(1, election.num_voters):
        relatives = distances - best_vec
        relatives = relatives * (relatives < 0)
        best = np.argmin(relatives.sum(axis=1))
        chosen_votes.append(best)
        best_vec = best_vec + relatives[best]
        res[i] = best_vec.sum()
        distances = np.vstack((distances[:best], distances[best + 1:]))

    chosen_votes = restore_order(chosen_votes)

    res = local_search_kKemeny(election, 1, chosen_votes)

    return res


def greedy_kKemenys_summed(election):
    if election.fake:
        return 'None'
    res = [0] * election.num_voters
    distances = div.calculate_vote_swap_dist(election)
    best = np.argmin(distances.sum(axis=1))
    best_vec = distances[best]
    res[0] = best_vec.sum()
    distances = np.vstack((distances[:best], distances[best + 1:]))

    for i in range(1, election.num_voters):
        relatives = distances - best_vec
        relatives = relatives * (relatives < 0)
        best = np.argmin(relatives.sum(axis=1))
        best_vec = best_vec + relatives[best]
        res[i] = best_vec.sum()
        distances = np.vstack((distances[:best], distances[best + 1:]))

    return res


def compare_k_kemeny_distances(experiment, saveas=None):
    res = []

    i = 0
    for election in experiment.instances.values():
        res.append([greedy_kKemenys_summed(election),
                    local_search_kKemeny(election, 1),
                    diversity_index(election)])
        i = i + 1
        print(str(round(100 * i / experiment.num_instances, 2)) + "%")

    res = np.array(res)

    if saveas is not None:
        if saveas == 'default':
            np.save('experiments/' + experiment.experiment_id + '/kKemenyDistances.npy', res)
        else:
            np.save('experiments/' + experiment.experiment_id + '/' + saveas + '.npy', res)

    return res

def analyze_k_kemeny_distances(experiment = None, data = None, load = None, saveas = None):
    if data is None:
        if load is None:
            print("No data given")
            return 0
        else:
            if load == 'default' and experiment is not None:
                data = np.load('experiments/' + experiment.experiment_id + '/kKemenyDistances.npy')
            else:
                data = np.load(load)

    algs = ['greedy approach', 'local search', 'combined heuristic']

    print(data.shape)
    print("Max difference: " + str((data.max(axis=1) - data.min(axis=1)).max()))
    if experiment is not None:
        print((data.max(axis=1) - data.min(axis=1)).shape)
        x = np.argmax((data.max(axis=1) - data.min(axis=1)).max(axis=1))
        print(x)
        print(list(experiment.instances.keys())[x])
    print("Mean difference: " + str((data.max(axis=1) - data.min(axis=1)).mean()))
    print("Min difference: " + str((data.max(axis=1) - data.min(axis=1)).min()))
    print("No. Zeros: " + str(((data.max(axis=1) - data.min(axis=1))==0).sum()))
    print()

    for i in range(2):
        for j in range(i+1, 3):
    # for i in [1]:
    #     for j in [2]:
            diff = data[:, i, :] - data[:, j, :]
    #         print(algs[j] + " vs. " + algs[i])
    #         print("Max advantage:" + str(diff.max()))
    #         print("Mean advantage:" + str(diff.mean()))
    #         print("Min advantage:" + str(diff.min()))
    #         print("Positives: " + str((diff > 0).sum()))
    #         print("Zeros: " + str((diff == 0).sum()))
    #         print("Negatives: " + str((diff < 0).sum()))
    #         print(':')
    #
    #         s = diff.shape[0]
    #         n = diff.shape[1]
    #         max_dist = data.max() / n * 2
    #
    #         # Diversity difference:
    #
    #         harmonic = np.array([1/(k+1) for k in range(n)])
    #         div_diff = (diff * harmonic / max_dist / n).sum(axis=1)
    #         print(div_diff.shape)
    #         print("_Diversity index_")
    #         print("Max advantage:" + str(div_diff.max()))
    #         print("Mean advantage:" + str(div_diff.mean()))
    #         print("Min advantage:" + str(div_diff.min()))
    #         print("Positives: " + str((div_diff > 0).sum()))
    #         print("Zeros: " + str((div_diff == 0).sum()))
    #         print("Negatives: " + str((div_diff < 0).sum()))
    #         print(':')
    #
    #         pol_diff = (diff[:, 0] - diff[:, 1]) / max_dist / n
    #         print(pol_diff.shape)
    #         print("_Polarization index_")
    #         print("Max advantage:" + str(pol_diff.max()))
    #         print("Mean advantage:" + str(pol_diff.mean()))
    #         print("Min advantage:" + str(pol_diff.min()))
    #         print("Positives: " + str((pol_diff > 0).sum()))
    #         print("Zeros: " + str((pol_diff == 0).sum()))
    #         print("Negatives: " + str((pol_diff < 0).sum()))
    #         print()

            #
            #
            diff = diff.reshape(diff.shape[0] * diff.shape[1])
            # min_bin = diff.min()
            # if min_bin > -10:
            #     bins_negative = list(range(int(min_bin), 0))
            # else:
            #     bins_negative = []
            #     for k in range(10):
            #         bins_negative.append(round(min_bin - k * min_bin / 10))
            # max_bin = diff.max()
            # if max_bin < 10:
            #     bins_positive = list(range(1, int(max_bin)+2))
            # else:
            #     bins_positive = []
            #     for k in range(11):
            #         bins_positive.append(1 + round(k * max_bin / 10))
            # bins = bins_negative + [0] + bins_positive
            # print(bins)
            plt.figure(figsize=(10,6))
            plt.hist(diff, bins=list(range(-75, 485, 10)), color='blue', edgecolor='black', alpha=0.5, log=True)
            plt.annotate("max: " + str(diff.max()), (250,9000), fontsize = 24)
            plt.annotate("mean: " + str(round(diff.mean(),4)), (250, 3000), fontsize = 24)
            plt.annotate("min: " + str(round(diff.min())), (250, 1000), fontsize = 24)
            plt.yticks(fontsize=24)
            plt.xticks(fontsize=24)
            # plt.title("Difference between " + algs[j] + " and " + algs[i])
            # # plt.violinplot(diff)
            # # plt.boxplot(diff)
            # # plt.axvline(diff.mean(), color='k', linestyle='dashed', linewidth=1)
            # # noise = np.random.random(len(diff))
            # plt.scatter(noise,diff,alpha=0.2)

            if saveas is not None:
                if saveas == 'default':
                    if experiment is not None:
                        import mapel.main.utils as ut
                        ut.make_folder_if_do_not_exist("images/kKemenyDistances")
                        file_name = os.path.join(os.getcwd(),
                                                 "images/kKemenyDistances",
                                                 experiment.experiment_id + "_" + algs[j] + "-" + algs[i] + ".png")
                    else:
                        print("No experiment given, default saving not valid!")
                else:
                    file_name = os.path.join(os.getcwd(), "images", str(saveas))
                plt.savefig(file_name, bbox_inches='tight', dpi=250)
                # plt.savefig(file_name[:-3] + 'pdf', bbox_inches='tight', format='pdf')
            plt.show()

    return 0

def print_diversity(experiment = None, data = None, load = None):
    if data is None:
        if load is None:
            print("No data given")
            return 0
        else:
            if load == 'default' and experiment is not None:
                data = np.load('experiments/' + experiment.experiment_id + '/kKemenyDistances.npy')
            else:
                data = np.load(load)

    n = data.shape[2]
    max_dist = data.max() / n * 2
    data = data.min(axis=1)
    harmonic = np.array([1/(k+1) for k in range(n)])
    div = (data * harmonic / max_dist / n).sum(axis=1)
    print("DIVERSITY")
    print(div.shape)
    for i in range(len(div)):
        print(div[i])

    return div

def print_polarization(experiment = None, data = None, load = None):
    if data is None:
        if load is None:
            print("No data given")
            return 0
        else:
            if load == 'default' and experiment is not None:
                data = np.load('experiments/' + experiment.experiment_id + '/kKemenyDistances.npy')
            else:
                data = np.load(load)

    n = data.shape[2]
    max_dist = data.max() / n * 2
    data = data.min(axis=1)
    pol = 2 * (data[:, 0] - data[:, 1]) / max_dist / n
    print("POLARIZATION")
    print(pol.shape)
    for i in range(len(pol)):
        print(pol[i])

    return pol
