import numpy as np

import mapel
import mapel.main.printing as meprint
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib import colors
import os
import math
from scipy.stats import stats

def feature_hist(experiment, features, scaled=True, bins=None):

    feature_values = []

    for feature_id in features:
        if feature_id in experiment.features:
            ftr = experiment.features[feature_id]['value'].values()
            feature_values.append(list(ftr))
        else:
            ftr = meprint.get_values_from_csv_file(experiment,
                                                   feature_id=feature_id,
                                                   column_id='value')
            ftr = ftr.values()
            feature_values.append(list(ftr))

    if scaled:
        feature_values_ = []
        for ftr in feature_values:
            print((min(ftr),max(ftr)))
            scale = 1 / (max(ftr) - min(ftr))
            subtract = min(ftr)
            feature_values_.append([(x - subtract) / scale for x in ftr])
        feature_values = feature_values_

    sum_value = []
    for i in range(len(feature_values[0])):
        x = 0
        for j in range(len(feature_values)):
            x += feature_values[j][i]
        sum_value.append(x)

    # if colors_in_bins is not None:
    #     bin_no = []
    #     for x in sum_value:
    #         i = 0
    #         for s, e in zip(bins[:-1],bins[1:]):
    #             if x >= s and x < e:
    #                 bin_no.append(i)
    #                 break
    #             i += 1
    #         else:
    #             bin_no.append(-1)

    if bins is None:
        plt.hist(sum_value)
    else:
        plt.hist(sum_value, bins=bins)
    plt.show()
    return 0

def feature_by_families(experiment, feature_id):
    result = {}

    if feature_id in experiment.features:
        feature = experiment.features[feature_id]['value']
    else:
        feature = meprint.get_values_from_csv_file(experiment, feature_id=feature_id, column_id='value')

    for family in experiment.families.values():
        result[family.family_id] = []
        for instance in family.instance_ids:
            result[family.family_id].append(feature[instance])

    return result

def distance_by_families(experiment, dist_from):
    result = {}

    for family in experiment.families.values():
        result[family.family_id] = []
        for instance in family.instance_ids:
            if instance not in experiment.distances[dist_from]:
                result[family.family_id].append(0)
            else:
                d = experiment.distances[dist_from][instance]
                result[family.family_id].append(d)

    return result

def distance_extraction(experiment, dist_from):
    res = {}
    for k in experiment.instances.keys():
        if k not in experiment.distances[dist_from]:
            # print(k)
            res[k] = 0
        else:
            res[k] = experiment.distances[dist_from][k]
    return res

def correlation_feature_distance(experiment, feature, dist_from, show=True):
    if feature in experiment.features:
        feature_val = experiment.features[feature]['value']
    else:
        feature_val = meprint.get_values_from_csv_file(experiment, feature_id=feature, column_id='value')

    # x = list(feature.values())
    # y = list(experiment.distances[dist_from].values())
    x = []
    y = []
    first = True
    for k in feature_val.keys():
        if k not in experiment.distances[dist_from]:
            # print(k)
            if first:
                x.append(feature_val[k])
                y.append(0)
                first = False
        else:
            x.append(feature_val[k])
            y.append(experiment.distances[dist_from][k])

    res = stats.pearsonr(x, y)[0]

    if show:
        print("Correlation between " + feature + " and the distance from " + dist_from + ": " + str(round(res, 5)))

    return res

def plot_feature_distance(experiment, feature, dist_from, show=True, saveas=None, shading=True):
# def print_map_by_features(experiment, feature1_id, feature2_id, saveas=None, shading=True, added_points=None):
    fig = plt.figure()
    ax = fig.add_subplot()

    feature_bf = feature_by_families(experiment, feature)
    distance_bf = distance_by_families(experiment, dist_from)
    if shading:
        alphas_by_families, tints_by_families = colors_by_families(experiment)
        for family in experiment.families.values():

            colorings = []
            if sum(tints_by_families[family.family_id])>0:
                alphas_and_tints = zip(alphas_by_families[family.family_id],
                                        tints_by_families[family.family_id])
                for a, t in alphas_and_tints:
                    # red = 0.3
                    # green = 0.8 - t
                    # blue = 0.3 + t
                    # colorings.append((red, green, blue, 1 - (1 - a)*0.8))
                    a = (a - 0.3) / 0.7
                    clr = (0.75 - 0.75 * a + 0.125 * t + 0.875 * a * t,
                           0.75 - 0.75 * a,
                           0.75 - 0.75 * a + 0.125 * (1 - t) + 0.875 * a * (1 - t))
                    colorings.append(clr)

                ax.scatter(feature_bf[family.family_id],
                           distance_bf[family.family_id],
                           color=colorings,
                           label=family.label,
                           alpha=1,
                           s=family.ms,
                           marker=family.marker)
            else:
                ax.scatter(feature_bf[family.family_id],
                           distance_bf[family.family_id],
                           color=family.color,
                           label=family.label,
                           alpha=alphas_by_families[family.family_id],
                           s=family.ms,
                           marker=family.marker)
    else:
        for family in experiment.families.values():
            ax.scatter(feature_bf[family.family_id],
                       distance_bf[family.family_id],
                       color=family.color,
                       label=family.label,
                       alpha=family.alpha,
                       s=family.ms,
                       marker=family.marker)

    # for point in added_points:
    #     if 'marker' not in point:
    #         point['marker'] = 'x'
    #     if 'color' not in point:
    #         point['color'] = 'red'
    #     if 'size' not in point:
    #         point['size'] = 20
    #     if 'alpha' not in point:
    #         point['alpha'] = 1
    #     if 'label' not in point:
    #         ax.scatter(point['x'],
    #                    point['y'],
    #                    color=point['color'],
    #                    alpha=point['alpha'],
    #                    s=point['size'],
    #                    marker=point['marker'])
    #     else:
    #         ax.scatter(point['x'],
    #                    point['y'],
    #                    color=point['color'],
    #                    alpha=point['alpha'],
    #                    s=point['size'],
    #                    marker=point['marker'],
    #                    label=point['label'])



    plt.xlabel(feature, fontsize=28)
    plt.ylabel("Distance from " + str(dist_from)[:2], fontsize=28)
    plt.yticks(fontsize=16)
    plt.xticks(fontsize=16)

    if saveas is not None:
        if saveas == 'default':
            import mapel.main.utils as ut
            ut.make_folder_if_do_not_exist("images/feature-maps")
            file_name = os.path.join(os.getcwd(),
                                     "images/feature-maps",
                                     experiment.experiment_id + "_" + feature + "-dfrom" + dist_from + ".png")
        else:
            file_name = os.path.join(os.getcwd(), "images", str(saveas))
        plt.savefig(file_name, bbox_inches='tight', dpi=250)
        plt.savefig(file_name[:-3] + 'pdf', bbox_inches='tight', format='pdf')

    plt.show()
    return 0


def colors_by_families(experiment):
    result = ({}, {})
    for family in experiment.families.values():
        alphas = []
        tints = []
        for i in range(family.size):
            election_id = list(family.instance_ids)[i]
            tint = 0
            try:
                alpha = experiment.instances[election_id].alpha
                if 'tint' in experiment.instances[election_id].params:
                    tint = 2 * experiment.instances[election_id].params['tint']
            except:
                try:
                    alpha = experiment.elections[election_id].alpha
                    if 'tint' in experiment.elections[election_id].params:
                        tint = 2 * experiment.elections[election_id].params['tint']
                except:
                    alpha = 1
                    tint = 0

            alphas.append(alpha)
            tints.append(tint)
        if min(alphas) < max(alphas):
            scale = 1 / max(alphas)
            alphas = [0.3 + 0.7 * a * scale for a in alphas]
        else:
            alphas = [family.alpha for _ in alphas]

        result[0][family.family_id] = alphas
        result[1][family.family_id] = tints
    return result

def print_map_by_features(experiment, feature1_id, feature2_id, saveas=None, shading=True, added_points=None):
    if added_points is None:
        added_points = []

    fig = plt.figure()
    ax = fig.add_subplot()

    feature1_by_families = feature_by_families(experiment, feature1_id)
    feature2_by_families = feature_by_families(experiment, feature2_id)
    if shading:
        alphas_by_families, tints_by_families = colors_by_families(experiment)
        for family in experiment.families.values():

            colorings = []
            if sum(tints_by_families[family.family_id])>0:
                alphas_and_tints = zip(alphas_by_families[family.family_id],
                                        tints_by_families[family.family_id])
                for a, t in alphas_and_tints:
                    # red = 0.3
                    # green = 0.8 - t
                    # blue = 0.3 + t
                    # colorings.append((red, green, blue, 1 - (1 - a)*0.8))
                    a = (a - 0.3) / 0.7
                    clr = (0.75 - 0.75 * a + 0.125 * t + 0.875 * a * t,
                           0.75 - 0.75 * a,
                           0.75 - 0.75 * a + 0.125 * (1 - t) + 0.875 * a * (1 - t))
                    colorings.append(clr)

                ax.scatter(feature1_by_families[family.family_id],
                           feature2_by_families[family.family_id],
                           color=colorings,
                           label=family.label,
                           alpha=1,
                           s=family.ms,
                           marker=family.marker)
            else:
                ax.scatter(feature1_by_families[family.family_id],
                           feature2_by_families[family.family_id],
                           color=family.color,
                           label=family.label,
                           alpha=alphas_by_families[family.family_id],
                           s=family.ms,
                           marker=family.marker)
    else:
        for family in experiment.families.values():
            ax.scatter(feature1_by_families[family.family_id],
                       feature2_by_families[family.family_id],
                       color=family.color,
                       label=family.label,
                       alpha=family.alpha,
                       s=family.ms,
                       marker=family.marker)

    for point in added_points:
        if 'marker' not in point:
            point['marker'] = 'x'
        if 'color' not in point:
            point['color'] = 'red'
        if 'size' not in point:
            point['size'] = 20
        if 'alpha' not in point:
            point['alpha'] = 1
        if 'label' not in point:
            ax.scatter(point['x'],
                       point['y'],
                       color=point['color'],
                       alpha=point['alpha'],
                       s=point['size'],
                       marker=point['marker'])
        else:
            ax.scatter(point['x'],
                       point['y'],
                       color=point['color'],
                       alpha=point['alpha'],
                       s=point['size'],
                       marker=point['marker'],
                       label=point['label'])



    plt.xlabel(feature1_id, fontsize=18)
    plt.ylabel(feature2_id, fontsize=18)
    plt.yticks(fontsize=16)
    plt.xticks(fontsize=16)

    if saveas is not None:
        if saveas == 'default':
            import mapel.main.utils as ut
            ut.make_folder_if_do_not_exist("images/feature-maps")
            file_name = os.path.join(os.getcwd(),
                                     "images/feature-maps",
                                     experiment.experiment_id + "_" + feature1_id + "-" + feature2_id + ".png")
        else:
            file_name = os.path.join(os.getcwd(), "images", str(saveas))
        plt.savefig(file_name, bbox_inches='tight', dpi=250)
        plt.savefig(file_name[:-3] + 'pdf', bbox_inches='tight', format='pdf')

    plt.show()
    return 0


def rotate_point(cx, cy, angle, px, py) -> (float, float):
    """ Rotate two-dimensional point by an angle """
    s, c = math.sin(angle), math.cos(angle)
    px -= cx
    py -= cy
    x_new, y_new = px * c - py * s, px * s + py * c
    px, py = x_new + cx, y_new + cy

    return px, py

def rotate_points(xs, ys, angle, cx, cy):
    new_xx = []
    new_yy = []
    for x, y in zip(xs,ys):
        x, y = rotate_point(cx, cy, angle, x, y)
        new_xx.append(x)
        new_yy.append(y)
    return new_xx, new_yy


def rotate_map(experiment, angle, point):
    if type(point) is str:
        point = experiment.coordinates[point]
    x = point[0]
    y = point[1]

    xx = []
    yy = []
    for p in experiment.coordinates.values():
        xx.append(p[0])
        yy.append(p[1])

    xx, yy = rotate_points(xx, yy, angle, x, y)

    ks = list(experiment.coordinates.keys())
    for i in range(len(ks)):
        experiment.coordinates[ks[i]] = [xx[i], yy[i]]

    return experiment

def upside_down_map(experiment):
    ks = list(experiment.coordinates.keys())
    for i in range(len(ks)):
        p = experiment.coordinates[ks[i]]
        experiment.coordinates[ks[i]] = [p[0], -p[1]]

    return experiment


def print_triangle_map(experiment, feature1_id, feature2_id, mode, saveas=None, shading=True, added_points=None):
    if added_points is None:
        added_points = []

    fig = plt.figure()
    ax = fig.add_subplot()

    feature1_by_families = feature_by_families(experiment, feature1_id)
    xx = []
    for k in feature1_by_families.keys():
        xx += feature1_by_families[k]
    feature2_by_families = feature_by_families(experiment, feature2_id)
    yy = []
    for k in feature2_by_families.keys():
        yy += feature2_by_families[k]
    if shading:
        alphas_by_families, tints_by_families = colors_by_families(experiment)

    for point in added_points:
        xx.append(point['x'])
        yy.append(point['y'])


    if mode=='agr-div':
        xx = np.array(xx)
        yy = np.array(yy)
        an_y = np.min(yy[xx == np.min(xx)]) # y position of antagonism
        yy = [y - an_y * (1 - x) for x, y in zip(xx, yy)] #bringing left part down

        a = max(xx) - min(xx)
        h = 0.86602540378 * a
        # height of equilateral triangle
        scale = h / np.max(yy) # h / y position of UN
        yy = [y * scale for y in yy] # scaling the height

        un = np.argmax(yy)
        mid = (max(xx) + min(xx))/2
        trans = mid - xx[un]
        xx = [x + trans * y / h for x, y in zip(xx, yy)] # moving un to the center

        xx, yy = rotate_points(xx, yy, math.pi / 3, max(xx), min(yy)) # rotating the map
        yy = [-y for y in yy] # flip
    elif mode=='agr-pol':
        xx, yy = rotate_points(xx, yy, - math.pi * 0.75, 1, 0)

        a = max(xx) - min(xx)
        h = 0.86602540378 * a
        # height of equilateral triangle
        scale = h / np.max(yy) # h / y position of UN
        yy = [y * scale for y in yy] # scaling the height
        xx = [-x for x in xx]  # flip

        print(max(yy))
        print(min(yy))
        un = np.argmax(yy)
        mid = (max(xx) + min(xx))/2
        trans = mid - xx[un]
        xx = [x + trans * y / h for x, y in zip(xx, yy)] # moving un to the center

        xx, yy = rotate_points(xx, yy, math.pi / 3, max(xx), min(yy)) # rotating the map
        yy = [-y for y in yy] # flip
    elif mode=='div-pol':
        xx = np.array(xx)
        yy = np.array(yy)
        an_x = np.min(xx[yy == np.max(yy)]) # x position of antagonism
        xx = [x - an_x * y for x, y in zip(xx, yy)] #bringing top part left
        xx, yy = rotate_points(xx, yy, math.pi * 0.5, 0, 0)

        a = max(xx) - min(xx)
        h = 0.86602540378 * a
        # height of equilateral triangle
        scale = h / np.max(yy) # h / y position of UN
        yy = [y * scale for y in yy] # scaling the height

        un = np.argmax(yy)
        mid = (max(xx) + min(xx))/2
        trans = mid - xx[un]
        xx = [x + trans * y / h for x, y in zip(xx, yy)] # moving un to the center

        xx, yy = rotate_points(xx, yy, math.pi / 3, max(xx), min(yy)) # rotating the map
        yy = [-y for y in yy] # flip

    s = 0
    for k in feature1_by_families.keys():
        l = len(feature1_by_families[k])
        # print(l)
        feature1_by_families[k] = xx[s:s + l]
        feature2_by_families[k] = yy[s:s + l]
        # print(len(feature2_by_families[k]))
        s += l

    for i in range(len(added_points)):
        added_points[i]['x'] = xx[s + i]
        added_points[i]['y'] = yy[s + i]


    if shading:
        alphas_by_families, tints_by_families = colors_by_families(experiment)
        for family in experiment.families.values():

            colorings = []
            if sum(tints_by_families[family.family_id])>0:
                alphas_and_tints = zip(alphas_by_families[family.family_id],
                                        tints_by_families[family.family_id])
                for a, t in alphas_and_tints:
                    # red = 0.3
                    # green = 0.8 - t
                    # blue = 0.3 + t
                    # colorings.append((red, green, blue, 1 - (1 - a)*0.8))
                    a = (a - 0.3) / 0.7
                    clr = (0.75 - 0.75 * a + 0.125 * t + 0.875 * a * t,
                           0.75 - 0.75 * a,
                           0.75 - 0.75 * a + 0.125 * (1 - t) + 0.875 * a * (1 - t))
                    colorings.append(clr)

                ax.scatter(feature1_by_families[family.family_id],
                           feature2_by_families[family.family_id],
                           color=colorings,
                           label=family.label,
                           alpha=1,
                           s=family.ms,
                           marker=family.marker)
            else:
                ax.scatter(feature1_by_families[family.family_id],
                           feature2_by_families[family.family_id],
                           color=family.color,
                           label=family.label,
                           alpha=alphas_by_families[family.family_id],
                           s=family.ms,
                           marker=family.marker)
    else:
        for family in experiment.families.values():
            ax.scatter(feature1_by_families[family.family_id],
                       feature2_by_families[family.family_id],
                       color=family.color,
                       label=family.label,
                       alpha=family.alpha,
                       s=family.ms,
                       marker=family.marker)

    # if shading:
    #     for family in experiment.families.values():
    #
    #         colorings = []
    #         if sum(tints_by_families[family.family_id]) > 0:
    #             alphas_and_tints = zip(alphas_by_families[family.family_id],
    #                                    tints_by_families[family.family_id])
    #             for a, t in alphas_and_tints:
    #                 red = 0.3
    #                 green = 0.8 - t
    #                 blue = 0.3 + t
    #                 colorings.append((red, green, blue, 1 - (1 - a) * 0.8))
    #
    #             ax.scatter(feature1_by_families[family.family_id],
    #                        feature2_by_families[family.family_id],
    #                        color=colorings,
    #                        label=family.label,
    #                        # alpha=family.alpha,
    #                        s=family.ms,
    #                        marker=family.marker)
    #         else:
    #             ax.scatter(feature1_by_families[family.family_id],
    #                        feature2_by_families[family.family_id],
    #                        color=family.color,
    #                        label=family.label,
    #                        alpha=alphas_by_families[family.family_id],
    #                        s=family.ms,
    #                        marker=family.marker)
    # else:
    #     for family in experiment.families.values():
    #         ax.scatter(feature1_by_families[family.family_id],
    #                    feature2_by_families[family.family_id],
    #                    color=family.color,
    #                    label=family.label,
    #                    alpha=family.alpha,
    #                    s=family.ms,
    #                    marker=family.marker)

    for point in added_points:
        if 'marker' not in point:
            point['marker'] = 'x'
        if 'color' not in point:
            point['color'] = 'red'
        if 'size' not in point:
            point['size'] = 20
        if 'alpha' not in point:
            point['alpha'] = 1
        ax.scatter(point['x'],
                   point['y'],
                   color=point['color'],
                   alpha=point['alpha'],
                   s=point['size'],
                   marker=point['marker'])

    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    if saveas is not None:
        if saveas == 'default':
            import mapel.main.utils as ut
            ut.make_folder_if_do_not_exist("images/feature-maps")
            file_name = os.path.join(os.getcwd(),
                                     "images/feature-maps",
                                     experiment.experiment_id + "_" + feature1_id + "-" + feature2_id + "-triangle_" + mode + ".png")
        else:
            file_name = os.path.join(os.getcwd(), "images", str(saveas))
        plt.savefig(file_name, bbox_inches='tight', dpi=250)
        plt.savefig(file_name[:-3] + 'pdf', bbox_inches='tight', format='pdf')

    plt.show()
    return 0

# def color_reverser(x, y, z):
#     x = 1 - x
#     y = 1 - y
#     z = 1 - z
#     a = (y + z - x + 1)/3
#     b = (z + x - y + 1)/3
#     c = (x + y - z + 1)/3
#     return (a, b, c)

def color_reverser(x, y, z):
    s = max([x, y/0.6, z])
    if s < 0.9:
        s = 0.9 - s
    else:
        s = 0
    return (x + s, y + s, z + s)

# def color_reverser(x, y, z):
#     s = max([x, y/0.6, z])
#     return (x * s, y * s, z * s)

def one_color_reverser(i, value):
    res = [0.95 - value * 0.95, 0.95 - value * 0.95, 0.95 - value * 0.95]
    if i == 1:
        res[i] = 0.9 - 0.3 * value
    else:
        res[i] = 0.9 + 0.05 * value
    return tuple(res)

def print_map_triple_feature(experiment, feature1_id, feature2_id, feature3_id,
                             saveas = None, reverse_colors = False, maxes=None):
    fig = plt.figure()
    ax = fig.add_subplot()

    if feature1_id in experiment.features:
        ftr1 = experiment.features[feature1_id]['value']
    else:
        ftr1 = meprint.get_values_from_csv_file(experiment, feature_id=feature1_id, column_id='value')
    if feature2_id in experiment.features:
        ftr2 = experiment.features[feature2_id]['value']
    else:
        ftr2 = meprint.get_values_from_csv_file(experiment, feature_id=feature2_id, column_id='value')
    if feature3_id in experiment.features:
        ftr3 = experiment.features[feature3_id]['value']
    else:
        ftr3 = meprint.get_values_from_csv_file(experiment, feature_id=feature3_id, column_id='value')

    if maxes is None:
        scale_ftr1 = 1 / max(list(ftr1.values()))
        print("Max feature1: " + str(max(list(ftr1.values()))))
        # scale_ftr2 = 0.5019607843137255 / max(list(ftr2.values())) # try colors.to_rgba('green')
        scale_ftr2 = 0.6 / max(list(ftr2.values()))
        print("Max feature2: " + str(max(list(ftr2.values()))))
        scale_ftr3 = 1 / max(list(ftr3.values()))
        print("Max feature3: " + str(max(list(ftr3.values()))))
    else:
        scale_ftr1 = 1 / maxes[0]
        scale_ftr2 = 0.6 / maxes[1]
        scale_ftr3 = 1 / maxes[2]


    for family in experiment.families.values():
        xx = []
        yy = []
        colorings = []
        for instance in family.instance_ids:
            if reverse_colors:
                clr = color_reverser(ftr1[instance] * scale_ftr1,
                                     ftr2[instance] * scale_ftr2,
                                     ftr3[instance] * scale_ftr3)
            else:
                clr = (ftr1[instance] * scale_ftr1,
                       ftr2[instance] * scale_ftr2,
                       ftr3[instance] * scale_ftr3)
            colorings.append(clr)
            xx.append(experiment.coordinates[instance][0])
            yy.append(experiment.coordinates[instance][1])

        ax.scatter(xx,
                   yy,
                   color=colorings,
                   alpha=1,
                   marker=family.marker)

    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    if saveas is not None:
        if saveas == 'default':
            file_name = os.path.join(os.getcwd(),
                                     "images/", experiment.experiment_id + "_triple_feature.png")
        else:
            file_name = os.path.join(os.getcwd(), "images", str(saveas))
        plt.savefig(file_name, bbox_inches='tight', dpi=250)
        plt.savefig(file_name[:-3] + 'pdf', bbox_inches='tight', format='pdf')

    plt.show()
    return 0

def print_map_sum_of_features(experiment, features, saveas = None, reverse_colors = False, maxes = None, scales = None):
    if scales is None:
        scales = [1] * len(features)

    fig = plt.figure()
    ax = fig.add_subplot()

    feature_values = []
    for ftr in features:
        if ftr in experiment.features:
            ftr_value = experiment.features[ftr]['value']
        else:
            ftr_value = meprint.get_values_from_csv_file(experiment, feature_id=ftr, column_id='value')
        feature_values.append(ftr_value)

    feature_sums = {}
    for family in experiment.families.values():
        for instance in family.instance_ids:
            val = 0
            for i, ftr in enumerate(feature_values):
                val = val + ftr[instance] * scales[i]
            # #double Polarization
            # val = val + feature_values[0][instance]
            feature_sums[instance] = val

    if maxes is None:
        move = min(list(feature_sums.values()))
        scale = 1 / (max(list(feature_sums.values())) - min(list(feature_sums.values())))
        print("Max sum of features: " + str(max(list(feature_sums.values()))))
        print("Min sum of features: " + str(min(list(feature_sums.values()))))
    else:
        scale = 1 / (maxes - 1)
        move = 1

    for family in experiment.families.values():
        xx = []
        yy = []
        colorings = []
        for instance in family.instance_ids:
            if reverse_colors:
                clr = (0.9 - (feature_sums[instance] - move) * scale * 0.9,
                       0.9 - (feature_sums[instance] - move) * scale * 0.9,
                       0.9 - (feature_sums[instance] - move) * scale * 0.9)
            else:
                clr = (feature_sums[instance] * scale,
                       feature_sums[instance] * scale,
                       feature_sums[instance] * scale)
            colorings.append(clr)
            xx.append(experiment.coordinates[instance][0])
            yy.append(experiment.coordinates[instance][1])

        ax.scatter(xx,
                   yy,
                   color=colorings,
                   alpha=1,
                   marker=family.marker)

    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    if saveas is not None:
        if saveas == 'default':
            file_name = os.path.join(os.getcwd(),
                                     "images/", experiment.experiment_id + "_sum_features.png")
        else:
            file_name = os.path.join(os.getcwd(), "images", str(saveas))
        plt.savefig(file_name, bbox_inches='tight', dpi=250)
        plt.savefig(file_name[:-3] + 'pdf', bbox_inches='tight', format='pdf')

    plt.show()
    return 0


def print_map_sum_of_dists(experiment, dist_froms, saveas = None, reverse_colors = False, maxes = None, scales = None):
    if scales is None:
        scales = [1] * len(dist_froms)

    fig = plt.figure()
    ax = fig.add_subplot()

    dist_values = []
    for df in dist_froms:
        df_values = distance_extraction(experiment, df)
        max_df = max(df_values.values())
        for k in df_values.keys():
            df_values[k] = 1 - (df_values[k] / max_df)
        dist_values.append(df_values)

    dist_sums = {}
    for family in experiment.families.values():
        for instance in family.instance_ids:
            val = 0
            for i, ftr in enumerate(dist_values):
                val = val + ftr[instance] * scales[i]
            # #double Polarization
            # val = val + feature_values[0][instance]
            dist_sums[instance] = val

    if maxes is None:
        move = min(list(dist_sums.values()))
        scale = 1 / (max(list(dist_sums.values())) - min(list(dist_sums.values())))
        print("Max sum of features: " + str(max(list(dist_sums.values()))))
        print("Min sum of features: " + str(min(list(dist_sums.values()))))
    else:
        scale = 1 / (maxes - 1)
        move = 1

    for family in experiment.families.values():
        xx = []
        yy = []
        colorings = []
        for instance in family.instance_ids:
            if reverse_colors:
                clr = (0.9 - (dist_sums[instance] - move) * scale * 0.9,
                       0.9 - (dist_sums[instance] - move) * scale * 0.9,
                       0.9 - (dist_sums[instance] - move) * scale * 0.9)
            else:
                clr = (dist_sums[instance] * scale,
                       dist_sums[instance] * scale,
                       dist_sums[instance] * scale)
            colorings.append(clr)
            xx.append(experiment.coordinates[instance][0])
            yy.append(experiment.coordinates[instance][1])

        ax.scatter(xx,
                   yy,
                   color=colorings,
                   alpha=1,
                   marker=family.marker)

    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    if saveas is not None:
        if saveas == 'default':
            file_name = os.path.join(os.getcwd(),
                                     "images/", experiment.experiment_id + "_sum_dist_froms.png")
        else:
            file_name = os.path.join(os.getcwd(), "images", str(saveas))
        plt.savefig(file_name, bbox_inches='tight', dpi=250)
        plt.savefig(file_name[:-3] + 'pdf', bbox_inches='tight', format='pdf')

    plt.show()
    return 0


# POSSIBLY NEEDS UPDATING
# def print_map_double_feature(experiment, feature1_id, feature2_id, saveas = None, reverse_colors = False):
#     fig = plt.figure()
#     ax = fig.add_subplot()
#
#     if feature1_id in experiment.features:
#         ftr1 = experiment.features[feature1_id]['value']
#     else:
#         ftr1 = meprint.get_values_from_csv_file(experiment, feature_id=feature1_id, column_id='value')
#     if feature2_id in experiment.features:
#         ftr2 = experiment.features[feature2_id]['value']
#     else:
#         ftr2 = meprint.get_values_from_csv_file(experiment, feature_id=feature2_id, column_id='value')
#
#     scale_ftr1 = max(list(ftr1.values()))
#     scale_ftr2 = max(list(ftr2.values()))
#
#
#     for family in experiment.families.values():
#         xx = []
#         yy = []
#         colorings = []
#
#         for instance in family.instance_ids:
#             if reverse_colors:
#                 clr = color_reverser(ftr1[instance]/scale_ftr1,
#                                      ftr2[instance]/scale_ftr2,
#                                      0)
#             else:
#                 clr = (ftr1[instance]/scale_ftr1, ftr2[instance]/scale_ftr2, 0)
#             colorings.append(clr)
#             xx.append(experiment.coordinates[instance][0])
#             yy.append(experiment.coordinates[instance][1])
#
#         ax.scatter(xx,
#                    yy,
#                    color=colorings,
#                    alpha=0.8,
#                    marker=family.marker)
#
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#     ax.spines['bottom'].set_visible(False)
#     ax.spines['left'].set_visible(False)
#
#     if saveas is not None:
#         if saveas == 'default':
#             file_name = os.path.join(os.getcwd(),
#                                      "images/", experiment.experiment_id + "_" + feature1_id + "-" + feature2_id + ".png")
#         else:
#             file_name = os.path.join(os.getcwd(), "images", str(saveas))
#         plt.savefig(file_name, bbox_inches='tight', dpi=250)
#         plt.savefig(file_name[:-3] + 'pdf', bbox_inches='tight', format='pdf')
#
#     plt.show()
#     return 0

def print_map_color_feature(experiment, feature_id, color, saveas = None,
                            reverse_colors = False, alpha_not_col = False, i = 0,
                            maxes = None):
    c = colors.to_rgba(color)
    fig = plt.figure()
    ax = fig.add_subplot()

    if feature_id in experiment.features:
        ftr = experiment.features[feature_id]['value']
    else:
        ftr = meprint.get_values_from_csv_file(experiment, feature_id=feature_id, column_id='value')

    if maxes is None:
        s = max(list(ftr.values()))
    else:
        s = maxes

    for family in experiment.families.values():
        xx = []
        yy = []
        colorings = []
        for instance in family.instance_ids:
            f = ftr[instance]
            if reverse_colors:
                clr = one_color_reverser(i, f/s)
            else:
                clr = (c[0]*f/s, c[1]*f/s, c[2]*f/s)
            if alpha_not_col:
                colorings.append((c[0], c[1], c[2], f/s*0.8 + 0.2))
            else:
                colorings.append((clr[0], clr[1], clr[2], c[3]*f/s))
            xx.append(experiment.coordinates[instance][0])
            yy.append(experiment.coordinates[instance][1])

        ax.scatter(xx,
                   yy,
                   color=colorings,
                   alpha=1,
                   marker=family.marker)

    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    if saveas is not None:
        if saveas == 'default':
            if color[0] == '#':
                color = color[1:]
            file_name = os.path.join(os.getcwd(),
                                     "images/", experiment.experiment_id + "_" + feature_id + "_as_" + color + ".png")
        else:
            file_name = os.path.join(os.getcwd(), "images", str(saveas))
        plt.savefig(file_name, bbox_inches='tight', dpi=250)
        plt.savefig(file_name[:-3] + 'pdf', bbox_inches='tight', format='pdf')

    plt.show()
    return 0

def print_map_with_chosen_votes(election, chosen_votes, numerics=False, show=True, radius=None, name=None, alpha=0.1, s=30, circles=False,
              object_type=None, double_gradient=False, saveas=None):

    if object_type is None:
        object_type = election.object_type

    plt.figure(figsize=(6.4, 6.4))

    X = []
    Y = []
    for elem in election.coordinates[object_type]:
        X.append(elem[0])
        Y.append(elem[1])

    if circles:
        weighted_points = {}
        Xs = {}
        Ys = {}
        for i in range(election.num_voters):
            str_elem = str(election.votes[i])
            if str_elem in weighted_points:
                weighted_points[str_elem] += 1
            else:
                weighted_points[str_elem] = 1
                Xs[str_elem] = X[i]
                Ys[str_elem] = Y[i]

        for str_elem in weighted_points:
            if weighted_points[str_elem] > election.num_voters * 0.03:
                plt.scatter(Xs[str_elem], Ys[str_elem],
                            color='purple',
                            s=10000 * weighted_points[str_elem] / election.num_voters,
                            alpha=0.2)

    if double_gradient:
        for i in range(election.num_voters):
            x = float(election.points['voters'][i][0])
            y = float(election.points['voters'][i][1])
            plt.scatter(X[i], Y[i], color=[0,y,x], s=s, alpha=alpha)
    else:
        plt.scatter(X, Y, color='blue', s=s, alpha=alpha)

    X_chosen = []
    Y_chosen = []
    for i in chosen_votes:
        X_chosen.append(X[i])
        Y_chosen.append(Y[i])
    if numerics:
        plt.scatter(X_chosen, Y_chosen, color='black', s=s, alpha=0.5, marker='s')
        for i in range(len(chosen_votes)):
            plt.annotate(str(i+1), (X_chosen[i] - 0.5, Y_chosen[i] - 0.5), color='lightcoral')
    else:
        plt.scatter(X_chosen, Y_chosen, color='red', s=s*0.25, alpha=1, marker='x')


    if radius:
        plt.xlim([-radius, radius])
        plt.ylim([-radius, radius])
    plt.title(election.label, size=38)
    plt.axis('off')

    if saveas is None:
        saveas = f'{election.label}_euc'

    file_name = os.path.join(os.getcwd(), "images", name, f'{saveas}.png')
    plt.savefig(file_name, bbox_inches='tight', dpi=100)
    # plt.savefig(file_name[:-3] + 'pdf', bbox_inches='tight', format='pdf')
    if show:
        plt.show()
    else:
        plt.clf()
