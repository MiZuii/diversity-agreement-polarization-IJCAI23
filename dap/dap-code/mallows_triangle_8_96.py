
# USE LOCAL MAPEL
import sys
sys.path.append("..")
import mapel
import printing_plus as pp
import kKemenyDistances as kkd
import math

name = 'mallows_triangle_8_96'
experiment_id = f'{name}'
instance_type = 'ordinal'
distance_id = 'swap'
embedding_id = 'mds'

agreement = 'Agreement'
diversity = 'Diversity'
polarization = 'Polarization'

experiment = mapel.prepare_experiment(experiment_id=experiment_id,
                                      instance_type=instance_type,
                                      distance_id=distance_id,
                                      embedding_id=embedding_id)

# # EXPERIMENT INITIALIZATION
# experiment.prepare_elections(printing=True)
# print("Elections prepared")
# experiment.compute_distances(distance_id = distance_id)
# print("Distances computed")
# experiment.embed(algorithm='mds')
# print("Coordinates computed")


# # MAP ORIENTATION
experiment = pp.rotate_map(experiment, math.pi * 0.9, 'ID')
experiment = pp.upside_down_map(experiment)


# # STANDARD MAP
experiment.print_map(legend=False,
                     legend_pos=[0.95, 1.15],
                     shading=True,
                     title=f'{name} ({embedding_id})',
                     figsize=(9.4, 6.4),
                     # textual=['ID', 'UN', 'AN', 'ST'],
                     saveas="diversity_map_8_96",
                     urn_orangered=False,
                     tex=True,
                     )


# # # k-KEMENY DISTANCES
# x = kkd.compare_k_kemeny_distances(experiment, saveas='default')
kkd.analyze_k_kemeny_distances(experiment, load='default', saveas='default')


# # INDICES COMPUTATION
# for ftr in [agreement, diversity, polarization]:
# for ftr in [polarization]:
#     print("Computing " + ftr)
#     experiment.compute_feature(feature_id=ftr)
#     experiment.print_map(feature_id=ftr,
#                          legend=False,
#                          shading=True,
#                          title=ftr,
#                          # textual=['ID', 'UN', 'AN', 'ST'],
#                          saveas="diversity_map_" + ftr)


# # INDICES VALUES ON THE MAP
pp.print_map_color_feature(experiment,
                           agreement,
                           '#0A0',
                           saveas='default',
                           reverse_colors=True,
                           i=1)
pp.print_map_color_feature(experiment,
                           diversity,
                           'blue',
                           saveas='default',
                           reverse_colors=True,
                           i=2,
                           maxes=1.307112342)
pp.print_map_color_feature(experiment,
                           polarization,
                           'red',
                           saveas='default',
                           reverse_colors=True,
                           i=0)
pp.print_map_triple_feature(experiment,
                            polarization,
                            agreement,
                            diversity,
                            saveas='default',
                            reverse_colors=True,
                            maxes=[1, 1, 1.307112342])
pp.print_map_sum_of_features(experiment,
                            [polarization,
                            agreement,
                            diversity],
                            saveas='default',
                            reverse_colors=True,
                            maxes=1.6046138196190476)


# # TRIANGLE MAPS
pp.print_map_by_features(experiment, agreement, diversity, saveas='default')
pp.print_triangle_map(experiment, agreement, diversity, mode='agr-div', saveas='default')
pp.print_map_by_features(experiment, agreement, polarization, saveas='default')
pp.print_triangle_map(experiment, agreement, polarization, mode='agr-pol', saveas='default')
pp.print_map_by_features(experiment, diversity, polarization, saveas='default')
pp.print_triangle_map(experiment, diversity, polarization, mode='div-pol', saveas='default')


# # CORRELATIONS TO DISTANCES
pp.correlation_feature_distance(experiment, agreement, 'ID')
pp.plot_feature_distance(experiment, agreement, 'ID', saveas='default')
pp.correlation_feature_distance(experiment, polarization, 'AN')
pp.plot_feature_distance(experiment, polarization, 'AN', saveas='default')
pp.correlation_feature_distance(experiment, diversity, 'UN')
pp.plot_feature_distance(experiment, diversity, 'UN', saveas='default')











#
# #
# # experiment.print_map(legend=True,
# #                      legend_pos=[0.95, 1.15],
# #                      shading=True,
# #                      title=f'{name} ({embedding_id})',
# #                      figsize=(9.4, 6.4),
# #                      saveas="mallows_triangle_legend"
# #                      )
# # experiment.print_map(legend=False,
# #                      legend_pos=[0.95, 1.15],
# #                      shading=True,
# #                      title=f'{name} ({embedding_id})',
# #                      figsize=(9.4, 6.4),
# #                      # textual=['ID', 'UN', 'AN', 'ST'],
# #                      saveas="_mallows"
# #                      )
# # #
# # # # Computing k-Kemeny distances
# # #
# # # # x = kkd.compare_k_kemeny_distances(experiment, saveas='default')
# kkd.analyze_k_kemeny_distances(experiment, load='default', saveas='default')
# #
# # #
# # # # Computing Indices
# # # #
# # # # # for ftr in [agreement, diversity, polarization]:
# # # # #     print("Computing " + ftr)
# # # # #     experiment.compute_feature(feature_id=ftr)
# # # # #     experiment.print_map(feature_id=ftr,
# # # # #                          legend=False,
# # # # #                          shading=True,
# # # # #                          title=ftr,
# # # # #                          # textual=['ID', 'UN', 'AN', 'ST'],
# # # # #                          saveas="diversity_map_" + ftr
# # # # #                          )
# # # #
# # #
# # # Color Maps
# #
# # pp.print_map_color_feature(experiment,
# #                            agreement,
# #                            '#0A0',
# #                            saveas='default',
# #                            reverse_colors=True,
# #                            i=1)
# # pp.print_map_color_feature(experiment,
# #                            diversity,
# #                            'blue',
# #                            saveas='default',
# #                            reverse_colors=True,
# #                            i=2,
# #                            maxes=1.307112342)
# # pp.print_map_color_feature(experiment,
# #                            polarization,
# #                            'red',
# #                            saveas='default',
# #                            reverse_colors=True,
# #                            i=0)
# # pp.print_map_triple_feature(experiment,
# #                             polarization,
# #                             agreement,
# #                             diversity,
# #                             saveas='default',
# #                             reverse_colors=True,
# #                             maxes=[1, 1, 1.307112342])
# # pp.print_map_sum_of_features(experiment,
# #                             [polarization,
# #                             agreement,
# #                             diversity],
# #                             saveas='default',
# #                             reverse_colors=True,
# #                             maxes = 1.6046138196190476)
# #
# #
# # # Triangle maps
# #
# # pp.print_map_by_features(experiment, agreement, diversity, saveas='default')
# # pp.print_triangle_map(experiment, agreement, diversity, mode='agr-div', saveas='default')
# # pp.print_map_by_features(experiment, agreement, polarization, saveas='default')
# # pp.print_triangle_map(experiment, agreement, polarization, mode='agr-pol', saveas='default')
# # pp.print_map_by_features(experiment, diversity, polarization, saveas='default')
# # pp.print_triangle_map(experiment, diversity, polarization, mode='div-pol', saveas='default')
#
# # # Correlation between distances and indices
# #
# pp.correlation_feature_distance(experiment, agreement, 'ID')
# pp.plot_feature_distance(experiment, agreement, 'ID', saveas='default')
# pp.correlation_feature_distance(experiment, polarization, 'AN')
# pp.plot_feature_distance(experiment, polarization, 'AN', saveas='default')
# pp.correlation_feature_distance(experiment, diversity, 'UN')
# pp.plot_feature_distance(experiment, diversity, 'UN', saveas='default')
# # pp.plot_feature_distance(experiment, diversity, 'UN_1', saveas='default')
# # pp.plot_feature_distance(experiment, diversity, 'UN_2', saveas='default')
# # pp.plot_feature_distance(experiment, diversity, 'UN_3', saveas='default')
#
# # pp.print_map_sum_of_dists(experiment, ['ID', 'AN', 'UN_0'], saveas='default', reverse_colors=True)
