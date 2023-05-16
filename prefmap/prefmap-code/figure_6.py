#!/usr/bin/env python

import mapel.elections as mapel

if __name__ == "__main__":

    name = 'original_8x1000_maxi'
    instance_type = 'ordinal'
    experiment_id = f'{name}'

    experiment = mapel.prepare_offline_ordinal_experiment(experiment_id=experiment_id)

    experiment.prepare_elections(store_points=True, aggregated=False)

    for election in experiment.instances.values():
        election.set_default_object_type('vote')
        election.compute_distances()
        election.embed(object_type='vote')

    for election in experiment.instances.values():
        election.print_map(show=False, radius=20, name=name,
                           alpha=0.0333,
                           s=100, circles=True,
                           object_type='vote')

    experiment.merge_election_images(name=name, show=True, ncol=9, nrow=3)
