from autofit.mapper import model_mapper
from autofit.optimize import grid_search as gs
from autolens.model.galaxy import galaxy_model as gm

if __name__ == "__main__":
    mapper = model_mapper.ModelMapper()

    mapper.galaxy = gm.GalaxyModel(variable_redshift=True)

    assert mapper.prior_count == 1

    grid_search = gs.GridSearch(model_mapper=mapper, number_of_steps=10)

    # GalaxyModel.redshift is a PriorModel; GalaxyModel.redshift.redshift is the underlying prior
    mappers = list(grid_search.model_mappers(grid_priors=[mapper.galaxy.redshift.redshift]))
    print([m.galaxy.redshift.redshift for m in mappers])
