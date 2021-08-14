from os import path
import shutil
import os
import numpy as np

import autolens as al


def test__point_dataset_structures_as_dict():

    point_dataset_0 = al.PointDataset(
        name="source_1",
        positions=al.Grid2DIrregular([[1.0, 1.0]]),
        positions_noise_map=al.ValuesIrregular([1.0]),
    )

    point_dict = al.PointDict(point_dataset_list=[point_dataset_0])

    assert point_dict["source_1"].name == "source_1"
    assert point_dict["source_1"].positions.in_list == [(1.0, 1.0)]
    assert point_dict["source_1"].positions_noise_map.in_list == [1.0]
    assert point_dict["source_1"].fluxes == None
    assert point_dict["source_1"].fluxes_noise_map == None

    point_dataset_1 = al.PointDataset(
        name="source_2",
        positions=al.Grid2DIrregular([[1.0, 1.0]]),
        positions_noise_map=al.ValuesIrregular([1.0]),
        fluxes=al.ValuesIrregular([2.0, 3.0]),
        fluxes_noise_map=al.ValuesIrregular([4.0, 5.0]),
    )

    point_dict = al.PointDict(point_dataset_list=[point_dataset_0, point_dataset_1])

    assert point_dict["source_1"].name == "source_1"
    assert point_dict["source_1"].positions.in_list == [(1.0, 1.0)]
    assert point_dict["source_1"].positions_noise_map.in_list == [1.0]
    assert point_dict["source_1"].fluxes == None
    assert point_dict["source_1"].fluxes_noise_map == None

    assert point_dict["source_2"].name == "source_2"
    assert point_dict["source_2"].positions.in_list == [(1.0, 1.0)]
    assert point_dict["source_2"].positions_noise_map.in_list == [1.0]
    assert point_dict["source_2"].fluxes.in_list == [2.0, 3.0]
    assert point_dict["source_2"].fluxes_noise_map.in_list == [4.0, 5.0]

    assert (point_dict.positions_list[0] == np.array([1.0, 1.0])).all()
    assert (point_dict.positions_list[1] == np.array([1.0, 1.0])).all()


def test__inputs_are_other_python_types__converted_correctly():

    point_dataset_0 = al.PointDataset(
        name="source_1", positions=[[1.0, 1.0]], positions_noise_map=[1.0]
    )

    point_dict = al.PointDict(point_dataset_list=[point_dataset_0])

    assert point_dict["source_1"].name == "source_1"
    assert point_dict["source_1"].positions.in_list == [(1.0, 1.0)]
    assert point_dict["source_1"].positions_noise_map.in_list == [1.0]
    assert point_dict["source_1"].fluxes == None
    assert point_dict["source_1"].fluxes_noise_map == None

    point_dataset_0 = al.PointDataset(
        name="source_1",
        positions=[(1.0, 1.0), (2.0, 2.0)],
        positions_noise_map=[1.0],
        fluxes=[2.0],
        fluxes_noise_map=[3.0],
    )

    point_dict = al.PointDict(point_dataset_list=[point_dataset_0])

    assert point_dict["source_1"].name == "source_1"
    assert point_dict["source_1"].positions.in_list == [(1.0, 1.0), (2.0, 2.0)]
    assert point_dict["source_1"].positions_noise_map.in_list == [1.0]
    assert point_dict["source_1"].fluxes.in_list == [2.0]
    assert point_dict["source_1"].fluxes_noise_map.in_list == [3.0]


def test__from_json_and_output_to_json():

    point_dataset_0 = al.PointDataset(
        name="source_1",
        positions=al.Grid2DIrregular([[1.0, 1.0]]),
        positions_noise_map=al.ValuesIrregular([1.0]),
    )

    point_dataset_1 = al.PointDataset(
        name="source_2",
        positions=al.Grid2DIrregular([[1.0, 1.0]]),
        positions_noise_map=al.ValuesIrregular([1.0]),
        fluxes=al.ValuesIrregular([2.0, 3.0]),
        fluxes_noise_map=al.ValuesIrregular([4.0, 5.0]),
    )

    point_dict = al.PointDict(point_dataset_list=[point_dataset_0, point_dataset_1])

    dir_path = path.join("{}".format(path.dirname(path.realpath(__file__))), "files")

    if path.exists(dir_path):
        shutil.rmtree(dir_path)

    os.makedirs(dir_path)

    file_path = path.join(dir_path, "point_dict.json")

    point_dict.output_to_json(file_path=file_path, overwrite=True)

    point_dict_via_json = al.PointDict.from_json(file_path=file_path)

    assert point_dict_via_json["source_1"].name == "source_1"
    assert point_dict_via_json["source_1"].positions.in_list == [(1.0, 1.0)]
    assert point_dict_via_json["source_1"].positions_noise_map.in_list == [1.0]
    assert point_dict_via_json["source_1"].fluxes == None
    assert point_dict_via_json["source_1"].fluxes_noise_map == None

    assert point_dict_via_json["source_2"].name == "source_2"
    assert point_dict_via_json["source_2"].positions.in_list == [(1.0, 1.0)]
    assert point_dict_via_json["source_2"].positions_noise_map.in_list == [1.0]
    assert point_dict_via_json["source_2"].fluxes.in_list == [2.0, 3.0]
    assert point_dict_via_json["source_2"].fluxes_noise_map.in_list == [4.0, 5.0]
