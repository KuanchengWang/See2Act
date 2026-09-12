"""Task registry."""
from see2act.sim.tasks.base import Task
from see2act.sim.tasks.place_red_in_green import PlaceRedInGreen
from see2act.sim.tasks.bin_picking import BinPicking
from see2act.sim.tasks.put_within_shelf import PutWithinShelf
from see2act.sim.tasks.bin_search import BinSearch

names = {
    "place-red-in-green": PlaceRedInGreen,
    "bin-picking": BinPicking,
    "put-within-shelf": PutWithinShelf,
    "bin-search": BinSearch,
}


def make(name):
    return names[name]()


__all__ = ["Task", "PlaceRedInGreen", "BinPicking", "PutWithinShelf", "BinSearch", "names", "make"]
