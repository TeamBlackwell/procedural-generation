import matplotlib.pyplot as plt
import numpy.typing as npt
import numpy as np

from typing import Callable, Dict, List
from qtree import QTree, find_children


class Generator(object):
    """
    Main class for procedural generation
    """

    def __init__(
        self,
        dimension: int,
        sampling_fnc: Callable,
        *,
        complex_buildings: bool = False,
        sampling_kwargs: Dict[str, any] = {},
        scale: int = 100,
        debug: bool = False
    ):
        """
        :param dimension: the dimension of the generator
        :param sampling_fnc: the random function used for sampling
        :complex_buildings: this tells us if the buildings should be rectangles or combined rectangles
        :sampling_kwargs: function arguments for the sampling function
        :scale: this is the factor by which the map should be expanded
        :debug: useful to see the output of the generator
        """
        self.debug = debug
        self.dimension = 2
        self.scale = scale
        self.complex_buildings = complex_buildings

        self.sampling_fnc = sampling_fnc
        if "scale" not in sampling_kwargs:
            sampling_kwargs["scale"] = scale
        if "dimension" not in sampling_kwargs:
            sampling_kwargs["dimension"] = dimension
        self.sampling_kwargs = sampling_kwargs
        self.qtree = None

        if self.debug:
            self.debug_fig, self.debug_ax = plt.subplots(2, 2)

    def generate_sample(self, *, show=False):
        X, Y = self.get_sample_from_sampling_fnc()
        self.make_qtree_from_samples(X, Y)
        if self.debug:
            self.debug_ax[0][1] = self.qtree.plot(self.debug_ax[0][1])
        self.populate_with_buildings()
        if show:
            plt.show()

    def get_sample_from_sampling_fnc(self):
        X, Y = self.sampling_fnc(**self.sampling_kwargs)
        if self.debug:
            sample_plot = self.debug_ax[0][0]
            sample_plot.scatter(X, Y)
            sample_plot.set_xlim([0, self.scale])
            sample_plot.set_ylim([0, self.scale])
            sample_plot.set_title("[DEBUG] Sampling Function")
            sample_plot.axis("equal")
        return X, Y

    def make_qtree_from_samples(self, X, Y):
        self.qtree = QTree(1, self.scale)
        for x, y in zip(X, Y):
            self.qtree.add_point(x, y)
        self.qtree.subdivide()

    def populate_with_buildings(self) -> npt.NDArray:
        """
        Populates the map with buildings
        """
        children = find_children(self.qtree.root)
        if self.debug:
            self.debug_ax[1][0] = self.qtree.plot(self.debug_ax[1][0])
            building_plot = self.debug_ax[1][0]
            final_plot = self.debug_ax[1][1]
            final_plot.axis("equal")
            final_plot.set_title("Generated City")
            final_plot.scatter([0, 100], [0, 100], alpha=0)
        for child in children:
            if not len(child.points):
                continue
            building_coords = make_building(
                child, complex_buildings=self.complex_buildings, debug=self.debug
            )
            if self.debug:
                plot_building(building_coords, final_plot)
                plot_building(building_coords, building_plot)


def make_building(node, *, complex_buildings=False, debug=False) -> List[npt.NDArray]:
    point = node.points[0]
    bounded_random = lambda: np.random.random() * 0.6 + 0.25
    width, height = bounded_random() * node.width, bounded_random() * node.height
    x1 = max(point.x - width / 2, node.x0)
    y1 = max(point.y - height / 2, node.y0)
    x2 = min(point.x + width / 2, node.width + node.x0)
    y2 = min(point.y + height / 2, node.height + node.y0)
    # n_rects = int(1/np.random.normal())
    return np.array([x1, y1, x2, y2])


def plot_building(coords, ax):
    ax.add_patch(
        plt.Rectangle(
            (coords[0], coords[1]),
            coords[2] - coords[0],
            coords[3] - coords[1],
        )
    )


if __name__ == "__main__":
    from sampling import sample_poisson_disk

    sampling_kwargs = {"density": 10, "n_buildings": 50}
    proc_gen = Generator(
        2,
        sample_poisson_disk,
        sampling_kwargs=sampling_kwargs,
        complex_buildings=True,
        debug=True,
    )
    proc_gen.generate_sample(show=True)
