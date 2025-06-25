import numpy as np

from autolens.point.solver.step import Step


def add_triangles(triangles, color):
    from matplotlib import pyplot as plt

    for triangle in triangles:
        triangle = np.append(triangle, [triangle[0]], axis=0)
        plt.plot(triangle[:, 0], triangle[:, 1], "o-", color=color)


def visualise(step: Step):
    from matplotlib import pyplot as plt

    plt.figure(figsize=(8, 8))
    add_triangles(step.initial_triangles, color="black")
    add_triangles(step.up_sampled, color="green")
    add_triangles(step.neighbourhood, color="red")
    add_triangles(step.filtered_triangles, color="blue")

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(f"Step {step.number}")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.show()


def plot_triangles(triangles, color="black", title="Triangles", point=None):
    from matplotlib import pyplot as plt

    plt.figure(figsize=(8, 8))
    for triangle in triangles:
        triangle = np.append(triangle, [triangle[0]], axis=0)
        plt.plot(triangle[:, 0], triangle[:, 1], "o-", color=color)

    if point:
        plt.plot(point[0], point[1], "x", color="red")

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(title)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.show()


def plot_triangles_compare(triangles_a, triangles_b, number=None):
    from matplotlib import pyplot as plt

    plt.figure(figsize=(8, 8))
    for triangle in triangles_a:
        triangle = np.append(triangle, [triangle[0]], axis=0)
        plt.plot(triangle[:, 0], triangle[:, 1], "o-", color="red")

    for triangle in triangles_b:
        triangle = np.append(triangle, [triangle[0]], axis=0)
        plt.plot(triangle[:, 0], triangle[:, 1], "o-", color="blue")

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Triangles" + f" {number}" if number is not None else "")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.show()
