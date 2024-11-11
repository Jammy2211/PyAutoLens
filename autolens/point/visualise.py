from matplotlib import pyplot as plt
import numpy as np

from autolens.point.solver.step import Step


def add_triangles(triangles, color):
    for triangle in triangles:
        triangle = np.append(triangle, [triangle[0]], axis=0)
        plt.plot(triangle[:, 0], triangle[:, 1], "o-", color=color)


def visualise(step: Step):
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


def plot_triangles(triangles, color="black"):
    plt.figure(figsize=(8, 8))
    for triangle in triangles:
        triangle = np.append(triangle, [triangle[0]], axis=0)
        plt.plot(triangle[:, 0], triangle[:, 1], "o-", color=color)

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(f"Triangles")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.show()
