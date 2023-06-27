import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.art3d as art3d
import numpy as np

from matplotlib.patches import FancyArrowPatch, Circle
from mpl_toolkits.mplot3d import proj3d
from typing import Any, Dict, List, Optional, AnyStr


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs: float, ys: float, zs: float, *args, **kwargs):
        """
        Class for drawing a 3D arrow.

        Args:
            xs (float): x coordinate of arrow
            ys (float): y coordinate of arrow
            zs (float): z coordinate of arrow
        """
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        """Draw the 3D arrow.

        Args:
            renderer: 3D renderer
        """
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)


class BlochSphere:
    """Class for plotting points and vectors on the Bloch Sphere.

    Typical usage example:

            b = BlochSphere(point_alpha=0.7,
                            xy_projection=True,
                            xz_projection=True,
                            yz_projection=True,
                            show_3d_projection=True,
                            plot_2d_slice=True)
            b.add_vector([x1, y1, z1], color='k', label='Vector 1')
            b.add_vector([x2, y2, z1], color='b', label='Vector 2')
            b.add_points([x_array, y_array, z_array],
                         color='orange', label='A bunch of scatter points')
            b.show(save=True, save_pdf=True,
                   directory='../data/Figures/', filename='Bloch Sphere')
    """

    def __init__(self,
                 figsize: tuple = None,
                 rotation_angle: float = 45,
                 label_fontsize: float = 35,
                 tick_label_fontsize: float = 20,
                 point_size: float = 30,
                 point_alpha: float = 1.0,
                 point_edgecolor: str = 'k',
                 vector_linewdith: float = 3,
                 vector_arrowhead_scale: float = 35,
                 show_background_grid: bool = True,
                 show_background: bool = True,
                 xy_projection: bool = False,
                 yz_projection: bool = False,
                 zx_projection: bool = False,
                 show_3d_projection: bool = False,
                 plot_2d_slice: bool = False):
        """Initialize the Bloch Sphere object.

        Args:
            figsize (tuple, optional): figure size for Bloch Sphere. Defaults 
                to None.
            rotation_angle (float, optional): angle about the z-axis to rotate 
                the Bloch sphere for viewing. Defaults to 45.
            label_fontsize (float, optional): fontsize for x-, y-, z-labels. 
                Defaults to 35.
            tick_label_fontsize (float, optional): fontsize for x-, y-, z-ticks. 
                Defaults to 20.
            point_size (float, optional): point size for scatter plots. 
                Defaults to 30.
            point_alpha (float, optional): opacity for points in scatter plots. 
                Defaults to 1.0.
            point_edgecolor (str, optional): color around edges of points. 
                Defaults to 'k'.
            vector_linewdith (float, optional): linewidth of vector in Bloch 
                sphere. Defaults to 3.
            vector_arrowhead_scale (float, optional): mutation scale of vector 
                arrowhead. Defaults to 35.
            show_background_grid (bool, optional):  display x, y, z grids 
                behind Bloch sphere. Defaults to True.
            show_background (bool, optional): display background behind Bloch 
                sphere. Defaults to True.
            xy_projection (bool, optional): plot a projection of the data on 
                the XY plane. Defaults to False.
            yz_projection (bool, optional): plot a projection of the data on 
                the YZ plane. Defaults to False.
            zx_projection (bool, optional): plot a projection of the data on 
                the ZX plane. Defaults to False.
            show_3d_projection (bool, optional): plot the projection onto a 2D 
                plane behind the 3D Bloch sphere. Defaults to False.
            plot_2d_slice (bool, optional):plot the projection as slice on a 
                separate 2D graph. Defaults to False.
        """

        self.figsize = figsize
        self.label_fontsize = label_fontsize
        self.tick_label_fontsize = tick_label_fontsize
        self.point_size = point_size
        self.point_alpha = point_alpha
        self.rotation_angle = rotation_angle % 360  # [0, 360)
        self.point_edgecolor = point_edgecolor
        self.vector_linewdith = vector_linewdith
        self.vector_arrowhead_scale = vector_arrowhead_scale
        self.show_background_grid = show_background_grid
        self.show_background = show_background
        self.xy_projection = xy_projection
        self.yz_projection = yz_projection
        self.zx_projection = zx_projection
        self.show_3d_projection = show_3d_projection
        self.plot_2d_slice = plot_2d_slice

        self.fig = None
        self.ax = None
        self.ax1 = None
        self.ax2 = None
        self.ax3 = None
        self.zorder = 1  # enforce the order of plotting
        self.include_legend = False

        if 90 < self.rotation_angle < 270:
            self.sign_yz = -1
        else:
            self.sign_yz = 1

        if 180 < self.rotation_angle:
            self.sign_zx = -1
        else:
            self.sign_zx = 1

    def draw_bloch_sphere(self):
        """Draws an empty Bloch sphere."""
        phi = np.linspace(0, 2 * np.pi, 50)
        theta = np.linspace(0, np.pi, 50)
        PHI, THETA = np.meshgrid(phi, theta)

        x_sphere = np.sin(PHI) * np.cos(THETA)
        y_sphere = np.sin(PHI) * np.sin(THETA)
        z_sphere = np.cos(PHI)

        num_subplots = 1
        if self.plot_2d_slice is True:
            if self.xy_projection is True:
                num_subplots += 1
            if self.yz_projection is True:
                num_subplots += 1
            if self.zx_projection is True:
                num_subplots += 1

        if num_subplots < 4:
            # figsize = (num_subplots * figsize[0], figsize[1])
            # subplots[1] = num_subplots
            figsize = (num_subplots * 10, 10)
            subplots = (1, num_subplots, 1)
        elif num_subplots == 4:
            figsize = (20, 20)
            subplots = (2, 2, 1)
        rows, cols, subplot = subplots

        self.fig = plt.figure(
            figsize=self.figsize if self.figsize is not None else figsize
        )

        # Main Bloch sphere plot
        self.ax = self.fig.add_subplot(rows, cols, subplot, projection='3d') 
        self.ax.plot_wireframe(x_sphere, y_sphere, z_sphere,rstride=1, 
                               cstride=1, color='k', alpha=0.1, lw=1)
        self.ax.plot([-1, 1], [0, 0], [0, 0], c='k', alpha=0.5)
        self.ax.plot([0, 0], [-1, 1], [0, 0], c='k', alpha=0.5)
        self.ax.plot([0, 0], [0, 0], [-1, 1], c='k', alpha=0.5)
        self.ax.plot(np.cos(phi), np.sin(phi), 0, c='k', alpha=0.5)
        self.ax.plot(np.zeros(50), np.sin(phi), np.cos(phi), c='k', alpha=0.5)
        self.ax.plot(np.sin(phi), np.zeros(50), np.cos(phi), c='k', alpha=0.5)
        self.ax.set_xlabel(r'$\langle x \rangle$',
                           fontsize=self.label_fontsize)
        self.ax.set_ylabel(r'$\langle y \rangle$',
                           fontsize=self.label_fontsize)
        self.ax.set_zlabel(r'$\langle z \rangle$',
                           fontsize=self.label_fontsize)
        self.ax.set_xlim(-1, 1)
        self.ax.set_ylim(-1, 1)
        self.ax.set_zlim(-1, 1)
        self.ax.set_xticklabels(['-1', '', '', '', '', '', '', '', '1'],
                                fontsize=self.tick_label_fontsize)
        self.ax.set_yticklabels(['-1', '', '', '', '', '', '', '', '1'],
                                fontsize=self.tick_label_fontsize)
        self.ax.set_zticklabels(['-1', '', '', '', '', '', '', '', '1'],
                                fontsize=self.tick_label_fontsize)
        self.ax.set_facecolor('white')
        self.ax.grid(self.show_background_grid, color='k')
        if self.show_background is False:
            self.ax.set_axis_off()
        if self.rotation_angle is not None:
            self.ax.view_init(30, self.rotation_angle)

        if self.xy_projection is True:
            subplot += 1

            if self.show_3d_projection is True:
                circle = Circle((0, 0), 1, color='grey', fill=False)
                self.ax.add_patch(circle)
                art3d.pathpatch_2d_to_3d(circle, z=-1, zdir='z')

            if self.plot_2d_slice is True:
                circle = plt.Circle((0, 0), 1, color='grey', lw=5, fill=False)
                self.ax1 = self.fig.add_subplot(rows, cols, subplot)
                self.ax1.add_artist(circle)
                self.ax1.set_xlabel(r'$\langle x \rangle$',
                                    fontsize=self.label_fontsize)
                self.ax1.set_ylabel(r'$\langle y \rangle$',
                                    fontsize=self.label_fontsize)
                self.ax1.set_xlim(-1.02, 1.02)
                self.ax1.set_ylim(-1.02, 1.02)
                self.ax1.set_xticks([-1, -0.5, 0, 0.5, 1])
                self.ax1.set_yticks([-1, -0.5, 0, 0.5, 1])
                self.ax1.set_xticklabels([-1, -0.5, 0, 0.5, 1],
                                         fontsize=self.tick_label_fontsize)
                self.ax1.set_yticklabels([-1, -0.5, 0, 0.5, 1],
                                         fontsize=self.tick_label_fontsize)

        if self.yz_projection is True:
            subplot += 1

            if self.show_3d_projection is True:
                circle = Circle((0, 0), 1, color='grey', fill=False)
                self.ax.add_patch(circle)
                art3d.pathpatch_2d_to_3d(circle, z=-1*self.sign_yz, zdir='x')

            if self.plot_2d_slice is True:
                circle = plt.Circle((0, 0), 1, color='grey', lw=5, fill=False)
                self.ax3 = self.fig.add_subplot(rows, cols, subplot)
                self.ax3.add_artist(circle)
                self.ax3.set_xlabel(r'$\langle y \rangle$',
                                    fontsize=self.label_fontsize)
                self.ax3.set_ylabel(r'$\langle z \rangle$',
                                    fontsize=self.label_fontsize)
                self.ax3.set_xlim(-1.02, 1.02)
                self.ax3.set_ylim(-1.02, 1.02)
                self.ax3.set_xticks([-1, -0.5, 0, 0.5, 1])
                self.ax3.set_yticks([-1, -0.5, 0, 0.5, 1])
                self.ax3.set_xticklabels([-1, -0.5, 0, 0.5, 1],
                                         fontsize=self.tick_label_fontsize)
                self.ax3.set_yticklabels([-1, -0.5, 0, 0.5, 1],
                                         fontsize=self.tick_label_fontsize)

        if self.zx_projection is True:
            subplot += 1

            if self.show_3d_projection is True:
                circle = Circle((0, 0), 1, color='grey', fill=False)
                self.ax.add_patch(circle)
                art3d.pathpatch_2d_to_3d(circle, z=-1*self.sign_zx, zdir='y')

            if self.plot_2d_slice is True:
                circle = plt.Circle((0, 0), 1, color='grey', lw=5, fill=False)
                self.ax2 = self.fig.add_subplot(rows, cols, subplot)
                self.ax2.add_artist(circle)
                self.ax2.set_xlabel(r'$\langle x \rangle$',
                                    fontsize=self.label_fontsize)
                self.ax2.set_ylabel(r'$\langle z \rangle$',
                                    fontsize=self.label_fontsize)
                self.ax2.set_xlim(-1.02, 1.02)
                self.ax2.set_ylim(-1.02, 1.02)
                self.ax2.set_xticks([-1, -0.5, 0, 0.5, 1])
                self.ax2.set_yticks([-1, -0.5, 0, 0.5, 1])
                self.ax2.set_xticklabels([-1, -0.5, 0, 0.5, 1],
                                         fontsize=self.tick_label_fontsize)
                self.ax2.set_yticklabels([-1, -0.5, 0, 0.5, 1],
                                         fontsize=self.tick_label_fontsize)

    def add_points(self, points: List, color: str = None, label: str = None, 
                   linewidth: float = 1.5, size: float = None) -> None:
        """Adds points to the Bloch sphere.

        Args:
            points (list[list]): [x, y, z] coordinates for a point. Each can be
                an individual list of multiple coordinates for multiple points.
            color (str, RGB, optional): color of points for scatter point. 
                Defaults to None.
            label (str, optional): label of the points for the legend. 
                Defaults to None.
            linewidth (float, int, optional): width of the edgecolor around the 
                points. Defaults to 1.5.
            size (float, int, optional): size of the scatter points. Defaults 
                to None.
        """
        if self.fig is None:
            self.draw_bloch_sphere()

        if label is not None:
            self.include_legend = True

        x, y, z = points
        if color is None:
            sc = self.ax.scatter3D(x, y, z, 
                                   alpha=self.point_alpha, 
                                   edgecolor=self.point_edgecolor,
                                   lw=linewidth, label=label, 
                                   s=self.point_size if size is None else size)
            color = sc.get_facecolor()
        else:
            self.ax.scatter3D(x, y, z, 
                              color=color, alpha=self.point_alpha,
                              edgecolor=self.point_edgecolor, lw=linewidth, 
                              label=label,
                              s=self.point_size if size is None else size)

        if self.xy_projection is True:

            if self.show_3d_projection is True:
                self.ax.scatter(x, y, 
                                zs=-1, zdir='z', color=color, 
                                alpha=self.point_alpha,
                                edgecolor=self.point_edgecolor, lw=linewidth, 
                                zorder=self.zorder,
                                s=self.point_size if size is None else size)

            if self.plot_2d_slice is True:
                self.ax1.scatter(x, y, 
                                 color=color, alpha=self.point_alpha,
                                 edgecolor=self.point_edgecolor, lw=linewidth,
                                 s=self.point_size if size is None else size)

        if self.yz_projection is True:

            if self.show_3d_projection is True:
                self.ax.scatter(y, z, 
                                zs=-1*self.sign_yz, zdir='x', color=color,
                                alpha=self.point_alpha, 
                                edgecolor=self.point_edgecolor, lw=linewidth,
                                zorder=self.zorder, 
                                s=self.point_size if size is None else size)

            if self.plot_2d_slice is True:
                self.ax3.scatter(y, z, 
                                 color=color, alpha=self.point_alpha,
                                 edgecolor=self.point_edgecolor, lw=linewidth, 
                                 s=self.point_size if size is None else size)

        if self.zx_projection is True:

            if self.show_3d_projection is True:
                self.ax.scatter(x, z, 
                                zs=-1*self.sign_zx, zdir='y', color=color,
                                alpha=self.point_alpha, 
                                edgecolor=self.point_edgecolor, lw=linewidth,
                                zorder=self.zorder, 
                                s=self.point_size if size is None else size)

            if self.plot_2d_slice is True:
                self.ax2.scatter(x, z, 
                                 color=color, alpha=self.point_alpha,
                                 edgecolor=self.point_edgecolor, lw=linewidth, 
                                 s=self.point_size if size is None else size)

        self.zorder += 1

    def add_trajectory(self, points, color=None, label=None, linestyle='-', linewidth=2.0, marker='o', ms=None):
        """Adds a trajectory to the Bloch sphere.

        :param points: [x, y, z] coordinates for a point
            Each can be an individual list of multiple coordinates for multiple points.
        :type points: list|np.array
        :param color: color of points for scatter point (default: None)
        :type color: None|str|RGB
        :param label: label of the points for the legend (default: None)
        :type label: None|str
        """
        if self.fig is None:
            self.draw_bloch_sphere()

        if label is not None:
            self.include_legend = True

        x, y, z = points
        if color is None:
            p = self.ax.plot(x, y, z, alpha=self.point_alpha, ls=linestyle, label=label, lw=linewidth, marker=marker,
                             ms=self.point_size if ms is None else ms)
            color = p[0].get_color()
        else:
            self.ax.plot(x, y, z, alpha=self.point_alpha, color=color, label=label, ls=linestyle, lw=linewidth,
                         marker=marker, ms=self.point_size if ms is None else ms)

        if self.xy_projection is True:

            if self.show_3d_projection is True:
                self.ax.plot(x, y, zs=-1, zdir='z', alpha=self.point_alpha, color=color, ls=linestyle, lw=linewidth,
                             marker=marker,  ms=self.point_size if ms is None else ms, zorder=self.zorder)

            if self.plot_2d_slice is True:
                self.ax1.plot(x, y, alpha=self.point_alpha, color=color, ls=linestyle, lw=linewidth, marker=marker,
                              ms=self.point_size if ms is None else ms)

        if self.yz_projection is True:

            if self.show_3d_projection is True:
                self.ax.plot(y, z, zs=-1*self.sign_yz, zdir='x', alpha=self.point_alpha, color=color, ls=linestyle,
                             lw=linewidth, marker=marker, ms=self.point_size if ms is None else ms, zorder=self.zorder)

            if self.plot_2d_slice is True:
                self.ax3.plot(y, z, alpha=self.point_alpha, color=color, ls=linestyle, lw=linewidth, marker=marker,
                              ms=self.point_size if ms is None else ms)

        if self.zx_projection is True:

            if self.show_3d_projection is True:
                self.ax.plot(x, z, zs=-1*self.sign_zx, zdir='y', alpha=self.point_alpha, color=color, ls=linestyle,
                             lw=linewidth, marker=marker, ms=self.point_size if ms is None else ms, zorder=self.zorder)

            if self.plot_2d_slice is True:
                self.ax2.plot(x, z, alpha=self.point_alpha, color=color, ls=linestyle, lw=linewidth, marker=marker,
                              ms=self.point_size if ms is None else ms)

        self.zorder += 1

    def add_vector(self, vector, color=None, label=None):
        """Add a vector to the Bloch sphere.

        :param vector: [x, y, z] coordinates for the tip of a vector
        :type vector: list|np.array
        :param color: color of vector (default: None)
        :type color: None|str|RGB
        :param label: label of the vector for the legend (default: None)
        :type label: None|str
        """
        if self.fig is None:
            self.draw_bloch_sphere()

        if label is not None:
            self.include_legend = True

        x, y, z = vector
        if color is None:
            p = self.ax.plot([0, x], [0, y], [0, z], lw=self.vector_linewdith, label=label)
            color = p[0].get_color()
        else:
            self.ax.plot([0, x], [0, y], [0, z], lw=self.vector_linewdith, color=color, label=label)
        a = Arrow3D([0, x], [0, y], [0, z], mutation_scale=self.vector_arrowhead_scale, arrowstyle='-|>', color=color)
        self.ax.add_artist(a)

        if self.xy_projection is True:

            if self.show_3d_projection is True:
                self.ax.plot([0, x], [0, y], zs=-1, zdir='z', color=color, lw=self.vector_linewdith-2,
                             zorder=self.zorder+1)
                a = Arrow3D([0, x], [0, y], [-1, -1], mutation_scale=self.vector_arrowhead_scale-10, arrowstyle='-|>',
                            color=color, zorder=self.zorder+2)
                self.ax.add_artist(a)

            if self.plot_2d_slice is True:
                self.ax1.arrow(0, 0, x, y, color=color, lw=self.vector_linewdith, head_width=0.04, head_length=0.04,
                               length_includes_head=True)

        if self.yz_projection is True:

            if self.show_3d_projection is True:
                self.ax.plot([0, y], [0, z], zs=-1*self.sign_yz, zdir='x', color=color,
                             lw=self.vector_linewdith-2, zorder=self.zorder+1)
                a = Arrow3D([-1*self.sign_yz, -1*self.sign_yz], [0, y], [0, z],
                            mutation_scale=self.vector_arrowhead_scale-10, arrowstyle='-|>', color=color,
                            zorder=self.zorder+2)
                self.ax.add_artist(a)

            if self.plot_2d_slice is True:
                self.ax3.arrow(0, 0, y, z, color=color, lw=self.vector_linewdith, head_width=0.04, head_length=0.04,
                               length_includes_head=True)

        if self.zx_projection is True:

            if self.show_3d_projection is True:
                self.ax.plot([0, x], [0, z], zs=-1*self.sign_zx, zdir='y', color=color,
                             lw=self.vector_linewdith-2, zorder=self.zorder+1)
                a = Arrow3D([0, x], [-1*self.sign_zx, -1*self.sign_zx], [0, z],
                            mutation_scale=self.vector_arrowhead_scale-10, arrowstyle='-|>', color=color,
                            zorder=self.zorder+2)
                self.ax.add_artist(a)

            if self.plot_2d_slice is True:
                self.ax2.arrow(0, 0, x, z, color=color, lw=self.vector_linewdith, head_width=0.04, head_length=0.04,
                               length_includes_head=True)

        self.zorder += 1

    def show(self, save=False, save_pdf=False, save_svg=False, directory=None, filename=None):
        """Plot the Bloch Sphere in a figure.

        :param save: save the figure (default: False)
        :type save: bool
        :param save_svg: save the figure as an svg (default: False)
        :type save_svg: bool
        :param directory: directory in which the save the figure (default: None)
            If None, it will save in the current directory.
        :type directory: None|str
        :param filename: string to prepend in front for 'Bloch_sphere.png' for a filename
        :type filename: None|str
        """
        if self.fig is None:
            self.draw_bloch_sphere()

        if self.include_legend is True:
            self.ax.legend(loc=0, fontsize=20)

        # plt.tight_layout()  # Creates issues with legend
        if save is True:
            plt.savefig(f'{directory}{filename}Bloch_sphere.png', dpi=300)
            if save_pdf is True:
                plt.savefig(f'{directory}{filename}Bloch_sphere.pdf', dpi=300)
            if save_svg is True:
                plt.savefig(f'{directory}{filename}Bloch_sphere.svg', dpi=300)
        plt.show()
