"""Unit tests for qcal.plotting.bloch_sphere.{Arrow3D,BlochSphere}."""
from matplotlib.patches import FancyArrowPatch

from qcal.plotting.bloch_sphere import Arrow3D, BlochSphere


class TestArrow3D:

    def test_stores_3d_vertices(self):
        a = Arrow3D(
            [0, 1], [0, 2], [0, 3], mutation_scale=20, arrowstyle='-|>',
        )
        assert a._verts3d == ([0, 1], [0, 2], [0, 3])

    def test_is_a_fancy_arrow_patch(self):
        a = Arrow3D([0, 1], [0, 1], [0, 1])
        assert isinstance(a, FancyArrowPatch)

    def test_has_do_3d_projection(self):
        # Regression test: matplotlib's 3D rendering pipeline (since
        # ~3.5) calls do_3d_projection() on every 3D artist to
        # depth-sort them before draw() runs. Without this method,
        # saving any figure containing an Arrow3D crashes with
        # AttributeError.
        assert hasattr(Arrow3D([0, 1], [0, 1], [0, 1]), 'do_3d_projection')


class TestRotationAngleAndSigns:

    def test_rotation_angle_is_wrapped_to_0_360(self):
        assert BlochSphere(rotation_angle=450).rotation_angle == 90
        assert BlochSphere(rotation_angle=-10).rotation_angle == 350

    def test_sign_yz_is_negative_strictly_between_90_and_270(self):
        assert BlochSphere(rotation_angle=0).sign_yz == 1
        assert BlochSphere(rotation_angle=90).sign_yz == 1
        assert BlochSphere(rotation_angle=180).sign_yz == -1
        assert BlochSphere(rotation_angle=270).sign_yz == 1
        assert BlochSphere(rotation_angle=360).sign_yz == 1

    def test_sign_zx_is_negative_above_180(self):
        assert BlochSphere(rotation_angle=0).sign_zx == 1
        assert BlochSphere(rotation_angle=180).sign_zx == 1
        assert BlochSphere(rotation_angle=181).sign_zx == -1
        assert BlochSphere(rotation_angle=359).sign_zx == -1


class TestDrawBlochSphere:

    def test_creates_figure_and_3d_axes(self):
        b = BlochSphere()
        b.draw_bloch_sphere()

        assert b.fig is not None
        assert b.ax is not None

    def test_no_extra_axes_without_2d_slices(self):
        b = BlochSphere()
        b.draw_bloch_sphere()

        assert b.ax1 is None
        assert b.ax2 is None
        assert b.ax3 is None
        assert len(b.fig.axes) == 1

    def test_one_extra_axis_per_requested_2d_slice(self):
        b = BlochSphere(xy_projection=True, plot_2d_slice=True)
        b.draw_bloch_sphere()

        assert b.ax1 is not None
        assert b.ax2 is None
        assert b.ax3 is None
        assert len(b.fig.axes) == 2

    def test_all_three_2d_slices_create_four_axes_total(self):
        b = BlochSphere(
            xy_projection=True, yz_projection=True, zx_projection=True,
            plot_2d_slice=True,
        )
        b.draw_bloch_sphere()

        assert b.ax1 is not None
        assert b.ax2 is not None
        assert b.ax3 is not None
        assert len(b.fig.axes) == 4


class TestAddPoints:

    def test_lazily_draws_the_sphere_if_needed(self):
        b = BlochSphere()
        assert b.fig is None

        b.add_points([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])

        assert b.fig is not None

    def test_adds_one_scatter_collection(self):
        b = BlochSphere()
        b.draw_bloch_sphere()
        n_before = len(b.ax.collections)

        b.add_points([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], color='red')

        assert len(b.ax.collections) == n_before + 1

    def test_label_enables_the_legend(self):
        b = BlochSphere()
        b.draw_bloch_sphere()
        assert b.include_legend is False

        b.add_points([[0.1], [0.2], [0.3]], label='points')

        assert b.include_legend is True


class TestAddTrajectory:

    def test_adds_one_line(self):
        b = BlochSphere()
        b.draw_bloch_sphere()
        n_before = len(b.ax.lines)

        b.add_trajectory([[0, 0.1], [0, 0.1], [0, 0.1]])

        assert len(b.ax.lines) == n_before + 1


class TestAddVector:

    def test_adds_a_line_and_an_arrow_patch(self):
        b = BlochSphere()
        b.draw_bloch_sphere()
        n_lines_before = len(b.ax.lines)
        n_patches_before = len(b.ax.patches)

        b.add_vector([0.5, 0.5, 0.5])

        assert len(b.ax.lines) == n_lines_before + 1
        assert len(b.ax.patches) == n_patches_before + 1

    def test_zorder_increments_with_each_added_element(self):
        b = BlochSphere()
        b.draw_bloch_sphere()
        z0 = b.zorder

        b.add_vector([0.5, 0.5, 0.5])
        b.add_points([[0.1], [0.2], [0.3]])
        b.add_trajectory([[0, 0.1], [0, 0.1], [0, 0.1]])

        assert b.zorder == z0 + 3


class TestShow:

    def test_does_not_crash(self):
        b = BlochSphere()
        b.add_vector([0.5, 0.5, 0.5])
        b.show()

    def test_save_writes_three_file_formats(self, tmp_path):
        b = BlochSphere()
        b.add_vector([0.5, 0.5, 0.5])

        b.show(save=True, filename=str(tmp_path) + '/')

        assert (tmp_path / 'Bloch_sphere.png').exists()
        assert (tmp_path / 'Bloch_sphere.pdf').exists()
        assert (tmp_path / 'Bloch_sphere.svg').exists()
