import numpy as np


# Helper functions
# -----------------------------------------------------------------------------


def check_argument_axis_(axis):
    """Raise an error on invalid axis."""
    if axis not in ('x', 'y', 'z'):
        raise ValueError("Expected one of ('x', 'y', 'z') for argument axis "
                         "but got {}.".format(axis))


def check_argument_units_(units):
    """Raise an error on invalid units."""
    if units not in ('deg', 'rad'):
        raise ValueError("Expected one of ('deg', 'rad') for argument units "
                         "but got {}.".format(units))


def convert_angle_(theta, units):
    """
    If units == 'deg', return theta converted to radians, else return theta.
    """
    check_argument_units_(units)
    if units == 'deg':
        return np.radians(theta)
    return theta


# Rotational matrix generation
# -----------------------------------------------------------------------------


def rot2(theta, units='rad'):
    """
    Generate a 2 x 2 rotation matrix representing a rotation by angle theta.

    Parameters
    ----------
    theta : int or float
        Rotation angle in radians or degrees.

    units : {'rad', 'deg'}, optional
        'rad' if `theta` is given in radians, 'deg' if degrees.

    Returns
    -------
    2 x 2 numpy.ndarray
        Rotation matrix.

    Raises
    ------
    ValueError
        If `units` is invalid.

    See Also
    --------
    rotx, roty, rotz : Generate 3 x 3 rotation matrix
    """
    check_argument_units_(units)

    theta = convert_angle_(theta, units)
    cos = np.cos(theta)
    sine = np.sin(theta)

    return np.array([[cos, -sine],
                     [sine, cos]])


def rot_any_(theta, axis, units='rad'):
    """
    Generate a 3 x 3 rotation matrix representing a rotation by angle theta
    about the given axis.

    theta: rotation angle in radians or degrees, see units argument
    axis: one of ('x', 'y', 'z')
    unit: 'rad' if theta is given in radians, 'deg' if degrees
    """
    check_argument_axis_(axis)
    check_argument_units_(units)

    theta = convert_angle_(theta, units)
    cos = np.cos(theta)
    sine = np.sin(theta)

    if axis == 'x':
        return np.array([[1, 0, 0],
                         [0, cos, -sine],
                         [0, sine, cos]])
    elif axis == 'y':
        return np.array([[cos, 0, sine],
                         [0, 1, 0],
                         [-sine, 0, cos]])
    elif axis == 'z':
        return np.array([[cos, -sine, 0],
                         [sine, cos, 0],
                         [0, 0, 1]])


def rotx(theta, units='rad'):
    """
    Generate a 3 x 3 rotation matrix representing a rotation by angle theta
    about the x axis.

    Parameters
    ----------
    theta : int or float
        Rotation angle in radians or degrees.

    units : {'rad', 'deg'}, optional
        'rad' if `theta` is given in radians, 'deg' if degrees.

    Returns
    -------
    3 x 3 numpy.ndarray
        Rotation matrix.

    Raises
    ------
    ValueError
        If `units` is invalid.

    See Also
    --------
    roty, rotz : Generate 3 x 3 rotation matrix for other axes
    rot2 : Generate 2 x 2 rotation matrix
    """
    return rot_any_(theta, 'x', units)


def roty(theta, units='rad'):
    """
    Generate a 3 x 3 rotation matrix representing a rotation by angle theta
    about the y axis.

    Parameters
    ----------
    theta : int or float
        Rotation angle in radians or degrees.

    units : {'rad', 'deg'}, optional
        'rad' if `theta` is given in radians, 'deg' if degrees.

    Returns
    -------
    3 x 3 numpy.ndarray
        Rotation matrix.

    Raises
    ------
    ValueError
        If `units` is invalid.

    See Also
    --------
    rotx, rotz : Generate 3 x 3 rotation matrix for other axes
    rot2 : Generate 2 x 2 rotation matrix
    """
    return rot_any_(theta, 'y', units)


def rotz(theta, units='rad'):
    """
    Generate a 3 x 3 rotation matrix representing a rotation by angle theta
    about the z axis.

    Parameters
    ----------
    theta : int or float
        Rotation angle in radians or degrees.

    units : {'rad', 'deg'}, optional
        'rad' if `theta` is given in radians, 'deg' if degrees.

    Returns
    -------
    3 x 3 numpy.ndarray
        Rotation matrix.

    Raises
    ------
    ValueError
        If `units` is invalid.

    See Also
    --------
    rotx, roty : Generate 3 x 3 rotation matrix for other axes
    rot2 : Generate 2 x 2 rotation matrix
    """
    return rot_any_(theta, 'z', units)


def trot2(theta, units='rad'):
    """
    Generate a 3 x 3 homogeneous transformation matrix representing a
    rotation by angle theta with no translational component.

    Parameters
    ----------
    theta : int or float
        Rotation angle in radians or degrees.

    units : {'rad', 'deg'}, optional
        'rad' if `theta` is given in radians, 'deg' if degrees.

    Returns
    -------
    3 x 3 numpy.ndarray
        Homogeneous transformation matrix.

    Raises
    ------
    ValueError
        If `units` is invalid.

    See Also
    --------
    trotx, troty, trotz : Generate 4 x 4 homogeneous transformation matrix
    """
    check_argument_units_(units)

    theta = convert_angle_(theta, units)

    homo_trans = np.column_stack((rot2(theta, units), [0, 0]))
    return np.row_stack((homo_trans, [0, 0, 1]))


# Homogeneous transformation matrix generation
# -----------------------------------------------------------------------------


def trot_any_(theta, axis, units='rad'):
    """
    Generate a 4 x 4 homogeneous transformation matrix representing a
    rotation by angle theta about the given axis.

    Note that the translational component is zero.

    theta: rotation angle in radians or degrees, see units argument
    axis: one of ('x', 'y', 'z')
    unit: 'rad' if theta is given in radians, 'deg' if degrees
    """
    check_argument_axis_(axis)
    check_argument_units_(units)

    rot_func = {'x': rotx, 'y': roty, 'z': rotz}[axis]

    homo_trans = np.column_stack((rot_func(theta, units), [0, 0, 0]))
    return np.row_stack((homo_trans, [0, 0, 0, 1]))


def trotx(theta, units='rad'):
    """
    Generate a 4 x 4 homogeneous transformation matrix representing a
    rotation by angle theta about the x axis with no translational component.

    Parameters
    ----------
    theta : int or float
        Rotation angle in radians or degrees.

    units : {'rad', 'deg'}, optional
        'rad' if `theta` is given in radians, 'deg' if degrees.

    Returns
    -------
    4 x 4 numpy.ndarray
        Homogeneous transformation matrix.

    Raises
    ------
    ValueError
        If `units` is invalid.

    See Also
    --------
    trot2 : Generate 3 x 3 homogeneous transformation matrix
    troty, trotz : Generate 4 x 4 homogeneous transformation matrix for other
                   axes
    """
    return trot_any_(theta, 'x', units)


def troty(theta, units='rad'):
    """
    Generate a 4 x 4 homogeneous transformation matrix representing a
    rotation by angle theta about the y axis with no translational component.

    Parameters
    ----------
    theta : int or float
        Rotation angle in radians or degrees.

    units : {'rad', 'deg'}, optional
        'rad' if `theta` is given in radians, 'deg' if degrees.

    Returns
    -------
    4 x 4 numpy.ndarray
        Homogeneous transformation matrix.

    Raises
    ------
    ValueError
        If `units` is invalid.

    See Also
    --------
    trot2 : Generate 3 x 3 homogeneous transformation matrix
    trotx, trotz : Generate 4 x 4 homogeneous transformation matrix for other
                   axes
    """
    return trot_any_(theta, 'y', units)


def trotz(theta, units='rad'):
    """
    Generate a 4 x 4 homogeneous transformation matrix representing a
    rotation by angle theta about the z axis with no translational component.

    Parameters
    ----------
    theta : int or float
        Rotation angle in radians or degrees.

    units : {'rad', 'deg'}, optional
        'rad' if `theta` is given in radians, 'deg' if degrees.

    Returns
    -------
    4 x 4 numpy.ndarray
        Homogeneous transformation matrix.

    Raises
    ------
    ValueError
        If `units` is invalid.

    See Also
    --------
    trot2 : Generate 3 x 3 homogeneous transformation matrix
    trotx, troty : Generate 4 x 4 homogeneous transformation matrix for other
                   axes
    """
    return trot_any_(theta, 'z', units)


# Conversion between roll/pitch/yaw and rotational matrices
# -----------------------------------------------------------------------------


def rpy2r(roll_pitch_yaw, units='rad', axis_order='xyz'):
    """
    Generate rotation matrix from roll-pitch-yaw angles.

    Parameters
    ----------
    roll_pitch_yaw : numpy.ndarray or tuple or list of int or float
        If tuple or list: roll, pitch, and yaw.
        If numpy.ndarray: n x 3 array of roll, pitch, and yaw.

    units : {'rad', 'deg'}, optional
        'rad' if `roll_pitch_yaw` is given in radians, 'deg' if degrees.

    axis_order : {'xyz', 'zyx'}, optional
        'xyz' for rotations about x, y, z axes or 'zyx' for rotations about
        z, y, x axes.

    Returns
    -------
    3 x 3 x n numpy.ndarray
        Array of n rotation matrices.

    Raises
    ------
    ValueError
        If units or axis_order is invalid
    """
    if not isinstance(roll_pitch_yaw, (tuple, list)):
        if not isinstance(roll_pitch_yaw, np.ndarray):
            raise ValueError('Expected tuple, list, or numpy.ndarray subclass '
                             'for roll_pitch_yaw, instead got {}.'.format(
                    type(roll_pitch_yaw)))

    if axis_order == 'xyz':
        rot_func_a, rot_func_b, rot_func_c = rotx, roty, rotz
    elif axis_order == 'zyx':
        rot_func_a, rot_func_b, rot_func_c = rotz, roty, rotx
    else:
        raise ValueError("Expected one of ('xyz', 'zyx') for argument "
                         "axis_order, instead got {}.".format(axis_order))

    roll_pitch_yaw = convert_angle_(roll_pitch_yaw, units)

    if isinstance(roll_pitch_yaw, (tuple, list)) or roll_pitch_yaw.ndim == 1:
        roll, pitch, yaw = roll_pitch_yaw
        return rot_func_a(roll) @ rot_func_b(pitch) @ rot_func_c(yaw)
    else:
        return np.stack((rot_func_a(roll) @ rot_func_b(pitch) @ rot_func_c(yaw)
                         for roll, pitch, yaw in roll_pitch_yaw), 0)
