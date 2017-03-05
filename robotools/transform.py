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


def check_argument_roll_pitch_yaw_(roll_pitch_yaw):
    if not isinstance(roll_pitch_yaw, (tuple, list)):
        if not isinstance(roll_pitch_yaw, np.ndarray):
            raise ValueError('Expected tuple, list, or numpy.ndarray subclass '
                             'for roll_pitch_yaw, instead got {}.'.format(
                    type(roll_pitch_yaw)))


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


# Homogeneous transformation matrix generation
# -----------------------------------------------------------------------------


def r2t(rot_mats):
    """
    Generate homogeneous transforms from rotation matrices.

    The transforms have no translational component.  If the rotation matrices
    are 2 x 2, the transforms are 3 x 3.  If the rotation matrices are 3 x 3
    the transforms are 4 x 4.

    Parameters
    ----------
    rot_mats : 2 x 2 or 3 x 3 or 2 x 2 x n or 3 x 3 x n numpy.ndarray
        Single or array of rotational matrices to be converted.

    Returns
    -------
    3 x 3 or 4 x 4 or 3 x 3 x n or 4 x 4 x n homogeneous transforms.

    Raises
    ------
    ValueError
        If `rot_mats` is not a valid shape for rotation matrices.

    See Also
    --------
    t2r : rotation matrices from homogeneous transformations
    """
    if not any(((rot_mats.ndim == 2 and rot_mats.shape in ((2, 2), (3, 3))),
                (rot_mats.ndim == 3 and rot_mats.shape[1:] in ((2, 2), (3, 3))))):
        raise ValueError('Argument rot_mats must have a shape in ((2, 2), (3, '
                         '3), (n, 2, 2), (n, 3, 3)) but instead was {}.'
                         .format(rot_mats.shape))

    # All we're doing is adding an extra column and row, all zeros except for
    # the bottom-right element which is 1.
    is_2d = rot_mats.shape[-1] == 2
    extra_column = np.zeros(2 if is_2d else 3)
    extra_row = np.append(np.zeros(2 if is_2d else 3), [1])

    # For the case with just one plane, the column_stack and row_stack functions
    # do the trick.  We need to use the more general concatenate function for
    # when there are multiple planes though.
    is_single_matrix = rot_mats.ndim == 2
    if is_single_matrix:
        homo_trans = np.column_stack((rot_mats, extra_column))
        return np.row_stack((homo_trans, extra_row))
    else:
        extra_columns = np.tile(extra_column[:, np.newaxis],
                                (rot_mats.shape[0], 1, 1))
        extra_rows = np.tile(extra_row, (rot_mats.shape[0], 1, 1))
        homo_trans = np.concatenate((rot_mats, extra_columns), axis=2)
        return np.concatenate((homo_trans, extra_rows), axis=1)


def t2r(trans_mats):
    """
    Generate rotation matrices from homogeneous transformations.

    If the transforms are 3 x 3, the rotational matrices are 2 x 2.  If the
    transforms are 4 x 4, the rotational matrices are 3 x 3.

    Parameters
    ----------
    trans_mats : 3 x 3 or 4 x 4 or 3 x 3 x n or 4 x 4 x n numpy.ndarray
        Single or array of homogeneous transformations to be converted.

    Returns
    -------
    2 x 2 or 3 x 3 or 2 x 2 x n or 3 x 3 x n rotation matrices.

    Raises
    ------
    ValueError
        If `trans_mats` is not a valid shape for homogeneous transforms.

    See Also
    --------
    r2t : homogeneous transformations from rotation matrices
    """
    if not any(((trans_mats.ndim == 2 and trans_mats.shape in ((3, 3), (4, 4))),
                (trans_mats.ndim == 3 and trans_mats.shape[1:] in ((3, 3), (4, 4))))):
        raise ValueError('Argument trans_mats must have a shape in ((3, 3), '
                         '(4, 4), (n, 3, 3), (n, 4, 4)) but instead was {}.'
                         .format(trans_mats.shape))

    # We're just chopping off the last row and column.
    is_single_matrix = trans_mats.ndim == 2
    if is_single_matrix:
        return trans_mats[:-1, :-1].copy()
    else:
        return trans_mats[:, :-1, :-1].copy()


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
    theta = convert_angle_(theta, units)
    return r2t(rot2(theta, units))


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

    rot_func = {'x': rotx, 'y': roty, 'z': rotz}[axis]

    return r2t(rot_func(theta, units))


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


def se2(x, y, theta, units='rad'):
    """
    Create planar translation and rotation information.

    Parameters
    ----------
    x : int or float
        Horizontal translation.

    y : int or float
        Vertical translation.

    theta : int or float
        Rotation angle.

    units : {'rad', 'deg'}, optional
        'rad' if `theta` is given in radians, 'deg' if degrees.

    Returns
    -------
    3 x 3 numpy.ndarray
        Homogeneous transform representing translation by `x` and `y` and
        rotation by `theta` in the plane.

    See Also
    --------
    rot2 : Generate 2 x 2 rotation matrix
    """
    theta = convert_angle_(theta, units)

    sine_theta = np.sin(theta)
    cos_theta = np.cos(theta)

    return np.array([[cos_theta, -sine_theta, x],
                     [sine_theta, cos_theta, y],
                     [0, 0, 1]])


def se3(transform):
    """
    Generate a 4 x 4 homogeneous transform from a 3 x 3 homogeneous transform.

    Parameters
    ----------
    transform : 3 x 3 numpy.ndarray
        Transform to lift to 4 x 4.

    Returns
    -------
    4 x 4 numpy.ndarray
        Lifted transform.

    Raises
    ------
    ValueError
        If `transform` is invalid.

    See Also
    --------
    se2 : Generate 3 x 3 homogeneous transform with translational component
    rotx, roty, rotz : Generate 3 x 3 rotation matrix
    trotx, troty, trotz : Generate 4 x 4 homogeneous transformation
    """
    if not isinstance(transform, np.ndarray) \
       or not transform.shape == (3, 3):
        raise ValueError('Expected 3 x 3 numpy.ndarray,'
                         'instead got {}'.format(type))

    return np.array([[transform[0, 0], transform[0, 1], 0, transform[0, 2]],
                     [transform[1, 0], transform[1, 1], 0, transform[1, 2]],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])


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
        If units or axis_order is invalid.

    See Also
    --------
    rpy2t : roll-pitch-yaw to homogeneous transform
    rpy2jac : roll-pitch-yaw to Jacobian matrix
    """
    check_argument_roll_pitch_yaw_(roll_pitch_yaw)

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


def rpy2t(roll_pitch_yaw, units='rad', axis_order='xyz'):
    """
    Generate homogeneous transform from roll-pitch-yaw angles.

    Note the transform has no translational component.

    Parameters
    ----------
    roll_pitch_yaw : numpy.ndarray or tuple of int or tuple of float or
                     list of int or list of float.
                     If tuple or list: roll, pitch, and yaw.
                     If numpy.ndarray: n x 3 array of roll, pitch, and yaw.

    units : {'rad', 'deg'}, optional
        'rad' if `roll_pitch_yaw` is given in radians, 'deg' if degrees.

    axis_order : {'xyz', 'zyx'}, optional
        'xyz' for rotations about x, y, z axes or 'zyx' for rotations about
        z, y, x axes.

    Returns
    -------
    4 x 4 x n numpy.ndarray
        Array of n homogeneous transforms.

    Raises
    ------
    ValueError
        If units or axis_order is invalid.

    See Also
    --------
    rpy2r : roll-pitch-yaw to rotation matrix
    rpy2jac : roll-pitch-yaw to Jacobian matrix
    """
    return r2t(rpy2r(roll_pitch_yaw, units, axis_order))


def tr2rpy(mat, units='rad', axis_order='xyz'):
    """
    Get roll-pitch-yaw angles from a rotation matrix or homogeneous transform.

    There is a singularity for the case where the pitch is pi / 2 in which case
    the roll is arbitrarily set to zero and the yaw is roll + yaw.

    Parameters
    ----------
    mat : 3 x 3 or 4 x 4 or 3 x 3 x n or 4 x 4 x n numpy.ndarray
        Single or array of either rotation matrices or homogeneous transforms.

    units : {'rad', 'deg'}, optional
        'rad' if `mat` is given in radians, 'deg' if degrees.

    axis_order : {'xyz', 'zyx'}, optional
        'xyz' for rotations about x, y, z axes or 'zyx' for rotations about
        z, y, x axes.

    Returns
    -------
    3-tuple of floats
        Roll, pitch, yaw.

    Raises
    ------
    ValueError
        If mat is not a numpy.ndarray or is an invalid shape or units or axis
        are invalid.

    See Also
    --------
    rpy2r : roll-pitch-yaw to rotation matrix
    rpy2t : roll-pitch-yaw to homogeneous transform
    """
    # TODO: the part of the docstring about the singularity was copied almost
    #       exactly, need to verify it is correct.
    if not isinstance(mat, np.ndarray):
        raise ValueError('Expected numpy.ndarray subclass for mat, instead '
                         'got {}.'.format(type(mat)))

    if not any(((mat.ndim == 2 and mat.shape in ((3, 3), (4, 4))),
                (mat.ndim == 3 and mat.shape[1:] in ((3, 3), (4, 4))))):
        raise ValueError('Argument mat must have a shape in ((3, 3), '
                         '(4, 4), (n, 3, 3), (n, 4, 4)) but instead was {}.'
                         .format(mat.shape))

    if axis_order not in ('xyz', 'zyx'):
        raise ValueError("Expected one of ('xyz', 'zyx') for argument "
                         "axis_order, instead got {}.".format(axis_order))

    # TODO: the rest of this function can surely be refactored to far fewer LOC

    is_single_matrix = mat.ndim == 2
    if not is_single_matrix:
        return np.stack((tr2rpy(single_mat, units, axis_order)
                         for single_mat in mat), 0)

    epsilon = np.spacing(1)  # equivalent to MATLAB's eps built-in
    if axis_order == 'xyz':
        singularity_present = all(abs(element) < epsilon for element in
                                  (mat[1, 2], mat[2, 2]))
        if singularity_present:
            roll = 0
            pitch = np.arctan2(mat[0, 2], mat[2, 2])
            yaw = np.arctan2(mat[1, 0], mat[1, 1])
        else:  # no singularity
            roll = np.arctan2(-mat[1, 2], mat[2, 2])
            sine_roll = np.sin(roll)
            cos_roll = np.cos(roll)
            pitch = np.arctan2(mat[0, 2],
                               cos_roll * mat[2, 2] - sine_roll * mat[1, 2])
            yaw = np.arctan2(-mat[0, 1], mat[0, 0])
    else:  # axis_order == 'zyx'
        singularity_present = all(abs(element) < epsilon for element in
                                  (mat[0, 0], mat[1, 0]))
        if singularity_present:
            roll = 0
            pitch = np.arctan2(-mat[2, 0], mat[0, 0])
            yaw = np.arctan2(-mat[1, 2], mat[1, 1])
        else:  # no singularity
            roll = np.arctan2(mat[1, 0], mat[0, 0])
            sine_roll = np.sin(roll)
            cos_roll = np.cos(roll)
            pitch = np.arctan2(-mat[2, 0],
                               cos_roll * mat[0, 0] + sine_roll * mat[1, 0])
            yaw = np.arctan2(sine_roll * mat[0, 2] - cos_roll * mat[1, 2],
                             cos_roll * mat[1, 1] - sine_roll * mat[0, 1])

    if units == 'deg':
        roll, pitch, yaw = np.degrees((roll, pitch, yaw))

    return roll, pitch, yaw


def rpy2jac(roll_pitch_yaw, units='rad'):
    """
    Generate angular velocity Jacobian matrix from roll-pitch-yaw angles.

    Used in the creation of an analytical Jacobian.

    Parameters
    ----------
    roll_pitch_yaw : numpy.ndarray or tuple or list of int or float
        If tuple or list: roll, pitch, and yaw.
        If numpy.ndarray: n x 3 array of roll, pitch, and yaw.

    units : {'rad', 'deg'}, optional
        'rad' if `roll_pitch_yaw` is given in radians, 'deg' if degrees.

    Returns
    -------
    3 x 3 x n numpy.ndarray
        Array of n Jacobian matrices.

    Raises
    ------
    ValueError
        If units is invalid.

    See Also
    --------
    rpy2r : roll-pitch-yaw to rotation matrix
    rpy2t : roll-pitch-yaw to homogeneous transform
    """
    check_argument_roll_pitch_yaw_(roll_pitch_yaw)

    roll_pitch_yaw = convert_angle_(roll_pitch_yaw, units)

    if isinstance(roll_pitch_yaw, (tuple, list)) or roll_pitch_yaw.ndim == 1:
        roll, pitch, yaw = roll_pitch_yaw
        return np.array([[1, 0, np.sin(pitch)],
                         [0, np.cos(roll), -np.cos(pitch) * np.sin(roll)],
                         [0, np.sin(roll), np.cos(pitch) * np.cos(roll)]])
    else:
        return np.stack((np.array([[1, 0, np.sin(pitch)],
                                   [0, np.cos(roll), -np.cos(pitch) * np.sin(roll)],
                                   [0, np.sin(roll), np.cos(pitch) * np.cos(roll)]])
                         for roll, pitch, yaw in roll_pitch_yaw), 0)
