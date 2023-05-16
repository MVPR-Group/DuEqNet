import numpy as np
import torch

import matplotlib
import matplotlib.pyplot as plt


class C4:

    @staticmethod
    def product(r: int, s: int) -> int:
        # Implements the *group law* of the group C_4.
        # The input `r` and `s` must be integers in {0, 1, 2, 3} and represent two elements of the group.
        # The method should return the integer representing the product of the two input elements.
        # You should also check that the inputs are valid.

        # BEGIN SOLUTION
        assert r in range(0, 4), "r not in {0, 1, 2, 3}"
        assert s in range(0, 4), "r not in {0, 1, 2, 3}"

        return (r + s) % 4
        # END SOLUTION

    @staticmethod
    def inverse(r: int) -> int:
        # Implements the *inverse* operation of the group C_4.
        # The input `r` must be an integer in {0, 1, 2, 3} and represents an element of the group.
        # The method should return the integer representing the inverse of input element.
        # You should also check that the input is valid.

        # BEGIN SOLUTION
        assert r in range(0, 4), "r not in {0, 1, 2, 3}"

        return (4 - r) % 4
        # END SOLUTION


def test_C4():
    # Some test cases to check if your implementation is working
    assert C4.product(1, 3) == 0
    assert C4.product(0, 0) == 0
    assert C4.product(2, 3) == 1
    assert C4.inverse(0) == 0
    assert C4.inverse(1) == 3

    print("class C4 test finished.")


class D4:

    @staticmethod
    def product(a: tuple, b: tuple) -> tuple:
        # Implements the *group law* of the group D_4.
        # The input `a` and `b` must be tuples containing two integers, e.g. `a = (f, r)`.
        # The two integeres indicate whether the group element includes a reflection and the number of rotations.
        # The method should return the tuple representing the product of the two input elements.
        # You should also check that the inputs are valid.

        # BEGIN SOLUTION
        assert isinstance(a, tuple) and isinstance(b, tuple)
        assert a[0] in range(0, 2) and a[1] in range(0, 4)
        assert b[0] in range(0, 2) and b[1] in range(0, 4)

        # (RbFb)*(RaFa) = (RbRa^(-1)FbFa) = (RbRa^(-1))(FbFa)
        if b[0] == 0:
            return a[0], C4.product(a[1], b[1])
        else:
            f = 0 if a[0] == 1 else 1
            r = C4.product(C4.inverse(a[1]), b[1])
            return f, r
        # END SOLUTION

    @staticmethod
    def inverse(g: tuple) -> tuple:
        # Implements the *inverse* operation of the group D_4.
        # The input `g` must be a tuple containing two integers, e.g. `g = (f, r)`.
        # The two integeres indicate whether the group element includes a reflection and the number of rotations.
        # The method should return the tuple representing the inverse of the input element.
        # You should also check that the input is valid.

        # BEGIN SOLUTION
        assert isinstance(g, tuple)
        assert g[0] in range(0, 2) and g[1] in range(0, 4)

        if g[0] == 1:
            return g
        else:
            return 0, C4.inverse(g[1])
        # END SOLUTION


def test_D4():
    e = (0, 0)  # the identity element
    f = (1, 0)  # the horizontal reflection
    r = (0, 1)  # the rotation by 90 degrees

    # Let's verify that the implementation is consistent with the instructions given
    assert D4.product(e, e) == e
    assert D4.product(f, f) == e
    assert D4.product(f, r) == D4.product(D4.inverse(r), f)

    # Let's verify that the implementation satisfies the group axioms
    a = (1, 2)
    b = (0, 3)
    c = (1, 1)
    assert D4.product(a, e) == a
    assert D4.product(e, a) == a
    assert D4.product(b, D4.inverse(b)) == e
    assert D4.product(D4.inverse(b), b) == e

    assert D4.product(D4.product(a, b), c) == D4.product(a, D4.product(b, c))

    print("class D4 test finished.")


def rotate(x: torch.Tensor, r: int) -> torch.Tensor:
    # Method which implements the action of the group element `g` indexed by `r` on the input image `x`.
    # The method returns the image `g.x`

    # note that we rotate the last 2 dimensions of the input, since we want to later use this method to rotate minibatches containing multiple images
    return x.rot90(r, dims=(-2, -1))
    # dims接受一个两个元素的list或tuple，这两个轴构成了一个矩阵
    # 然后这个矩阵逆时针旋转k个90°


def test_rotate():
    x = torch.randn(1, 1, 33, 33) ** 2

    r = 1
    gx = rotate(x, r)

    plt.imshow(x[0, 0].numpy())
    plt.title('Original Image $x$')
    plt.show()

    plt.imshow(gx[0, 0].numpy())
    plt.title('Rotated Image $g.x$')
    plt.show()


def test_normal_conv():
    x = torch.randn(1, 1, 33, 33) ** 2
    r = 1

    # 测试普通卷积只具有平移不变性，没有旋转不变性
    filter3x3 = torch.randn(1, 1, 3, 3)  # (out_C, in_c, H, W)
    # 卷积核的权重

    plt.imshow(filter3x3[0, 0].numpy())
    plt.title('Filter')
    plt.show()

    psi_x = torch.conv2d(x, filter3x3, bias=None, padding=1)  # x是原图
    psi_gx = torch.conv2d(gx, filter3x3, bias=None, padding=1)  # gx是逆时针旋转90°的图

    g_psi_x = rotate(psi_x, r)  # 对原图x卷积后的结果进行逆时针旋转90°
    # 目标是验证 f(g(x)) 是否等于 g(f(x))，即卷积后旋转是否等于旋转后卷积

    plt.imshow(g_psi_x[0, 0].numpy())
    plt.title('$g.\psi(x)$')
    plt.show()

    plt.imshow(psi_gx[0, 0].numpy())
    plt.title('$\psi(g.x)$')
    plt.show()


def test_isotropic_conv():
    x = torch.randn(1, 1, 33, 33) ** 2
    r = 1

    # 约束卷积核，使得卷积核是各项同性的。测试各项同性的卷积核的卷积操作是旋转不变的。
    filter3x3 = torch.empty((1, 1, 3, 3))
    # fill the central pixel
    filter3x3[0, 0, 1, 1] = np.random.randn()
    # fill the ring of radius 1 around it with a unique value
    mask = torch.ones(3, 3, dtype=torch.bool)
    mask[1, 1] = 0
    filter3x3[0, 0, mask] = np.random.randn()
    # 分开调整了卷积核中心的权重以及中心外的权重。中心是一个值，中心外围是另一个值。

    plt.imshow(filter3x3[0, 0].numpy())
    plt.title('Filter')
    plt.show()

    psi_x = torch.conv2d(x, filter3x3, bias=None, padding=1)
    psi_gx = torch.conv2d(gx, filter3x3, bias=None, padding=1)

    g_psi_x = rotate(psi_x, r)

    plt.imshow(g_psi_x[0, 0].numpy())
    plt.title('$g.\psi(x)$')
    plt.show()

    plt.imshow(psi_gx[0, 0].numpy())
    plt.title('$\psi(g.x)$')
    plt.show()

    assert torch.allclose(psi_gx, g_psi_x, atol=1e-6, rtol=1e-6)


def rotate_p4(y: torch.Tensor, r: int) -> torch.Tensor:
    # `y` is a function over p4, i.e. over the pixel positions and over the elements of the group C_4.
    # This method implements the action of a rotation `r` on `y`.
    # To be able to reuse this function later with a minibatch of inputs, assume that the last two dimensions (`dim=-2` and `dim=-1`) of `y` are the spatial dimensions
    # while `dim=-3` has size `4` and is the C_4 dimension.
    # All other dimensions are considered batch dimensions
    assert len(y.shape) >= 3
    assert y.shape[-3] == 4

    # BEGIN SOLUTION
    # y的形状 (B, C, 4, H, W)，其中B,C不是必须的。'4'代表C4群的4种角度，这一维取0代表 0*pi/2,取1代表 1*pi/2
    rotate_y = []

    # 考虑y的不同维数
    y_shape = y.shape
    y = y.reshape(-1, y_shape[-3], y_shape[-2], y_shape[-1])
    for i in range(y_shape[-3]):
        rotate_y.append(rotate(y[:, C4.product(C4.inverse(r), i)], r))
        # C4.product(C4.inverse(r), i) 实现了 r^(-1)s
        # rotate(y[:, :, C4.product(C4.inverse(r), i)], r) 实现了 r^(-1)p
    return torch.stack(rotate_y, dim=-3).reshape(*y_shape)
    # the shape like the input y
    # END SOLUTION


def test_rotate_p4():
    # Let's test a rotation by r=1

    y = torch.randn(1, 1, 4, 33, 33) ** 2

    ry = rotate_p4(y, 1)  # shape like y

    fig, axes = plt.subplots(1, 4, sharex=True, sharey=True, squeeze=True, figsize=(16, 4))
    for i in range(4):
        axes[i].imshow(y[0, 0, i].numpy())
    fig.suptitle('Original y')
    plt.show()

    fig, axes = plt.subplots(1, 4, sharex=True, sharey=True, squeeze=True, figsize=(16, 4))
    for i in range(4):
        axes[i].imshow(ry[0, 0, i].numpy())
    fig.suptitle('Rotated y')
    plt.show()

    # check that the images are actually rotated:
    for _ in range(10):
        p = np.random.randint(0, 33, size=2)
        s = np.random.randint(0, 4)

        # compute r^-1 s
        _rs = C4.product(C4.inverse(1), s)

        # compute r^-1 p
        # note that the rotation is around the central pixel (16, 16)
        # A rotation by r^-1 = -90 degrees maps (X, Y) -> (Y, -X)
        center = np.array([16, 16])
        # center the point
        centered_p = p - center
        # rotate round the center
        rotated_p = np.array([centered_p[1], -centered_p[0]])
        # shift the point back
        _rp = rotated_p + center

        # Finally check that [r.y](p, s) = y(r^-1 p, r^-1 s)

        # However, in a machine, an image is stored with the coordinates (H-1-Y, X) rather than the usual (X, Y), where H is the height of the image;
        # we need to take this into account
        assert torch.isclose(
            ry[..., s, 32 - p[1], p[0]],
            y[..., _rs, 32 - _rp[1], _rp[0]],
            atol=1e-5, rtol=1e-5
        )


if __name__ == '__main__':
    test_C4()

    test_D4()
