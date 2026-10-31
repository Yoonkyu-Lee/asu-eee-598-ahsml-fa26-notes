"""CP11: BN254 arithmetic that records every base-field operation, so a proof verification can be replayed
on a cycle model of the verification unit (src/sim_cp11.py).

The formulas follow arkworks 0.5 (the library measured in CP6-CP8): tower Fp2 = Fp[u]/(u^2+1),
Fp6 = Fp2[v]/(v^3 - (9+u)), Fp12 = Fp6[w]/(w^2 - v); homogeneous-projective Miller loop for a D-type twist
with sparse line multiplication (mul_by_034); final exponentiation with the same hard-part chain.
Two simplifications, both on the conservative side (more work, not less): cyclotomic squarings are plain
Fp12 squarings, and multiplications by constants are counted as full multiplications.

Every Fp value carries the id of the operation that produced it (-1 for inputs and constants).
trace = [(kind, out_id, in_id_a, in_id_b)], kind 'm' (multiply or square) or 'a' (add, sub, neg, double).
Run this file to self-test (bilinearity, subgroup check, decompression).
"""
from __future__ import annotations

P = 21888242871839275222246405745257275088696311157297823662689037894645226208583
R = 21888242871839275222246405745257275088548364400416034343698204186575808495617
X = 4965661367192848881  # BN parameter, positive for BN254

TRACE: list = []
_on = [False]


def tracing(flag: bool) -> None:
    _on[0] = flag


def _rec(kind: str, a: "Fp", b: "Fp | None") -> int:
    if not _on[0]:
        return -1
    i = len(TRACE)
    TRACE.append((kind, i, a.id, -1 if b is None else b.id))
    return i


class Fp:
    __slots__ = ("v", "id")

    def __init__(self, v: int, id_: int = -1):
        self.v = v % P
        self.id = id_

    def __add__(self, o): return Fp(self.v + o.v, _rec("a", self, o))
    def __sub__(self, o): return Fp(self.v - o.v, _rec("a", self, o))
    def __neg__(self): return Fp(-self.v, _rec("a", self, None))
    def __mul__(self, o): return Fp(self.v * o.v, _rec("m", self, o))
    def sq(self): return Fp(self.v * self.v, _rec("m", self, self))
    def dbl(self): return Fp(2 * self.v, _rec("a", self, None))
    def __eq__(self, o): return self.v == o.v
    def is_zero(self): return self.v == 0

    def pow(self, e: int) -> "Fp":
        r, base, first = None, self, True
        for bit in bin(e)[2:]:
            if not first:
                r = r.sq()
            if bit == "1":
                r = base if first else r * base
            first = False
        return r

    def inv(self) -> "Fp": return self.pow(P - 2)
    def sqrt(self) -> "Fp": return self.pow((P + 1) // 4)  # p = 3 mod 4


def C(v: int) -> Fp:
    """A constant: no producing operation."""
    return Fp(v)


class Fp2:
    __slots__ = ("c0", "c1")

    def __init__(self, c0: Fp, c1: Fp): self.c0, self.c1 = c0, c1
    @staticmethod
    def const(a: int, b: int = 0): return Fp2(C(a), C(b))
    def __add__(self, o): return Fp2(self.c0 + o.c0, self.c1 + o.c1)
    def __sub__(self, o): return Fp2(self.c0 - o.c0, self.c1 - o.c1)
    def __neg__(self): return Fp2(-self.c0, -self.c1)
    def dbl(self): return Fp2(self.c0.dbl(), self.c1.dbl())

    def __mul__(self, o):  # Karatsuba, 3 M
        v0, v1 = self.c0 * o.c0, self.c1 * o.c1
        return Fp2(v0 - v1, (self.c0 + self.c1) * (o.c0 + o.c1) - v0 - v1)

    def sq(self):  # 2 M
        a = (self.c0 + self.c1) * (self.c0 - self.c1)
        b = self.c0 * self.c1
        return Fp2(a, b.dbl())

    def mul_fp(self, f: Fp): return Fp2(self.c0 * f, self.c1 * f)
    def conj(self): return Fp2(self.c0, -self.c1)
    def mul_xi(self):  # (9 + u) * (c0 + c1 u) = (9c0 - c1) + (c0 + 9c1) u, as additions
        n9 = lambda a: (a.dbl().dbl().dbl() + a)
        return Fp2(n9(self.c0) - self.c1, self.c0 + n9(self.c1))
    def is_zero(self): return self.c0.is_zero() and self.c1.is_zero()
    def __eq__(self, o): return self.c0 == o.c0 and self.c1 == o.c1

    def inv(self):
        t = (self.c0.sq() + self.c1.sq()).inv()
        return Fp2(self.c0 * t, -(self.c1 * t))

    def pow(self, e: int):
        r, first = None, True
        for bit in bin(e)[2:]:
            if not first:
                r = r.sq()
            if bit == "1":
                r = self if first else r * self
            first = False
        return r

    def sqrt(self):
        """Fp2 square root (complex method); assumes a square."""
        a0, a1 = self.c0, self.c1
        if a1.is_zero():
            s = a0.sqrt()
            if s.sq() == a0:
                return Fp2(s, C(0))
            return Fp2(C(0), (-a0).sqrt())
        alpha = a0.sq() + a1.sq()
        gamma = alpha.sqrt()
        half = C((P + 1) // 2)
        delta = (a0 + gamma) * half
        x0 = delta.sqrt()
        if not (x0.sq() == delta):
            delta = (a0 - gamma) * half
            x0 = delta.sqrt()
        x1 = a1 * (x0.dbl()).inv()
        return Fp2(x0, x1)


XI = (9, 1)


def _fp2_pow_const(a, b, e):
    """(a + b u)^e on plain integers, for Frobenius constants."""
    def mul(x, y):
        return ((x[0] * y[0] - x[1] * y[1]) % P, (x[0] * y[1] + x[1] * y[0]) % P)
    r, base = (1, 0), (a, b)
    while e:
        if e & 1:
            r = mul(r, base)
        base = mul(base, base)
        e >>= 1
    return r


# Frobenius coefficients: FROB6_C1[k] = xi^((p^k-1)/3), FROB6_C2[k] = xi^(2(p^k-1)/3), FROB12_C1[k] = xi^((p^k-1)/6)
FROB6_C1 = [Fp2.const(*_fp2_pow_const(*XI, (P ** k - 1) // 3)) for k in range(4)]
FROB6_C2 = [Fp2.const(*_fp2_pow_const(*XI, 2 * (P ** k - 1) // 3)) for k in range(4)]
FROB12_C1 = [Fp2.const(*_fp2_pow_const(*XI, (P ** k - 1) // 6)) for k in range(4)]
TWIST_Q_X = Fp2.const(*_fp2_pow_const(*XI, (P - 1) // 3))
TWIST_Q_Y = Fp2.const(*_fp2_pow_const(*XI, (P - 1) // 2))


class Fp6:
    __slots__ = ("c0", "c1", "c2")

    def __init__(self, c0, c1, c2): self.c0, self.c1, self.c2 = c0, c1, c2
    @staticmethod
    def zero(): return Fp6(Fp2.const(0), Fp2.const(0), Fp2.const(0))
    @staticmethod
    def one(): return Fp6(Fp2.const(1), Fp2.const(0), Fp2.const(0))
    def __add__(self, o): return Fp6(self.c0 + o.c0, self.c1 + o.c1, self.c2 + o.c2)
    def __sub__(self, o): return Fp6(self.c0 - o.c0, self.c1 - o.c1, self.c2 - o.c2)
    def __neg__(self): return Fp6(-self.c0, -self.c1, -self.c2)

    def __mul__(self, o):  # 6 Fp2 multiplications (Karatsuba-style, as arkworks)
        a_a, b_b, c_c = self.c0 * o.c0, self.c1 * o.c1, self.c2 * o.c2
        t1 = ((self.c1 + self.c2) * (o.c1 + o.c2) - b_b - c_c).mul_xi() + a_a
        t2 = (self.c0 + self.c1) * (o.c0 + o.c1) - a_a - b_b + c_c.mul_xi()
        t3 = (self.c0 + self.c2) * (o.c0 + o.c2) - a_a + b_b - c_c
        return Fp6(t1, t2, t3)

    def sq(self): return self * self
    def mul_by_v(self): return Fp6(self.c2.mul_xi(), self.c0, self.c1)

    def mul_by_01(self, b0, b1):  # sparse: (b0 + b1 v)
        a_a, b_b = self.c0 * b0, self.c1 * b1
        t1 = ((self.c1 + self.c2) * b1 - b_b).mul_xi() + a_a
        t2 = (b0 + b1) * (self.c0 + self.c1) - a_a - b_b
        t3 = (self.c0 + self.c2) * b0 - a_a + b_b
        return Fp6(t1, t2, t3)

    def mul_by_1(self, b1):
        b_b = self.c1 * b1
        return Fp6((self.c2 * b1).mul_xi(), (self.c0 * b1), b_b)

    def frob(self, k):
        c0, c1, c2 = self.c0, self.c1, self.c2
        if k % 2:
            c0, c1, c2 = c0.conj(), c1.conj(), c2.conj()
        return Fp6(c0, c1 * FROB6_C1[k], c2 * FROB6_C2[k])

    def inv(self):
        c0 = self.c0.sq() - (self.c1 * self.c2).mul_xi()
        c1 = self.c2.sq().mul_xi() - self.c0 * self.c1
        c2 = self.c1.sq() - self.c0 * self.c2
        t = ((self.c2 * c1 + self.c1 * c2).mul_xi() + self.c0 * c0).inv()
        return Fp6(t * c0, t * c1, t * c2)

    def __eq__(self, o): return self.c0 == o.c0 and self.c1 == o.c1 and self.c2 == o.c2


class Fp12:
    __slots__ = ("c0", "c1")

    def __init__(self, c0, c1): self.c0, self.c1 = c0, c1
    @staticmethod
    def one(): return Fp12(Fp6.one(), Fp6.zero())

    def __mul__(self, o):  # 3 Fp6 multiplications
        aa, bb = self.c0 * o.c0, self.c1 * o.c1
        c1 = (self.c0 + self.c1) * (o.c0 + o.c1) - aa - bb
        return Fp12(aa + bb.mul_by_v(), c1)

    def sq(self):  # complex squaring, 2 Fp6 multiplications
        v0 = self.c0 - self.c1
        v3 = self.c0 - self.c1.mul_by_v()
        v2 = self.c0 * self.c1
        v0 = v0 * v3 + v2
        c1 = v2 + v2
        c0 = v0 + v2.mul_by_v()
        return Fp12(c0, c1)

    def mul_by_034(self, c0, c3, c4):  # sparse line for a D-type twist (arkworks)
        a = Fp6(self.c0.c0 * c0, self.c0.c1 * c0, self.c0.c2 * c0)
        b = self.c1.mul_by_01(c3, c4)
        e = (self.c0 + self.c1).mul_by_01(c0 + c3, c4)
        return Fp12(b.mul_by_v() + a, e - (a + b))

    def conj(self): return Fp12(self.c0, -self.c1)  # = inverse in the cyclotomic subgroup

    def frob(self, k):
        c1 = self.c1.frob(k)
        return Fp12(self.c0.frob(k), Fp6(c1.c0 * FROB12_C1[k], c1.c1 * FROB12_C1[k], c1.c2 * FROB12_C1[k]))

    def inv(self):
        t = (self.c0.sq() - self.c1.sq().mul_by_v()).inv()
        return Fp12(self.c0 * t, -(self.c1 * t))

    def exp_cyc(self, e):  # square-and-multiply; cyclotomic squaring counted as a plain squaring
        r, first = None, True
        for bit in bin(e)[2:]:
            if not first:
                r = r.sq()
            if bit == "1":
                r = self if first else r * self
            first = False
        return r

    def __eq__(self, o): return self.c0 == o.c0 and self.c1 == o.c1


# ------------------------------------------------------------------ curves

B1 = C(3)
B2 = None  # 3 / (9 + u), filled below


def _init_b2():
    global B2
    inv_xi = Fp2.const(9, 1).inv()
    B2 = Fp2(C(3 * inv_xi.c0.v), C(3 * inv_xi.c1.v))


_init_b2()
G1_GEN = (C(1), C(2))
G2_GEN = (Fp2.const(10857046999023057135944570762232829481370756359578518086990519993285655852781,
                    11559732032986387107991004021392285783925812861821192530917403151452391805634),
          Fp2.const(8495653923123431417604973247489272438418190587263600148770280649306958101930,
                    4082367875863433681332203403145435568316851327593401208105741076214120093531))


class Jac:
    """Jacobian point over F (Fp or Fp2) on y^2 = x^3 + b (a = 0). None = infinity."""

    def __init__(self, x, y, z): self.x, self.y, self.z = x, y, z

    def dbl(self):  # dbl-2009-l: 2M + 5S
        a, b = self.x.sq(), self.y.sq()
        c = b.sq()
        d = ((self.x + b).sq() - a - c).dbl()
        e = a.dbl() + a
        f = e.sq()
        x3 = f - d.dbl()
        y3 = e * (d - x3) - c.dbl().dbl().dbl()
        z3 = (self.y * self.z).dbl()
        return Jac(x3, y3, z3)

    def add_affine(self, qx, qy):  # madd-2007-bl: 7M + 4S (no special cases: used on generic points)
        z1z1 = self.z.sq()
        u2, s2 = qx * z1z1, qy * self.z * z1z1
        h = u2 - self.x
        hh = h.sq()
        i = hh.dbl().dbl()
        j = h * i
        r = (s2 - self.y).dbl()
        v = self.x * i
        x3 = r.sq() - j - v.dbl()
        y3 = r * (v - x3) - (self.y * j).dbl()
        z3 = (self.z + h).sq() - z1z1 - hh
        return Jac(x3, y3, z3)

    def to_affine(self):
        zi = self.z.inv()
        zi2 = zi.sq()
        return self.x * zi2, self.y * zi2 * zi


def one_like(v):
    return C(1) if isinstance(v, Fp) else Fp2.const(1)


def scalar_mul(pt, k: int):
    """Left-to-right double-and-add from an affine point (k > 0). Returns a Jac point."""
    x, y = pt
    acc = Jac(x, y, one_like(x))
    for bit in bin(k)[3:]:
        acc = acc.dbl()
        if bit == "1":
            acc = acc.add_affine(x, y)
    return acc


def naf(k: int) -> list[int]:
    out = []
    while k:
        if k & 1:
            d = 2 - (k % 4)
            k -= d
        else:
            d = 0
        out.append(d)
        k //= 2
    return out  # least significant first


ATE = naf(6 * X + 2)


# ------------------------------------------------------------------ pairing (arkworks BN, D-type twist)

TWO_INV = C((P + 1) // 2)


def _doubling(r):
    rx, ry, rz = r
    a = (rx * ry).mul_fp(TWO_INV)
    b, c = ry.sq(), rz.sq()
    e = B2 * (c.dbl() + c)
    f = e.dbl() + e
    g = (b + f).mul_fp(TWO_INV)
    h = (ry + rz).sq() - (b + c)
    i = e - b
    j = rx.sq()
    e2 = e.sq()
    nr = (a * (b - f), g.sq() - (e2.dbl() + e2), b * h)
    return nr, (-h, j.dbl() + j, i)


def _addition(r, q):
    rx, ry, rz = r
    qx, qy = q
    theta = ry - qy * rz
    lam = rx - qx * rz
    c, d = theta.sq(), lam.sq()
    e = lam * d
    f, g = rz * c, rx * d
    h = e + f - g.dbl()
    nr = (lam * h, theta * (g - h) - e * ry, rz * e)
    j = theta * qx - lam * qy
    return nr, (lam, -theta, j)


def _frob_point(q):
    return (q[0].conj() * TWIST_Q_X, q[1].conj() * TWIST_Q_Y)


def prepare(q):
    """Line coefficients for a G2 point (affine), in loop order."""
    r = (q[0], q[1], Fp2.const(1))
    nq = (q[0], -q[1])
    coeffs = []
    for bit in reversed(ATE[:-1]):
        r, c = _doubling(r)
        coeffs.append(c)
        if bit == 1:
            r, c = _addition(r, q)
            coeffs.append(c)
        elif bit == -1:
            r, c = _addition(r, nq)
            coeffs.append(c)
    q1 = _frob_point(q)
    q2 = _frob_point(q1)
    q2 = (q2[0], -q2[1])
    r, c = _addition(r, q1)
    coeffs.append(c)
    r, c = _addition(r, q2)
    coeffs.append(c)
    return coeffs


def _ell(f, coeff, p):
    c0, c1, c2 = coeff
    return f.mul_by_034(c0.mul_fp(p[1]), c1.mul_fp(p[0]), c2)


def miller_loop(pairs):
    """pairs: [(G1 affine p, G2 line coefficients)]. Shared squaring, one line per pair per step."""
    its = [iter(c) for _, c in pairs]
    f = Fp12.one()
    bits = list(reversed(ATE[:-1]))
    for idx, bit in enumerate(bits):
        if idx:
            f = f.sq()
        for (p, _), it in zip(pairs, its):
            f = _ell(f, next(it), p)
        if bit:
            for (p, _), it in zip(pairs, its):
                f = _ell(f, next(it), p)
    for _ in range(2):
        for (p, _), it in zip(pairs, its):
            f = _ell(f, next(it), p)
    return f


def miller_loop_otf(fresh_pairs, prepared_pairs):
    """Same result as miller_loop, but lines for the fresh G2 points are computed in the loop, right before
    they are used (as a streaming unit would), instead of preparing every line first.
    fresh_pairs: [(G1 affine p, G2 affine q)]; prepared_pairs: [(G1 affine p, line coefficients)]."""
    st = [[p, (q[0], q[1], Fp2.const(1)), q, (q[0], -q[1])] for p, q in fresh_pairs]
    its = [iter(c) for _, c in prepared_pairs]
    f = Fp12.one()
    bits = list(reversed(ATE[:-1]))

    def step(kind):
        nonlocal f
        for s_ in st:
            if kind == "dbl":
                s_[1], c = _doubling(s_[1])
            else:
                s_[1], c = _addition(s_[1], s_[2] if kind == 1 else s_[3] if kind == -1 else kind(s_))
            f = _ell(f, c, s_[0])
        for (p, _), it in zip(prepared_pairs, its):
            f = _ell(f, next(it), p)

    for idx, bit in enumerate(bits):
        if idx:
            f = f.sq()
        step("dbl")
        if bit:
            step(bit)
    for s_ in st:
        q1 = _frob_point(s_[2])
        q2 = _frob_point(q1)
        s_.append(q1)
        s_.append((q2[0], -q2[1]))
    step(lambda s_: s_[4])
    step(lambda s_: s_[5])
    return f


def _exp_by_neg_x(f):
    return f.exp_cyc(X).conj()  # x > 0 for BN254


def final_exp(f):
    f1 = f.conj() * f.inv()           # easy part: f^(p^6 - 1)
    r = f1.frob(2) * f1               # ^(p^2 + 1)
    y0 = _exp_by_neg_x(r)
    y1 = y0.sq()
    y2 = y1.sq()
    y3 = y2 * y1
    y4 = _exp_by_neg_x(y3)
    y5 = y4.sq()
    y6 = _exp_by_neg_x(y5)
    y3, y6 = y3.conj(), y6.conj()
    y7 = y6 * y4
    y8 = y7 * y3
    y9 = y8 * y1
    y10 = y8 * y4
    y11 = y10 * r
    y13 = y9.frob(1) * y11
    y14 = y8.frob(2) * y13
    y15 = (r.conj() * y9).frob(3)
    return y15 * y14


def pairing(p, q):
    return final_exp(miller_loop([(p, prepare(q))]))


# ------------------------------------------------------------------ point decoding and checks

def g1_decompress(x: Fp):
    return x, (x.sq() * x + B1).sqrt()


def g2_decompress(x: Fp2):
    return x, (x.sq() * x + B2).sqrt()


def g2_in_subgroup(q) -> bool:
    """psi(Q) == [6x^2] Q (El Housni, Guillevic, Piellard 2022, BN curves)."""
    lhs = _frob_point(q)
    rhs = scalar_mul(q, 6 * X * X).to_affine()
    return lhs[0] == rhs[0] and lhs[1] == rhs[1]


def g1_on_curve(p) -> bool:
    return p[1].sq() == p[0].sq() * p[0] + B1


# ------------------------------------------------------------------ self-test

def _selftest() -> None:
    assert g1_on_curve(G1_GEN)
    gx, gy = G2_GEN
    assert gy.sq() == gx.sq() * gx + B2, "G2 generator not on the twist"
    assert g2_in_subgroup(G2_GEN), "subgroup check rejects the generator"
    # a twist point outside G2: decompress some x until it is on the curve, then test
    t = 5
    while True:
        xs = Fp2.const(t, 1)
        rhs = xs.sq() * xs + B2
        y = rhs.sqrt()
        if y.sq() == rhs:
            break
        t += 1
    assert not g2_in_subgroup((xs, y)), "subgroup check accepts a random twist point"
    p2 = scalar_mul(G1_GEN, 2).to_affine()
    q2 = scalar_mul(G2_GEN, 2).to_affine()
    e = pairing(G1_GEN, G2_GEN)
    assert not (e == Fp12.one()), "degenerate pairing"
    assert pairing(p2, G2_GEN) == pairing(G1_GEN, q2) == e * e, "bilinearity fails"
    x, y = g1_decompress(G1_GEN[0])
    assert y == G1_GEN[1] or y == -G1_GEN[1]
    print("self-test passed: bilinear, non-degenerate; G2 subgroup check and decompression work")


if __name__ == "__main__":
    _selftest()
