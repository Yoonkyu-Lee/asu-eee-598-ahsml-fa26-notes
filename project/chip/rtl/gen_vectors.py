"""CP9: reference vectors for rtl/mont_mul.sv (Python 3.6+, no packages; also runs on Apporto).

  python3 gen_vectors.py [count] [out]      default 1000 vectors to vectors.hex
Each line: a b y in hex (64 digits each), y = a * b * 2^-256 mod p, the Montgomery product that
arkworks' Fq multiplication computes on Montgomery-form operands. Edge cases come first.
Also prints NP = -p^-1 mod 2^256, the constant in the RTL.
"""
import random
import sys

P = 21888242871839275222246405745257275088696311157297823662689037894645226208583  # BN254 base field
R = 1 << 256


def inv(x, m):
    """Modular inverse by extended Euclid (pow(x, -1, m) needs Python 3.8)."""
    a, b, u, v = x % m, m, 1, 0
    while b:
        q = a // b
        a, b, u, v = b, a - q * b, v, u - q * v
    assert a == 1
    return u % m


NP = (-inv(P, R)) % R
R_INV = inv(R, P)


def mont(a, b):
    return a * b * R_INV % P


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    out = sys.argv[2] if len(sys.argv) > 2 else "vectors.hex"
    rng = random.Random(598)
    edge = [(0, 0), (0, P - 1), (1, 1), (P - 1, P - 1), (P - 1, 1), (R % P, R % P), (P - 2, P - 3)]
    pairs = edge + [(rng.randrange(P), rng.randrange(P)) for _ in range(n - len(edge))]
    with open(out, "w") as f:
        for a, b in pairs:
            f.write("%064x %064x %064x\n" % (a, b, mont(a, b)))
    print("P  = %064x" % P)
    print("NP = %064x" % NP)
    print("wrote %d vectors to %s" % (len(pairs), out))


if __name__ == "__main__":
    main()
