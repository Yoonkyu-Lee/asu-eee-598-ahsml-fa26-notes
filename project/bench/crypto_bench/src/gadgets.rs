//! Shared R1CS pieces for the crypto benches (CP23 RLN-n, CP37 reputation): a hand-written Poseidon (t = 3, alpha = 5,
//! 8 full + 57 partial rounds; seeded stand-in constants, so the constraint count is right but the hash is not a
//! standard one), linear-combination helpers, a bit range check and a median timer.

use std::time::Instant;

use ark_bn254::Fr;
use ark_ff::{Field, PrimeField, Zero};
use ark_relations::lc;
use ark_relations::r1cs::{ConstraintSystemRef, LinearCombination, SynthesisError, Variable};
use ark_std::UniformRand;
use rand::rngs::StdRng;
use rand::SeedableRng;

pub const T: usize = 3;
pub const RF: usize = 8;
pub const RP: usize = 57;

pub type LC = LinearCombination<Fr>;

pub struct Poseidon {
    pub c: Vec<[Fr; T]>,
    pub m: [[Fr; T]; T],
}

impl Poseidon {
    pub fn new() -> Self {
        let mut rng = StdRng::seed_from_u64(23);
        let c = (0..RF + RP).map(|_| [Fr::rand(&mut rng), Fr::rand(&mut rng), Fr::rand(&mut rng)]).collect();
        // Cauchy matrix 1 / (x_i + y_j), x = (0, 1, 2), y = (3, 4, 5)
        let mut m = [[Fr::zero(); T]; T];
        for (i, row) in m.iter_mut().enumerate() {
            for (j, e) in row.iter_mut().enumerate() {
                *e = Fr::from((i + T + j) as u64).inverse().unwrap();
            }
        }
        Poseidon { c, m }
    }

    pub fn full(r: usize) -> bool {
        r < RF / 2 || r >= RF / 2 + RP
    }

    pub fn hash(&self, a: Fr, b: Fr) -> Fr {
        let mut s = [Fr::zero(), a, b];
        for r in 0..RF + RP {
            for i in 0..T {
                s[i] += self.c[r][i];
            }
            for i in 0..if Self::full(r) { T } else { 1 } {
                s[i] = s[i].pow([5u64]);
            }
            let mut n = [Fr::zero(); T];
            for (i, ni) in n.iter_mut().enumerate() {
                for j in 0..T {
                    *ni += self.m[i][j] * s[j];
                }
            }
            s = n;
        }
        s[0]
    }
}

/// A linear combination together with its value.
#[derive(Clone)]
pub struct Sig {
    pub lc: LC,
    pub v: Fr,
}

pub fn var(v: Variable, val: Fr) -> Sig {
    Sig { lc: lc!() + v, v: val }
}

pub fn konst(c: Fr) -> Sig {
    Sig { lc: lc!() + (c, Variable::One), v: c }
}

pub fn add(a: &Sig, b: &Sig) -> Sig {
    let mut l = LinearCombination([a.lc.0.clone(), b.lc.0.clone()].concat());
    l.compactify();
    Sig { lc: l, v: a.v + b.v }
}

pub fn scale(a: &Sig, f: Fr) -> Sig {
    Sig { lc: LinearCombination(a.lc.0.iter().map(|(c, v)| (*c * f, *v)).collect()), v: a.v * f }
}

pub fn sub(a: &Sig, b: &Sig) -> Sig {
    add(a, &scale(b, -Fr::from(1u64)))
}

pub fn witness(cs: &ConstraintSystemRef<Fr>, v: Fr) -> Result<Sig, SynthesisError> {
    Ok(var(cs.new_witness_variable(|| Ok(v))?, v))
}

pub fn mul(cs: &ConstraintSystemRef<Fr>, a: &Sig, b: &Sig) -> Result<Sig, SynthesisError> {
    let p = witness(cs, a.v * b.v)?;
    cs.enforce_constraint(a.lc.clone(), b.lc.clone(), p.lc.clone())?;
    Ok(p)
}

pub fn equal(cs: &ConstraintSystemRef<Fr>, a: &Sig, b: &Sig) -> Result<(), SynthesisError> {
    cs.enforce_constraint(a.lc.clone(), lc!() + Variable::One, b.lc.clone())
}

impl Poseidon {
    pub fn gadget(&self, cs: &ConstraintSystemRef<Fr>, a: &Sig, b: &Sig) -> Result<Sig, SynthesisError> {
        let mut s = [konst(Fr::zero()), a.clone(), b.clone()];
        for r in 0..RF + RP {
            for i in 0..T {
                s[i] = add(&s[i], &konst(self.c[r][i]));
            }
            for i in 0..if Self::full(r) { T } else { 1 } {
                let x2 = mul(cs, &s[i], &s[i])?;
                let x4 = mul(cs, &x2, &x2)?;
                s[i] = mul(cs, &x4, &s[i])?;
            }
            let n: Vec<Sig> = (0..T)
                .map(|i| (0..T).fold(konst(Fr::zero()), |acc, j| add(&acc, &scale(&s[j], self.m[i][j]))))
                .collect();
            s = [n[0].clone(), n[1].clone(), n[2].clone()];
        }
        Ok(s[0].clone())
    }
}

pub fn range(cs: &ConstraintSystemRef<Fr>, a: &Sig, bits: usize) -> Result<(), SynthesisError> {
    let x = a.v.into_bigint().as_ref()[0];
    let mut sum = konst(Fr::zero());
    for j in 0..bits {
        let b = witness(cs, Fr::from((x >> j) & 1))?;
        cs.enforce_constraint(b.lc.clone(), lc!() + Variable::One - b.lc.clone(), lc!())?;
        sum = add(&sum, &scale(&b, Fr::from(1u64 << j)));
    }
    equal(cs, &sum, a)
}

pub fn median_us<F: FnMut()>(warm: usize, reps: usize, mut f: F) -> f64 {
    for _ in 0..warm {
        f();
    }
    let mut t: Vec<f64> = (0..reps)
        .map(|_| {
            let a = Instant::now();
            f();
            a.elapsed().as_secs_f64() * 1e6
        })
        .collect();
    t.sort_by(|a, b| a.partial_cmp(b).unwrap());
    t[t.len() / 2]
}
