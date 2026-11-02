//! CP23: what a weight limit costs when it is enforced in the RLN proof.
//!
//! Build and run from project/:
//!   cargo run --release --manifest-path src/crypto_bench/Cargo.toml --bin cp23 -- results/cp23/rln_n.json
//! RLN-n: one message spends n consecutive message indices k..k+n-1 < limit and publishes, for each, the share
//! y_i = a0 + a1_i x and the nullifier H(a1_i). Circuit (BN254, Groth16):
//!   commitment H(a0, limit) -> depth-20 Merkle path -> root (public)
//!   pre = H(a0, external nullifier)   a1_i = H(pre, k + i)   y_i, nullifier_i (public)
//!   k and limit - k - n in 16 bits
//! Poseidon t = 3, alpha = 5, 8 full + 57 partial rounds, written by hand on ark-relations; the round constants and
//! the MDS matrix are seeded stand-ins (not the audited ones), so the constraint count is right but the hash is not
//! a standard one. Single thread. Times are medians in microseconds.

use std::time::Instant;

use ark_bn254::{Bn254, Fr};
use ark_ff::{Field, PrimeField, Zero};
use ark_groth16::Groth16;
use ark_relations::lc;
use ark_relations::r1cs::{
    ConstraintSynthesizer, ConstraintSystem, ConstraintSystemRef, LinearCombination, SynthesisError, Variable,
};
use ark_serialize::CanonicalSerialize;
use ark_snark::SNARK;
use ark_std::UniformRand;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde_json::{json, Value};

const T: usize = 3;
const RF: usize = 8;
const RP: usize = 57;
const DEPTH: usize = 20;
const RANGE_BITS: usize = 16;

type LC = LinearCombination<Fr>;

struct Poseidon {
    c: Vec<[Fr; T]>,
    m: [[Fr; T]; T],
}

impl Poseidon {
    fn new() -> Self {
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

    fn full(r: usize) -> bool {
        r < RF / 2 || r >= RF / 2 + RP
    }

    fn hash(&self, a: Fr, b: Fr) -> Fr {
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
struct Sig {
    lc: LC,
    v: Fr,
}

fn var(v: Variable, val: Fr) -> Sig {
    Sig { lc: lc!() + v, v: val }
}

fn konst(c: Fr) -> Sig {
    Sig { lc: lc!() + (c, Variable::One), v: c }
}

fn add(a: &Sig, b: &Sig) -> Sig {
    let mut l = LinearCombination([a.lc.0.clone(), b.lc.0.clone()].concat());
    l.compactify();
    Sig { lc: l, v: a.v + b.v }
}

fn scale(a: &Sig, f: Fr) -> Sig {
    Sig { lc: LinearCombination(a.lc.0.iter().map(|(c, v)| (*c * f, *v)).collect()), v: a.v * f }
}

fn sub(a: &Sig, b: &Sig) -> Sig {
    add(a, &scale(b, -Fr::from(1u64)))
}

fn witness(cs: &ConstraintSystemRef<Fr>, v: Fr) -> Result<Sig, SynthesisError> {
    Ok(var(cs.new_witness_variable(|| Ok(v))?, v))
}

fn mul(cs: &ConstraintSystemRef<Fr>, a: &Sig, b: &Sig) -> Result<Sig, SynthesisError> {
    let p = witness(cs, a.v * b.v)?;
    cs.enforce_constraint(a.lc.clone(), b.lc.clone(), p.lc.clone())?;
    Ok(p)
}

fn equal(cs: &ConstraintSystemRef<Fr>, a: &Sig, b: &Sig) -> Result<(), SynthesisError> {
    cs.enforce_constraint(a.lc.clone(), lc!() + Variable::One, b.lc.clone())
}

impl Poseidon {
    fn gadget(&self, cs: &ConstraintSystemRef<Fr>, a: &Sig, b: &Sig) -> Result<Sig, SynthesisError> {
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

fn range(cs: &ConstraintSystemRef<Fr>, a: &Sig, bits: usize) -> Result<(), SynthesisError> {
    let x = a.v.into_bigint().as_ref()[0];
    let mut sum = konst(Fr::zero());
    for j in 0..bits {
        let b = witness(cs, Fr::from((x >> j) & 1))?;
        cs.enforce_constraint(b.lc.clone(), lc!() + Variable::One - b.lc.clone(), lc!())?;
        sum = add(&sum, &scale(&b, Fr::from(1u64 << j)));
    }
    equal(cs, &sum, a)
}

#[derive(Clone)]
struct RlnN<'a> {
    h: &'a Poseidon,
    n: usize,
    a0: Fr,
    limit: u64,
    k: u64,
    path: Vec<(Fr, bool)>,
    ext: Fr,
    x: Fr,
}

impl RlnN<'_> {
    /// The public inputs, computed natively: root, external nullifier, x, then (y_i, nullifier_i).
    fn publics(&self) -> Vec<Fr> {
        let h = self.h;
        let mut cur = h.hash(self.a0, Fr::from(self.limit));
        for (s, b) in &self.path {
            cur = if *b { h.hash(*s, cur) } else { h.hash(cur, *s) };
        }
        let pre = h.hash(self.a0, self.ext);
        let mut out = vec![cur, self.ext, self.x];
        for i in 0..self.n {
            let a1 = h.hash(pre, Fr::from(self.k + i as u64));
            out.push(self.a0 + a1 * self.x);
            out.push(h.hash(a1, Fr::zero()));
        }
        out
    }
}

impl ConstraintSynthesizer<Fr> for RlnN<'_> {
    fn generate_constraints(self, cs: ConstraintSystemRef<Fr>) -> Result<(), SynthesisError> {
        let p = self.publics();
        let pubs: Vec<Sig> = p.iter().map(|v| Ok(var(cs.new_input_variable(|| Ok(*v))?, *v))).collect::<Result<_, _>>()?;
        let (root, ext, x) = (&pubs[0], &pubs[1], &pubs[2]);
        let h = self.h;
        let a0 = witness(&cs, self.a0)?;
        let limit = witness(&cs, Fr::from(self.limit))?;
        let mut cur = h.gadget(&cs, &a0, &limit)?;
        for (s, b) in &self.path {
            let s = witness(&cs, *s)?;
            let b = witness(&cs, Fr::from(*b as u64))?;
            cs.enforce_constraint(b.lc.clone(), lc!() + Variable::One - b.lc.clone(), lc!())?;
            let t = mul(&cs, &b, &sub(&s, &cur))?; // b (s - cur)
            let (l, r) = (add(&cur, &t), sub(&s, &t)); // b = 0: (cur, s); b = 1: (s, cur)
            cur = h.gadget(&cs, &l, &r)?;
        }
        equal(&cs, &cur, root)?;
        let k = witness(&cs, Fr::from(self.k))?;
        range(&cs, &k, RANGE_BITS)?;
        range(&cs, &sub(&sub(&limit, &k), &konst(Fr::from(self.n as u64))), RANGE_BITS)?;
        let pre = h.gadget(&cs, &a0, ext)?;
        for i in 0..self.n {
            let idx = add(&k, &konst(Fr::from(i as u64)));
            let a1 = h.gadget(&cs, &pre, &idx)?;
            let (y, nul) = (&pubs[3 + 2 * i], &pubs[4 + 2 * i]);
            cs.enforce_constraint(a1.lc.clone(), x.lc.clone(), sub(y, &a0).lc)?; // a1 x = y - a0
            let nv = h.gadget(&cs, &a1, &konst(Fr::zero()))?;
            equal(&cs, &nv, nul)?;
        }
        Ok(())
    }
}

fn median_us<F: FnMut()>(warm: usize, reps: usize, mut f: F) -> f64 {
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

fn main() {
    let out = std::env::args().nth(1).unwrap_or_else(|| "results/cp23/rln_n.json".into());
    let h = Poseidon::new();
    let mut rng = StdRng::seed_from_u64(2301);
    let mut rows = Vec::new();
    for n in [1usize, 2, 3, 4, 6, 8, 12, 16, 24] {
        let c = RlnN {
            h: &h,
            n,
            a0: Fr::rand(&mut rng),
            limit: 64,
            k: 5,
            path: (0..DEPTH).map(|_| (Fr::rand(&mut rng), rng.gen::<bool>())).collect(),
            ext: Fr::rand(&mut rng),
            x: Fr::rand(&mut rng),
        };
        let cs = ConstraintSystem::<Fr>::new_ref();
        c.clone().generate_constraints(cs.clone()).unwrap();
        assert!(cs.is_satisfied().unwrap(), "n = {n} not satisfied");
        let constraints = cs.num_constraints();
        let publics = c.publics();

        let a = Instant::now();
        let (pk, vk) = Groth16::<Bn254>::circuit_specific_setup(c.clone(), &mut rng).unwrap();
        let setup_s = a.elapsed().as_secs_f64();
        let pvk = Groth16::<Bn254>::process_vk(&vk).unwrap();
        let proof = Groth16::<Bn254>::prove(&pk, c.clone(), &mut rng).unwrap();
        assert!(Groth16::<Bn254>::verify_with_processed_vk(&pvk, &publics, &proof).unwrap());
        let mut bad = publics.clone();
        bad[3] += Fr::from(1u64);
        assert!(!Groth16::<Bn254>::verify_with_processed_vk(&pvk, &bad, &proof).unwrap());

        let reps = if constraints > 12_000 { 3 } else { 5 };
        let prove_us = median_us(1, reps, || {
            std::hint::black_box(Groth16::<Bn254>::prove(&pk, c.clone(), &mut StdRng::seed_from_u64(1)).unwrap());
        });
        let verify_us = median_us(20, 300, || {
            std::hint::black_box(Groth16::<Bn254>::verify_with_processed_vk(&pvk, &publics, &proof).unwrap());
        });
        let proof_bytes = proof.compressed_size();
        let row = json!({"n": n, "public_inputs": publics.len(), "constraints": constraints, "setup_s": setup_s,
                         "prove_us": prove_us, "verify_us": verify_us, "proof_bytes": proof_bytes,
                         "publics_bytes": 32 * publics.len()});
        eprintln!("{row}");
        rows.push(row);
    }
    let doc: Value = json!({"bench": "cp23 RLN-n (arkworks 0.5, BN254 Groth16, single thread)",
                            "poseidon": {"t": T, "full_rounds": RF, "partial_rounds": RP, "alpha": 5,
                                         "constants": "seeded stand-ins (constraint count only)"},
                            "merkle_depth": DEPTH, "range_bits": RANGE_BITS, "rows": rows});
    if let Some(dir) = std::path::Path::new(&out).parent() {
        std::fs::create_dir_all(dir).unwrap();
    }
    std::fs::write(&out, serde_json::to_string_pretty(&doc).unwrap()).unwrap();
    eprintln!("wrote {out}");
}
