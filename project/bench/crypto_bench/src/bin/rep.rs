//! CP37 (1): what an anonymous reputation proof costs, "I hold at least N receipts", under design (2) where every box
//! receives every public post (journal/cp/CP37.md, rules written before the run).
//!
//! Build and run from project/:
//!   cargo run --release --manifest-path bench/crypto_bench/Cargo.toml --bin rep -- results/chip/cp37_rep.json
//! REP-N (BN254, Groth16): receipt leaf j = H(a0, r_j) in a depth-20 receipt tree (public root); N Merkle paths to that
//! root; leaf positions strictly increasing (20-bit range check on the gaps), so no receipt counts twice; public tag
//! = H(a0, external nullifier) binds the claim to the same identity secret as RLN. Single thread. Times are medians.
//! Also re-measures the CP6 squaring chain 2^16 prove, to relate this PC to the laptop numbers (CP6, CP34), and
//! splits a prove of each size into its parts: the G1 MSMs (A, B in G1, C, H), the G2 MSM (B) and the seven Fr
//! FFTs (R1CS to QAP), each timed alone on random scalars, so the chip's share (G1 MSMs) can be weighed.

use std::collections::HashMap;
use std::time::Instant;

use ark_bn254::{Bn254, Fr, G1Projective, G2Projective};
use ark_ec::{CurveGroup, PrimeGroup, VariableBaseMSM};
use ark_ff::{Field, Zero};
use ark_poly::{EvaluationDomain, Radix2EvaluationDomain};
use ark_groth16::Groth16;
use ark_relations::lc;
use ark_relations::r1cs::{ConstraintSynthesizer, ConstraintSystem, ConstraintSystemRef, SynthesisError};
use ark_serialize::CanonicalSerialize;
use ark_snark::SNARK;
use ark_std::UniformRand;
use rand::rngs::StdRng;
use rand::{seq::index::sample, SeedableRng};
use serde_json::{json, Value};

#[path = "../gadgets.rs"]
mod gadgets;
use gadgets::*;

const DEPTH: usize = 20;

/// A sparse receipt tree: only the given leaves are set, every other leaf is zero.
fn sparse_tree(h: &Poseidon, leaves: &[(usize, Fr)]) -> (Fr, Vec<Vec<(Fr, bool)>>) {
    let mut empty = vec![Fr::zero()];
    for i in 0..DEPTH {
        empty.push(h.hash(empty[i], empty[i]));
    }
    let mut level: HashMap<usize, Fr> = leaves.iter().cloned().collect();
    let mut levels = vec![level.clone()];
    for i in 0..DEPTH {
        let mut up = HashMap::new();
        for &idx in level.keys() {
            let p = idx >> 1;
            if up.contains_key(&p) {
                continue;
            }
            let l = *level.get(&(2 * p)).unwrap_or(&empty[i]);
            let r = *level.get(&(2 * p + 1)).unwrap_or(&empty[i]);
            up.insert(p, h.hash(l, r));
        }
        level = up;
        levels.push(level.clone());
    }
    let root = levels[DEPTH][&0];
    let paths = leaves
        .iter()
        .map(|&(pos, _)| {
            (0..DEPTH)
                .map(|i| {
                    let idx = pos >> i;
                    (*levels[i].get(&(idx ^ 1)).unwrap_or(&empty[i]), idx & 1 == 1)
                })
                .collect()
        })
        .collect();
    (root, paths)
}

#[derive(Clone)]
struct Rep<'a> {
    h: &'a Poseidon,
    a0: Fr,
    ext: Fr,
    r: Vec<Fr>,
    paths: Vec<Vec<(Fr, bool)>>,
    root: Fr,
}

impl Rep<'_> {
    fn publics(&self) -> Vec<Fr> {
        vec![self.root, self.ext, self.h.hash(self.a0, self.ext)]
    }
}

impl ConstraintSynthesizer<Fr> for Rep<'_> {
    fn generate_constraints(self, cs: ConstraintSystemRef<Fr>) -> Result<(), SynthesisError> {
        let p = self.publics();
        let pubs: Vec<Sig> = p.iter().map(|v| Ok(var(cs.new_input_variable(|| Ok(*v))?, *v))).collect::<Result<_, _>>()?;
        let (root, ext, tag) = (&pubs[0], &pubs[1], &pubs[2]);
        let h = self.h;
        let a0 = witness(&cs, self.a0)?;
        let mut prev: Option<Sig> = None;
        for (r, path) in self.r.iter().zip(&self.paths) {
            let r = witness(&cs, *r)?;
            let mut cur = h.gadget(&cs, &a0, &r)?;
            let mut pos = konst(Fr::zero());
            for (i, (s, b)) in path.iter().enumerate() {
                let s = witness(&cs, *s)?;
                let b = witness(&cs, Fr::from(*b as u64))?;
                cs.enforce_constraint(b.lc.clone(), lc!() + ark_relations::r1cs::Variable::One - b.lc.clone(), lc!())?;
                pos = add(&pos, &scale(&b, Fr::from(1u64 << i)));
                let t = mul(&cs, &b, &sub(&s, &cur))?; // b (s - cur)
                let (l, rr) = (add(&cur, &t), sub(&s, &t)); // b = 0: (cur, s); b = 1: (s, cur)
                cur = h.gadget(&cs, &l, &rr)?;
            }
            equal(&cs, &cur, root)?;
            if let Some(q) = prev {
                range(&cs, &sub(&sub(&pos, &q), &konst(Fr::from(1u64))), DEPTH)?; // pos_j > pos_{j-1}
            }
            prev = Some(pos);
        }
        let t = h.gadget(&cs, &a0, ext)?;
        equal(&cs, &t, tag)
    }
}

fn rep_instance<'a>(h: &'a Poseidon, n: usize, rng: &mut StdRng) -> Rep<'a> {
    let a0 = Fr::rand(rng);
    let mut pos: Vec<usize> = sample(rng, 1 << DEPTH, n).into_iter().collect();
    pos.sort();
    let r: Vec<Fr> = (0..n).map(|_| Fr::rand(rng)).collect();
    let leaves: Vec<(usize, Fr)> = pos.iter().zip(&r).map(|(&p, &rj)| (p, h.hash(a0, rj))).collect();
    let (root, paths) = sparse_tree(h, &leaves);
    Rep { h, a0, ext: Fr::rand(rng), r, paths, root }
}

/// CP6's synthetic circuit: n squarings and 5 public inputs.
#[derive(Clone)]
struct Chain {
    publics: Vec<Fr>,
    x0: Fr,
    n: usize,
}

impl ConstraintSynthesizer<Fr> for Chain {
    fn generate_constraints(self, cs: ConstraintSystemRef<Fr>) -> Result<(), SynthesisError> {
        for p in &self.publics {
            let v = cs.new_input_variable(|| Ok(*p))?;
            cs.enforce_constraint(lc!() + v, lc!() + ark_relations::r1cs::Variable::One, lc!() + v)?;
        }
        let mut x = self.x0;
        let mut xv = cs.new_witness_variable(|| Ok(x))?;
        for _ in 0..self.n {
            let y = x.square();
            let yv = cs.new_witness_variable(|| Ok(y))?;
            cs.enforce_constraint(lc!() + xv, lc!() + xv, lc!() + yv)?;
            x = y;
            xv = yv;
        }
        Ok(())
    }
}

/// Seconds for one G1 MSM and one G2 MSM of size m, and for seven FFTs over a domain of size dom.
fn split(m: usize, dom: usize, rng: &mut StdRng) -> (f64, f64, f64) {
    let scalars: Vec<Fr> = (0..m).map(|_| Fr::rand(rng)).collect();
    let (mut g1, mut g2) = (Vec::with_capacity(m), Vec::with_capacity(m));
    let (mut a, mut b) = (G1Projective::generator() * Fr::rand(rng), G2Projective::generator() * Fr::rand(rng));
    let (da, db) = (G1Projective::generator() * Fr::rand(rng), G2Projective::generator() * Fr::rand(rng));
    for _ in 0..m {
        g1.push(a);
        g2.push(b);
        a += da;
        b += db;
    }
    let (g1, g2) = (G1Projective::normalize_batch(&g1), G2Projective::normalize_batch(&g2));
    let reps = if m > 100_000 { 1 } else { 3 };
    let t1 = median_us(0, reps, || {
        let _ = std::hint::black_box(G1Projective::msm(&g1, &scalars).unwrap());
    }) / 1e6;
    let t2 = median_us(0, reps, || {
        let _ = std::hint::black_box(G2Projective::msm(&g2, &scalars).unwrap());
    }) / 1e6;
    let d = Radix2EvaluationDomain::<Fr>::new(dom).unwrap();
    let mut v: Vec<Fr> = (0..dom).map(|_| Fr::rand(rng)).collect();
    let tf = median_us(0, reps, || d.fft_in_place(&mut v)) / 1e6;
    (t1, t2, 7.0 * tf)
}

fn main() {
    let out = std::env::args().nth(1).unwrap_or_else(|| "results/chip/cp37_rep.json".into());
    let mut rng = StdRng::seed_from_u64(3701);

    let c = Chain { publics: (0..5).map(|_| Fr::rand(&mut rng)).collect(), x0: Fr::rand(&mut rng), n: 1 << 16 };
    let (pk, _) = Groth16::<Bn254>::circuit_specific_setup(c.clone(), &mut rng).unwrap();
    let chain_ms = median_us(1, 3, || {
        std::hint::black_box(Groth16::<Bn254>::prove(&pk, c.clone(), &mut StdRng::seed_from_u64(1)).unwrap());
    }) / 1e3;
    eprintln!("chain 2^16 prove {chain_ms:.1} ms (CP6 laptop P-core 3436 ms)");

    let h = Poseidon::new();
    let mut rows = Vec::new();
    for n in [1usize, 10, 100] {
        let c = rep_instance(&h, n, &mut rng);
        let cs = ConstraintSystem::<Fr>::new_ref();
        c.clone().generate_constraints(cs.clone()).unwrap();
        assert!(cs.is_satisfied().unwrap(), "N = {n} not satisfied");
        let (constraints, witnesses) = (cs.num_constraints(), cs.num_witness_variables());
        let publics = c.publics();

        let a = Instant::now();
        let (pk, vk) = Groth16::<Bn254>::circuit_specific_setup(c.clone(), &mut rng).unwrap();
        let setup_s = a.elapsed().as_secs_f64();
        let pvk = Groth16::<Bn254>::process_vk(&vk).unwrap();
        let proof = Groth16::<Bn254>::prove(&pk, c.clone(), &mut rng).unwrap();
        assert!(Groth16::<Bn254>::verify_with_processed_vk(&pvk, &publics, &proof).unwrap());
        let mut bad = publics.clone();
        bad[2] += Fr::from(1u64);
        assert!(!Groth16::<Bn254>::verify_with_processed_vk(&pvk, &bad, &proof).unwrap());

        let reps = if n >= 100 { 1 } else { 3 };
        let prove_us = median_us(if n >= 100 { 0 } else { 1 }, reps, || {
            std::hint::black_box(Groth16::<Bn254>::prove(&pk, c.clone(), &mut StdRng::seed_from_u64(1)).unwrap());
        });
        let verify_us = median_us(20, 300, || {
            std::hint::black_box(Groth16::<Bn254>::verify_with_processed_vk(&pvk, &publics, &proof).unwrap());
        });
        let dom = (constraints + publics.len()).next_power_of_two();
        let (g1_s, g2_s, fft_s) = split(witnesses + publics.len() + 1, dom, &mut rng);
        let row = json!({"n": n, "constraints": constraints, "witnesses": witnesses, "public_inputs": publics.len(),
                         "split_s": {"g1_msm_one": g1_s, "g1_msms_four": 4.0 * g1_s, "g2_msm": g2_s, "fft_seven": fft_s},
                         "setup_s": setup_s, "prove_us": prove_us, "verify_us": verify_us,
                         "proof_bytes": proof.compressed_size(), "publics_bytes": 32 * publics.len(),
                         "pk_bytes": pk.compressed_size()});
        eprintln!("{row}");
        rows.push(row);
    }
    let doc: Value = json!({"bench": "cp37 REP-N (arkworks 0.5, BN254 Groth16, single thread)",
                            "machine": "second PC (Core Ultra 9 185H, IT13 Max), Windows best-performance overlay",
                            "label": "measured (this PC); Pi 5 by ratio is an estimate",
                            "poseidon": {"t": 3, "full_rounds": 8, "partial_rounds": 57, "alpha": 5,
                                         "constants": "seeded stand-ins (constraint count only)"},
                            "merkle_depth": DEPTH, "chain_2_16_prove_ms": chain_ms, "cp6_laptop_chain_2_16_prove_ms": 3436.3,
                            "rows": rows});
    if let Some(dir) = std::path::Path::new(&out).parent() {
        std::fs::create_dir_all(dir).unwrap();
    }
    std::fs::write(&out, serde_json::to_string_pretty(&doc).unwrap()).unwrap();
    eprintln!("wrote {out}");
}
