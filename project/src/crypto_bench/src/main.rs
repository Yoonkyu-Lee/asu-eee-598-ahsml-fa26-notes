//! CP6: single-thread CPU cost of the crypto a community node box runs per post.
//!
//! Build from project/:  cargo build --release --manifest-path src/crypto_bench/Cargo.toml
//! Run through src/bench_cp6.py (pins a core, adds energy). Alone it prints one JSON object;
//! every time is the median of repeated runs, in microseconds.
//!   ed25519      verify one signature; batch-verify 64 (per signature)
//!   chacha       decrypt + authenticate a 1 KB post (ChaCha20-Poly1305)
//!   pairing      BN254 and BLS12-381: one pairing, a 3-pairing product, a G1 MSM
//!   groth16      BN254 (the curve Semaphore and RLN use): verify with 5 public inputs,
//!                and prove a synthetic circuit of 2^12..2^16 constraints (squaring chain);
//!                batch-verify 64 proofs with random weights (per proof)
//! Proof verification cost does not depend on circuit size, only on the number of public inputs.
//! Proving cost scales with constraints; the synthetic circuit only stands in for the real one.

use std::time::Instant;

use ark_bls12_381::Bls12_381;
use ark_bn254::{Bn254, Fr};
use ark_ec::pairing::Pairing;
use ark_ec::{CurveGroup, VariableBaseMSM};
use ark_ff::Field;
use ark_groth16::Groth16;
use ark_relations::lc;
use ark_relations::r1cs::{ConstraintSynthesizer, ConstraintSystemRef, SynthesisError};
use ark_snark::SNARK;
use ark_std::UniformRand;
use chacha20poly1305::aead::{Aead, KeyInit};
use chacha20poly1305::{ChaCha20Poly1305, Key, Nonce};
use ed25519_dalek::{Signature, Signer, SigningKey, VerifyingKey};
use rand::rngs::OsRng;
use serde_json::{json, Value};

/// Median microseconds of `reps` calls after `warm` warm-up calls.
fn time_us<F: FnMut()>(warm: usize, reps: usize, mut f: F) -> f64 {
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

/// Batch Groth16 verification with random weights r_i (one shared final exponentiation):
/// prod e(r_i A_i, B_i) * e(-sum r_i L_i, gamma) * e(-sum r_i C_i, delta) == e(alpha, beta)^(sum r_i),
/// with L_i = IC_0 + sum_j x_ij IC_j. n + 2 Miller loops instead of 3n pairings' worth.
fn batch_verify(pvk: &ark_groth16::PreparedVerifyingKey<Bn254>, items: &[(Vec<Fr>, ark_groth16::Proof<Bn254>)]) -> bool {
    use ark_bn254::{G1Projective, G2Affine};
    use ark_ec::pairing::PairingOutput;
    let rng = &mut OsRng;
    let ic = &pvk.vk.gamma_abc_g1;
    let r: Vec<Fr> = items.iter().map(|_| Fr::rand(rng)).collect();
    let mut g1: Vec<<Bn254 as Pairing>::G1Affine> = Vec::with_capacity(items.len() + 2);
    let mut g2: Vec<G2Affine> = Vec::with_capacity(items.len() + 2);
    for ((_, pr), ri) in items.iter().zip(&r) {
        g1.push((pr.a * ri).into_affine());
        g2.push(pr.b);
    }
    // sum_i r_i L_i = (sum r_i) IC_0 + sum_j (sum_i r_i x_ij) IC_j
    let mut coeff = vec![Fr::from(0u64); ic.len()];
    for ((x, _), ri) in items.iter().zip(&r) {
        coeff[0] += ri;
        for (j, xj) in x.iter().enumerate() {
            coeff[j + 1] += *ri * xj;
        }
    }
    let l = G1Projective::msm(ic, &coeff).unwrap();
    let cs: Vec<_> = items.iter().map(|(_, pr)| pr.c).collect();
    let c = G1Projective::msm(&cs, &r).unwrap();
    g1.push((-l).into_affine());
    g2.push(pvk.vk.gamma_g2);
    g1.push((-c).into_affine());
    g2.push(pvk.vk.delta_g2);
    let lhs = Bn254::multi_pairing(g1, g2);
    let sum_r: Fr = r.iter().sum();
    lhs == PairingOutput(pvk.alpha_g1_beta_g2) * sum_r
}

fn pairing_costs<E: Pairing>(msm_sizes: &[usize]) -> Value {
    let rng = &mut OsRng;
    let g1: Vec<E::G1Affine> = (0..3).map(|_| E::G1::rand(rng).into_affine()).collect();
    let g2: Vec<E::G2Affine> = (0..3).map(|_| E::G2::rand(rng).into_affine()).collect();
    let one = time_us(5, 50, || {
        let _ = std::hint::black_box(E::pairing(g1[0], g2[0]));
    });
    let three = time_us(5, 50, || {
        let _ = std::hint::black_box(E::multi_pairing(g1.clone(), g2.clone()));
    });
    let mut msm = serde_json::Map::new();
    for &n in msm_sizes {
        let bases: Vec<E::G1Affine> = (0..n).map(|_| E::G1::rand(rng).into_affine()).collect();
        let scalars: Vec<E::ScalarField> = (0..n).map(|_| E::ScalarField::rand(rng)).collect();
        let us = time_us(5, 50, || {
            std::hint::black_box(E::G1::msm(&bases, &scalars).unwrap());
        });
        msm.insert(n.to_string(), json!(us));
    }
    json!({"pairing_us": one, "multi_pairing3_us": three, "g1_msm_us": msm})
}

/// `loop <groth16|ed25519> <seconds>`: repeat one verification for a fixed time, so an outside
/// meter (src/bench_cp6.py, package energy) can attribute joules per verification.
fn verify_loop(what: &str, secs: f64) {
    let rng = &mut OsRng;
    let msg = vec![7u8; 1024];
    let mut verify: Box<dyn FnMut()> = match what {
        "groth16" => {
            let c = Chain { publics: (0..5).map(|_| Fr::rand(rng)).collect(), x0: Fr::rand(rng), n: 1 << 12 };
            let (pk, vk) = Groth16::<Bn254>::circuit_specific_setup(c.clone(), rng).unwrap();
            let pvk = Groth16::<Bn254>::process_vk(&vk).unwrap();
            let proof = Groth16::<Bn254>::prove(&pk, c.clone(), rng).unwrap();
            Box::new(move || assert!(Groth16::<Bn254>::verify_with_processed_vk(&pvk, &c.publics, &proof).unwrap()))
        }
        _ => {
            let k = SigningKey::generate(rng);
            let (vk, sig) = (k.verifying_key(), k.sign(&msg));
            Box::new(move || vk.verify_strict(&msg, &sig).unwrap())
        }
    };
    let a = Instant::now();
    let mut n = 0u64;
    while a.elapsed().as_secs_f64() < secs {
        verify();
        n += 1;
    }
    println!("{}", json!({"what": what, "ops": n, "secs": a.elapsed().as_secs_f64()}));
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() == 4 && args[1] == "loop" {
        return verify_loop(&args[2], args[3].parse().unwrap());
    }
    let rng = &mut OsRng;
    let mut out = serde_json::Map::new();

    // Ed25519: what CP1 measured in Python (cryptography/OpenSSL), here in Rust, plus batching.
    let msg = vec![7u8; 1024];
    let keys: Vec<SigningKey> = (0..64).map(|_| SigningKey::generate(rng)).collect();
    let vks: Vec<VerifyingKey> = keys.iter().map(|k| k.verifying_key()).collect();
    let sigs: Vec<Signature> = keys.iter().map(|k| k.sign(&msg)).collect();
    let msgs: Vec<&[u8]> = (0..64).map(|_| msg.as_slice()).collect();
    let single = time_us(100, 2000, || vks[0].verify_strict(&msg, &sigs[0]).unwrap());
    let batch = time_us(10, 200, || ed25519_dalek::verify_batch(&msgs, &sigs, &vks).unwrap()) / 64.0;
    out.insert("ed25519".into(), json!({"verify_us": single, "batch64_verify_per_sig_us": batch}));

    // ChaCha20-Poly1305 on a 1 KB post.
    let cipher = ChaCha20Poly1305::new(Key::from_slice(&[1u8; 32]));
    let nonce = Nonce::from_slice(&[2u8; 12]);
    let ct = cipher.encrypt(nonce, msg.as_slice()).unwrap();
    let dec = time_us(100, 2000, || {
        std::hint::black_box(cipher.decrypt(nonce, ct.as_slice()).unwrap());
    });
    out.insert("chacha20poly1305".into(), json!({"decrypt_1KB_us": dec}));

    out.insert("bn254".into(), pairing_costs::<Bn254>(&[5, 16, 64]));
    out.insert("bls12_381".into(), pairing_costs::<Bls12_381>(&[5, 16, 64]));
    eprintln!("pairings done");

    // Groth16 on BN254.
    let mut g = serde_json::Map::new();
    for log_n in [12usize, 14, 16] {
        let c = Chain { publics: (0..5).map(|_| Fr::rand(rng)).collect(), x0: Fr::rand(rng), n: 1 << log_n };
        let (pk, vk) = Groth16::<Bn254>::circuit_specific_setup(c.clone(), rng).unwrap();
        let pvk = Groth16::<Bn254>::process_vk(&vk).unwrap();
        let reps = if log_n >= 16 { 3 } else { 7 };
        let mut proof = None;
        let prove = time_us(1, reps, || {
            proof = Some(Groth16::<Bn254>::prove(&pk, c.clone(), &mut OsRng).unwrap());
        }) / 1e3;
        let proof = proof.unwrap();
        assert!(Groth16::<Bn254>::verify_with_processed_vk(&pvk, &c.publics, &proof).unwrap());
        let verify = time_us(10, 200, || {
            std::hint::black_box(Groth16::<Bn254>::verify_with_processed_vk(&pvk, &c.publics, &proof).unwrap());
        });
        g.insert(format!("2^{log_n}"), json!({"constraints": (1usize << log_n) + 5, "prove_ms": prove, "verify_us": verify}));
        eprintln!("groth16 2^{log_n}: prove {prove:.1} ms, verify {verify:.0} us");
    }
    out.insert("groth16_bn254_5_public_inputs".into(), Value::Object(g));

    // Batch verification of 64 proofs (8 distinct proofs, each repeated 8 times; weights differ).
    let c = Chain { publics: (0..5).map(|_| Fr::rand(rng)).collect(), x0: Fr::rand(rng), n: 1 << 12 };
    let (pk, vk) = Groth16::<Bn254>::circuit_specific_setup(c.clone(), rng).unwrap();
    let pvk = Groth16::<Bn254>::process_vk(&vk).unwrap();
    let distinct: Vec<(Vec<Fr>, ark_groth16::Proof<Bn254>)> = (0..8)
        .map(|_| {
            let ci = Chain { publics: (0..5).map(|_| Fr::rand(rng)).collect(), x0: Fr::rand(rng), n: 1 << 12 };
            (ci.publics.clone(), Groth16::<Bn254>::prove(&pk, ci, rng).unwrap())
        })
        .collect();
    let items: Vec<_> = (0..64).map(|i| distinct[i % 8].clone()).collect();
    assert!(batch_verify(&pvk, &items));
    let mut bad = items.clone();
    bad[5].0[0] += Fr::from(1u64);
    assert!(!batch_verify(&pvk, &bad)); // one wrong public input must fail the whole batch
    let per = time_us(3, 30, || assert!(batch_verify(&pvk, &items))) / 64.0;
    out.insert("groth16_bn254_batch64_verify_per_proof_us".into(), json!(per));
    eprintln!("groth16 batch64: {per:.0} us per proof");
    println!("{}", serde_json::to_string_pretty(&Value::Object(out)).unwrap());
}
