mod bitmat;
mod boltzmann;
mod lattice1d;
mod lattice2d;
mod rng;

use boltzmann::BoltzmannTable;
use lattice1d::Lattice1DAMC;
use lattice2d::IsingAMC2D;
use std::time::Instant;

fn bench_2d() {
    let l = 256;
    let sweeps = 500;
    let mut sim = IsingAMC2D::new(l, 42);
    println!("=== 2D Asynchronous Multispin Ising Quench (64 Replicas/Word) ===");
    println!("Lattice: {}x{} ({} spins x 64 replicas)", l, l, l * l);
    println!("Step 0   | P(0)   = {:.5} | <m> = {:+.5} | G(2) = {:+.5}",
             sim.mean_persistence(), sim.mean_magnetization(), sim.correlation_even_x(1));
    let start = Instant::now();
    for step in 1..=sweeps {
        sim.sweep_t0_glauber();
        if step == 10 || step == 100 || step == sweeps {
            println!("Step {:<3} | P({:<3}) = {:.5} | <m> = {:+.5} | G(2) = {:+.5}",
                     step, step, sim.mean_persistence(), sim.mean_magnetization(),
                     sim.correlation_even_x(1));
        }
    }
    let elapsed = start.elapsed().as_secs_f64();
    let updates = (l * l * 64 * sweeps) as f64;
    println!("Elapsed: {:.4} s | Throughput: {:.2} B spin updates/s", elapsed, updates / elapsed / 1e9);
}

fn bench_1d() {
    let l = 3840;
    let sweeps = 2000;
    let table_t0 = BoltzmannTable::new(0.0);
    let table_finite = BoltzmannTable::new(1.2);
    let mut glauber = Lattice1DAMC::new(l, 777);
    let mut kawasaki = Lattice1DAMC::new(l, 777);
    println!("=== 1D Modular AMC: Glauber vs Conserved Kawasaki (64 Replicas) ===");
    println!("Initial <|m|>: Glauber = {:.5}, Kawasaki = {:.5}",
             glauber.ensemble_abs_magnetization(), kawasaki.ensemble_abs_magnetization());
    let t0 = Instant::now();
    for _ in 0..sweeps { glauber.sweep_glauber(&table_t0); }
    let dt_g = t0.elapsed().as_secs_f64();
    let t1 = Instant::now();
    for _ in 0..sweeps { kawasaki.sweep_kawasaki(&table_finite); }
    let dt_k = t1.elapsed().as_secs_f64();
    let n = (l * 64 * sweeps) as f64;
    println!("After {} sweeps:", sweeps);
    println!("  [Glauber  T=0.0] P(t) = {:.5} | <|m|> = {:.5} | {:.2} B updates/s",
             glauber.mean_persistence(), glauber.ensemble_abs_magnetization(), n / dt_g / 1e9);
    println!("  [Kawasaki T=1.2] P(t) = {:.5} | <|m|> = {:.5} (conserved) | {:.2} B bond-updates/s",
             kawasaki.mean_persistence(), kawasaki.ensemble_abs_magnetization(), n / dt_k / 1e9);
}

fn main() {
    bench_2d();
    println!();
    bench_1d();
}

#[cfg(test)]
mod tests {
    use super::bitmat::*;
    use super::lattice1d::*;
    use super::lattice2d::*;
    use super::rng::FastRng;

    #[test]
    fn test_bitwise_adder_all_16_truth_table_states() {
        let (mut n0, mut n1, mut n2, mut n3) = (0u64, 0u64, 0u64, 0u64);
        for idx in 0..16 {
            if idx & 1 != 0 { n0 |= 1 << idx; }
            if idx & 2 != 0 { n1 |= 1 << idx; }
            if idx & 4 != 0 { n2 |= 1 << idx; }
            if idx & 8 != 0 { n3 |= 1 << idx; }
        }
        let (maj_up, tie) = sum_4_neighbors(n0, n1, n2, n3);
        for idx in 0..16u32 {
            let sum = idx.count_ones();
            assert_eq!(((maj_up >> idx) & 1) == 1, sum >= 3, "maj_up bit {}", idx);
            assert_eq!(((tie >> idx) & 1) == 1, sum == 2, "tie bit {}", idx);
        }
    }

    #[test]
    fn test_bmat_transpose_involution_and_lanes() {
        let mut rng = FastRng::new(12345);
        let mut orig = [0u64; 64];
        for w in orig.iter_mut() { *w = rng.next_u64(); }
        let mut mat = orig;
        bmat_transpose_64x64(&mut mat);
        for i in 0..64 {
            for r in 0..64 {
                assert_eq!((orig[i] >> r) & 1, (mat[r] >> i) & 1, "site {}, replica {}", i, r);
            }
        }
        bmat_transpose_64x64(&mut mat);
        assert_eq!(orig, mat);
    }

    #[test]
    fn test_kawasaki_strict_conservation_across_all_64_replicas() {
        let l = 384;
        let mut sim = Lattice1DAMC::new(l, 999);
        let m_initial = replica_magnetizations(&sim.spins, l);
        let table = super::boltzmann::BoltzmannTable::new(1.5);
        for _ in 0..200 { sim.sweep_kawasaki(&table); }
        let m_final = replica_magnetizations(&sim.spins, l);
        for r in 0..64 {
            assert!((m_initial[r] - m_final[r]).abs() < 1e-15, "replica {} not conserved", r);
        }
    }

    #[test]
    fn test_2d_persistence_decreases() {
        let mut sim = IsingAMC2D::new(32, 1);
        let p0 = sim.mean_persistence();
        for _ in 0..10 { sim.sweep_t0_glauber(); }
        assert!(p0 == 1.0 && sim.mean_persistence() < 1.0);
    }
}
