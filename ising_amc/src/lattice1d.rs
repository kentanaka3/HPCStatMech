//! 1D lattice, 64 replicas per word: Glauber spin-flip and Kawasaki spin-exchange.
use crate::bitmat::replica_magnetizations;
use crate::boltzmann::BoltzmannTable;
use crate::rng::FastRng;

/// Glauber spin-flip for centre spin `s` with neighbours `n_left`, `n_right`.
#[inline(always)]
pub fn isa_1d_glauber_site(
    s: u64,
    n_left: u64,
    n_right: u64,
    table: &BoltzmannTable,
    rng: &mut FastRng,
) -> u64 {
    let tie = n_left ^ n_right;
    let favorable_flip = !tie & (s ^ n_left);
    let unfavorable_flip = !tie & !(s ^ n_left);
    let rand_tie = rng.next_u64();
    let thermal_mask = if table.is_t0 { 0 } else { rng.bernoulli_mask::<10>(table.p_de_plus4) };
    s ^ (favorable_flip | (tie & rand_tie) | (unfavorable_flip & thermal_mask))
}

/// Kawasaki exchange of adjacent spins; conserves popcount per replica.
#[inline(always)]
pub fn isa_1d_kawasaki_bond(
    s_l: u64,
    s_r: u64,
    ext_l: u64,
    ext_r: u64,
    table: &BoltzmannTable,
    rng: &mut FastRng,
) -> (u64, u64) {
    let can_swap = s_l ^ s_r;
    let agree_l = !(s_l ^ ext_l);
    let agree_r = !(s_r ^ ext_r);
    let de_neg = !agree_l & !agree_r;
    let de_zero = agree_l ^ agree_r;
    let de_pos = agree_l & agree_r;
    let rand_tie = rng.next_u64();
    let thermal_mask = if table.is_t0 { 0 } else { rng.bernoulli_mask::<10>(table.p_de_plus4) };
    let do_swap = can_swap & (de_neg | (de_zero & rand_tie) | (de_pos & thermal_mask));
    (s_l ^ do_swap, s_r ^ do_swap)
}

pub struct Lattice1DAMC {
    pub l: usize,
    pub spins: Vec<u64>,
    pub spins_t0: Vec<u64>,
    pub persist: Vec<u64>,
    pub rng: FastRng,
}

impl Lattice1DAMC {
    pub fn new(l: usize, seed: u64) -> Self {
        assert!(l % 6 == 0, "L must be a multiple of 6 (2-sublattice Glauber, 3-sublattice Kawasaki).");
        let mut rng = FastRng::new(seed);
        let spins: Vec<u64> = (0..l).map(|_| rng.next_u64()).collect();
        let spins_t0 = spins.clone();
        let persist = vec![u64::MAX; l];
        Self { l, spins, spins_t0, persist, rng }
    }

    pub fn sweep_glauber(&mut self, table: &BoltzmannTable) {
        let l = self.l;
        for sub in 0..2 {
            for i in (sub..l).step_by(2) {
                let left = if i == 0 { l - 1 } else { i - 1 };
                let right = if i + 1 == l { 0 } else { i + 1 };
                let s_new = isa_1d_glauber_site(
                    self.spins[i], self.spins[left], self.spins[right], table, &mut self.rng,
                );
                self.spins[i] = s_new;
                self.persist[i] &= !(s_new ^ self.spins_t0[i]);
            }
        }
    }

    /// Stride 3 guarantees bond (i, i+1) and its external neighbours never overlap.
    pub fn sweep_kawasaki(&mut self, table: &BoltzmannTable) {
        let l = self.l;
        for sub in 0..3 {
            for i in (sub..l).step_by(3) {
                let i_r = if i + 1 == l { 0 } else { i + 1 };
                let ext_l = if i == 0 { l - 1 } else { i - 1 };
                let ext_r = if i_r + 1 == l { 0 } else { i_r + 1 };
                let (new_l, new_r) = isa_1d_kawasaki_bond(
                    self.spins[i], self.spins[i_r], self.spins[ext_l], self.spins[ext_r],
                    table, &mut self.rng,
                );
                self.spins[i] = new_l;
                self.spins[i_r] = new_r;
                self.persist[i] &= !(new_l ^ self.spins_t0[i]);
                self.persist[i_r] &= !(new_r ^ self.spins_t0[i_r]);
            }
        }
    }

    pub fn mean_persistence(&self) -> f64 {
        let ones: u64 = self.persist.iter().map(|&w| w.count_ones() as u64).sum();
        ones as f64 / (self.l * 64) as f64
    }

    pub fn ensemble_abs_magnetization(&self) -> f64 {
        let m_r = replica_magnetizations(&self.spins, self.l);
        m_r.iter().map(|m| m.abs()).sum::<f64>() / 64.0
    }
}
