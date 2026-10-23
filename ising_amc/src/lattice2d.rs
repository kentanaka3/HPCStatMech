//! Checkerboard-decomposed 2D lattice, 64 replicas per site, T = 0 Glauber dynamics.
use crate::rng::FastRng;

/// 64-replica bitwise adder for 4 neighbours. Returns (maj_up, tie) masks
/// (sum >= 3, sum == 2).
#[inline(always)]
pub fn sum_4_neighbors(n0: u64, n1: u64, n2: u64, n3: u64) -> (u64, u64) {
    let s0 = n0 ^ n1;
    let c0 = n0 & n1;
    let s1 = n2 ^ n3;
    let c1 = n2 & n3;
    let bit0 = s0 ^ s1;
    let c_mid = s0 & s1;
    let bit1 = c0 ^ c1 ^ c_mid;
    let bit2 = (c0 & c1) | (c_mid & (c0 ^ c1));
    let maj_up = bit2 | (bit1 & bit0);
    let tie = bit1 & !bit0;
    (maj_up, tie)
}

/// L must be even. Each sublattice has L * (L / 2) words.
pub struct IsingAMC2D {
    pub l: usize,
    pub half_l: usize,
    pub black: Vec<u64>,
    pub white: Vec<u64>,
    pub black_t0: Vec<u64>,
    pub white_t0: Vec<u64>,
    pub persist_black: Vec<u64>,
    pub persist_white: Vec<u64>,
    pub rng: FastRng,
}

impl IsingAMC2D {
    pub fn new(l: usize, seed: u64) -> Self {
        assert!(l % 2 == 0, "Lattice dimension L must be even for checkerboard decomposition.");
        let half_l = l / 2;
        let sub_size = l * half_l;
        let mut rng = FastRng::new(seed);
        let black: Vec<u64> = (0..sub_size).map(|_| rng.next_u64()).collect();
        let white: Vec<u64> = (0..sub_size).map(|_| rng.next_u64()).collect();
        Self {
            l,
            half_l,
            black_t0: black.clone(),
            white_t0: white.clone(),
            black,
            white,
            persist_black: vec![u64::MAX; sub_size],
            persist_white: vec![u64::MAX; sub_size],
            rng,
        }
    }

    /// One full sweep (Black then White) for all 64 replicas, updating persistence.
    pub fn sweep_t0_glauber(&mut self) {
        let (l, half_l) = (self.l, self.half_l);
        for (update_black, y_parity_left) in [(true, 0usize), (false, 1usize)] {
            let (tgt, tgt_t0, persist, src) = if update_black {
                (&mut self.black, &self.black_t0, &mut self.persist_black, &self.white)
            } else {
                (&mut self.white, &self.white_t0, &mut self.persist_white, &self.black)
            };
            for y in 0..l {
                let y_up = if y == 0 { l - 1 } else { y - 1 };
                let y_dn = if y + 1 == l { 0 } else { y + 1 };
                let parity = y & 1;
                let row = y * half_l..(y + 1) * half_l;
                let row_t = &mut tgt[row.clone()];
                let row_t0 = &tgt_t0[row.clone()];
                let row_p = &mut persist[row.clone()];
                let r_curr = &src[row];
                let r_up = &src[y_up * half_l..(y_up + 1) * half_l];
                let r_dn = &src[y_dn * half_l..(y_dn + 1) * half_l];
                for x in 0..half_l {
                    let x_other = if parity == y_parity_left {
                        if x == 0 { half_l - 1 } else { x - 1 }
                    } else if x + 1 == half_l {
                        0
                    } else {
                        x + 1
                    };
                    let (maj_up, tie) = sum_4_neighbors(r_up[x], r_dn[x], r_curr[x], r_curr[x_other]);
                    let s_new = maj_up | (tie & self.rng.next_u64());
                    row_t[x] = s_new;
                    row_p[x] &= !(s_new ^ row_t0[x]);
                }
            }
        }
    }

    pub fn mean_persistence(&self) -> f64 {
        let n: u64 = self
            .persist_black
            .iter()
            .chain(self.persist_white.iter())
            .map(|&w| w.count_ones() as u64)
            .sum();
        n as f64 / (self.l * self.l * 64) as f64
    }

    pub fn mean_magnetization(&self) -> f64 {
        let up: u64 = self
            .black
            .iter()
            .chain(self.white.iter())
            .map(|&w| w.count_ones() as u64)
            .sum();
        let total = (self.l * self.l * 64) as f64;
        (2.0 * up as f64 - total) / total
    }

    /// Equal-time correlation G(r) at horizontal distance r = 2 * dx_sub.
    pub fn correlation_even_x(&self, dx_sub: usize) -> f64 {
        let half_l = self.half_l;
        let mut diff_bits: u64 = 0;
        for y in 0..self.l {
            let row_b = &self.black[y * half_l..(y + 1) * half_l];
            let row_w = &self.white[y * half_l..(y + 1) * half_l];
            for x in 0..half_l {
                let x_r = (x + dx_sub) % half_l;
                diff_bits += (row_b[x] ^ row_b[x_r]).count_ones() as u64;
                diff_bits += (row_w[x] ^ row_w[x_r]).count_ones() as u64;
            }
        }
        1.0 - (2.0 * diff_bits as f64) / (self.l * self.l * 64) as f64
    }
}
