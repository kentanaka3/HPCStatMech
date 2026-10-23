//! Fast Xoshiro256**-style PRNG plus a bitwise Bernoulli mask synthesizer.

#[derive(Clone)]
pub struct FastRng {
    s: [u64; 4],
}

impl FastRng {
    pub fn new(seed: u64) -> Self {
        let mut z = seed;
        let mut sm64 = || {
            z = z.wrapping_add(0x9e3779b97f4a7c15);
            let mut x = z;
            x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
            x ^ (x >> 31)
        };
        Self { s: [sm64(), sm64(), sm64(), sm64()] }
    }

    #[inline(always)]
    pub fn next_u64(&mut self) -> u64 {
        let res = self.s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        res
    }

    /// 64-bit mask where each bit is 1 with probability `num / 2^K`
    /// (Horner's binary probability scheme, at most K RNG draws).
    #[inline(always)]
    pub fn bernoulli_mask<const K: u32>(&mut self, num: u32) -> u64 {
        if num == 0 {
            return 0;
        }
        if num >= (1 << K) {
            return u64::MAX;
        }
        let first_bit = num.trailing_zeros();
        let mut acc = self.next_u64();
        for bit_idx in (first_bit + 1)..K {
            let r = self.next_u64();
            if ((num >> bit_idx) & 1) == 1 {
                acc |= r;
            } else {
                acc &= r;
            }
        }
        acc
    }
}
