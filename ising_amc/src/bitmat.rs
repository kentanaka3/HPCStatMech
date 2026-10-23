//! 64x64 bit-matrix transpose for replica-resolved measurements.

/// In-place transpose: input `a[site]` holds replica bits; output `a[replica]` holds site bits.
#[inline]
pub fn bmat_transpose_64x64(a: &mut [u64; 64]) {
    const MASKS: [u64; 6] = [
        0x5555555555555555,
        0x3333333333333333,
        0x0F0F0F0F0F0F0F0F,
        0x00FF00FF00FF00FF,
        0x0000FFFF0000FFFF,
        0x00000000FFFFFFFF,
    ];
    for stage in (0..6).rev() {
        let step = 1usize << stage;
        let shift = 1u32 << stage;
        let m = MASKS[stage];
        let inv_m = !m;
        let mut i = 0;
        while i < 64 {
            for j in 0..step {
                let u = a[i + j];
                let v = a[i + j + step];
                a[i + j] = (u & m) | ((v & m) << shift);
                a[i + j + step] = ((u & inv_m) >> shift) | (v & inv_m);
            }
            i += step * 2;
        }
    }
}

/// Magnetization of each of the 64 replicas over all words of `words`
/// (`total_sites` = number of sites, i.e. number of words).
pub fn replica_magnetizations(words: &[u64], total_sites: usize) -> [f64; 64] {
    let mut up_counts = [0u64; 64];
    let mut accumulate = |block: &mut [u64; 64]| {
        bmat_transpose_64x64(block);
        for r in 0..64 {
            up_counts[r] += block[r].count_ones() as u64;
        }
    };
    let mut chunks = words.chunks_exact(64);
    for chunk in &mut chunks {
        let mut block = [0u64; 64];
        block.copy_from_slice(chunk);
        accumulate(&mut block);
    }
    let rem = chunks.remainder();
    if !rem.is_empty() {
        let mut block = [0u64; 64];
        block[..rem.len()].copy_from_slice(rem);
        accumulate(&mut block);
    }
    let n = total_sites as f64;
    let mut m_r = [0.0f64; 64];
    for r in 0..64 {
        m_r[r] = (2.0 * (up_counts[r] as f64) - n) / n;
    }
    m_r
}
