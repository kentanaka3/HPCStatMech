//! Dyadic Boltzmann probabilities (K = 10 -> 1/1024 precision).

pub struct BoltzmannTable {
    /// exp(-4J/T) scaled by 1024
    pub p_de_plus4: u32,
    pub is_t0: bool,
}

impl BoltzmannTable {
    pub fn new(temp: f64) -> Self {
        if temp <= 1e-9 {
            Self { p_de_plus4: 0, is_t0: true }
        } else {
            let p = (-4.0 / temp).exp();
            let scaled = (p * 1024.0).round().clamp(0.0, 1024.0) as u32;
            Self { p_de_plus4: scaled, is_t0: false }
        }
    }
}
