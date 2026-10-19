use num_bigint::BigUint;
use num_complex::Complex;
use rayon::join;
use rustfft::FftPlanner;
use std::env;
use std::fs::File;
use std::io::{self, Write};

const LIMB_BITS: u32 = 15;
const BASE: u64 = 1 << LIMB_BITS;
const SCHOOLBOOK_LIMIT: usize = 96;
const LEAF_TERMS: u32 = 32;
const PARALLEL_CUTOFF: u32 = 256;

#[derive(Clone, Debug, Eq, PartialEq)]
struct BigNat {
    // Little-endian base-2^15 limbs.
    limbs: Vec<u32>,
}

impl BigNat {
    fn zero() -> Self {
        Self { limbs: Vec::new() }
    }

    fn one() -> Self {
        Self { limbs: vec![1] }
    }

    #[cfg(test)]
    fn from_u32(value: u32) -> Self {
        if value == 0 {
            return Self::zero();
        }
        let mut value = value;
        let mut limbs = Vec::new();
        while value != 0 {
            limbs.push(value & (BASE as u32 - 1));
            value >>= LIMB_BITS;
        }
        Self { limbs }
    }

    fn normalize(&mut self) {
        while self.limbs.last() == Some(&0) {
            self.limbs.pop();
        }
    }

    fn add_assign(&mut self, other: &Self) {
        let length = self.limbs.len().max(other.limbs.len());
        self.limbs.resize(length, 0);
        let mut carry = 0u64;
        for i in 0..length {
            let sum =
                self.limbs[i] as u64 + other.limbs.get(i).copied().unwrap_or(0) as u64 + carry;
            self.limbs[i] = (sum % BASE) as u32;
            carry = sum / BASE;
        }
        if carry != 0 {
            self.limbs.push(carry as u32);
        }
    }

    fn mul_small(&self, factor: u32) -> Self {
        if factor == 0 || self.limbs.is_empty() {
            return Self::zero();
        }
        let mut result = Vec::with_capacity(self.limbs.len() + 1);
        let mut carry = 0u64;
        for &limb in &self.limbs {
            let value = limb as u64 * factor as u64 + carry;
            result.push((value % BASE) as u32);
            carry = value / BASE;
        }
        if carry != 0 {
            result.push(carry as u32);
        }
        Self { limbs: result }
    }

    fn mul(&self, other: &Self) -> Self {
        if self.limbs.is_empty() || other.limbs.is_empty() {
            return Self::zero();
        }
        if self.limbs.len().min(other.limbs.len()) < SCHOOLBOOK_LIMIT {
            return self.mul_schoolbook(other);
        }
        self.mul_fft(other)
    }

    fn mul_schoolbook(&self, other: &Self) -> Self {
        let mut coefficients = vec![0u64; self.limbs.len() + other.limbs.len()];
        for (i, &left) in self.limbs.iter().enumerate() {
            for (j, &right) in other.limbs.iter().enumerate() {
                coefficients[i + j] += left as u64 * right as u64;
            }
        }
        Self::from_coefficients(coefficients)
    }

    fn mul_fft(&self, other: &Self) -> Self {
        let length = (self.limbs.len() + other.limbs.len()).next_power_of_two();
        let mut left = vec![Complex::new(0.0, 0.0); length];
        let mut right = vec![Complex::new(0.0, 0.0); length];
        for (i, &limb) in self.limbs.iter().enumerate() {
            left[i].re = limb as f64;
        }
        for (i, &limb) in other.limbs.iter().enumerate() {
            right[i].re = limb as f64;
        }

        let mut planner = FftPlanner::<f64>::new();
        let forward = planner.plan_fft_forward(length);
        let inverse = planner.plan_fft_inverse(length);
        forward.process(&mut left);
        forward.process(&mut right);
        for (a, b) in left.iter_mut().zip(right) {
            *a *= b;
        }
        inverse.process(&mut left);

        let coefficients = left
            .into_iter()
            .map(|value| (value.re / length as f64).round() as u64)
            .collect();
        Self::from_coefficients(coefficients)
    }

    fn from_coefficients(coefficients: Vec<u64>) -> Self {
        let mut limbs = Vec::with_capacity(coefficients.len() + 1);
        let mut carry = 0u64;
        for coefficient in coefficients {
            let value = coefficient + carry;
            limbs.push((value % BASE) as u32);
            carry = value / BASE;
        }
        while carry != 0 {
            limbs.push((carry % BASE) as u32);
            carry /= BASE;
        }
        let mut result = Self { limbs };
        result.normalize();
        result
    }

    fn to_biguint(&self) -> BigUint {
        let mut result = BigUint::default();
        for &limb in self.limbs.iter().rev() {
            result <<= LIMB_BITS;
            result += limb;
        }
        result
    }
}

fn binary_split(a: u32, b: u32) -> (BigNat, BigNat) {
    if b - a <= LEAF_TERMS {
        let mut p = BigNat::zero();
        let mut q = BigNat::one();
        for k in (a + 1)..=b {
            p = p.mul_small(k);
            p.add_assign(&BigNat::one());
            q = q.mul_small(k);
        }
        return (p, q);
    }

    let middle = (a + b) / 2;
    let ((left_p, left_q), (right_p, right_q)) = if b - a >= PARALLEL_CUTOFF {
        join(|| binary_split(a, middle), || binary_split(middle, b))
    } else {
        (binary_split(a, middle), binary_split(middle, b))
    };

    let mut p = left_p.mul(&right_q);
    p.add_assign(&right_p);
    let q = left_q.mul(&right_q);
    (p, q)
}

fn terms_needed(digits: usize) -> u32 {
    let target = digits as f64 * std::f64::consts::LN_10;
    let mut factorial_log = 0.0;
    let mut n = 1u32;
    while factorial_log < target {
        n += 1;
        factorial_log += (n as f64).ln();
    }
    n
}

fn compute_e(digits: usize) -> String {
    let n = terms_needed(digits + 16);
    let (p, q) = binary_split(0, n);
    let numerator = p.to_biguint() + q.to_biguint();
    let denominator = q.to_biguint();
    let scale = BigUint::from(10u32).pow(digits as u32);
    let scaled = numerator * scale / denominator;
    let mut text = scaled.to_str_radix(10);
    if text.len() <= digits {
        let zeros = "0".repeat(digits + 1 - text.len());
        text = zeros + &text;
    }
    let split = text.len() - digits;
    format!("{}.{}", &text[..split], &text[split..])
}

fn main() -> io::Result<()> {
    let mut args = env::args().skip(1);
    let digits = args
        .next()
        .map(|value| value.parse::<usize>())
        .transpose()
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "digits must be an integer"))?
        .unwrap_or(100);
    let output_path = args.next();

    if digits == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "digits must be greater than zero",
        ));
    }
    if digits > 1_000_000_000 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "the maximum supported precision is 1,000,000,000 digits",
        ));
    }

    let value = compute_e(digits);
    match output_path {
        Some(path) => File::create(path)?.write_all(value.as_bytes())?,
        None => println!("{value}"),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn multiplication_matches_small_integer_arithmetic() {
        for left in 0..500u32 {
            for right in 0..500u32 {
                let actual = BigNat::from_u32(left).mul(&BigNat::from_u32(right));
                let expected = BigUint::from(left) * BigUint::from(right);
                assert_eq!(actual.to_biguint(), expected);
            }
        }
    }

    #[test]
    fn fft_multiplication_matches_exact_arithmetic() {
        let mut left = BigNat::one();
        let mut right = BigNat::one();
        for i in 1..=400 {
            left = left.mul_small(i);
            right = right.mul_small(401 - i);
        }
        let actual = left.mul(&right).to_biguint();
        let expected = left.to_biguint() * right.to_biguint();
        assert_eq!(actual, expected);
    }

    #[test]
    fn binary_split_produces_digits_of_e() {
        assert_eq!(&compute_e(20), "2.71828182845904523536");
    }
}
