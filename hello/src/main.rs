use std::error::Error;
use std::fs::File;
use std::io::BufReader;

use serde::Deserialize;
use csv::ReaderBuilder;
use rayon::prelude::*;

#[derive(Debug, Deserialize)]
struct Record {
    open: f64,
    close: f64,
}

// Optimized tolerance computation using iterators
fn compute_tolerance(changes: &[f64], factor: f64) -> f64 {
    let n = changes.len() as f64;
    let mean = changes.iter().sum::<f64>() / n;
    let variance = changes.iter()
        .map(|&x| (x - mean).powi(2))
        .sum::<f64>() / n;
    variance.sqrt() * factor
}

// SIMD-friendly within tolerance check
fn within_tolerance(a: f64, b: f64, tolerance: f64) -> bool {
    (a - b).abs() <= tolerance
}

// Optimized pattern matching using parallel iterator
fn pattern_matching_with_tolerance(text: &[f64], pattern: &[f64], tolerance: f64) -> Vec<usize> {
    let m = pattern.len();
    let n = text.len();
    
    if m == 0 || m > n {
        return Vec::new();
    }

    // Parallel iterator for matching
    (0..=n - m)
        .into_par_iter()
        .filter_map(|s| {
            let is_match = pattern.iter()
                .enumerate()
                .all(|(j, &pat_val)| within_tolerance(text[s + j], pat_val, tolerance));
            
            if is_match { Some(s) } else { None }
        })
        .collect()
}

// Efficient pattern generation focusing on recent patterns
fn get_efficient_patterns(changes: &[f64], min_length: usize, max_length: usize) -> Vec<Vec<f64>> {
    let n = changes.len();
    let mut patterns = Vec::new();
    
    // Focus on recent patterns from the end
    for length in (min_length..=max_length).rev() {
        if length <= n {
            // Add the most recent patterns first
            for start in (0..=n - length).rev().take(5) {
                patterns.push(changes[start..start + length].to_vec());
            }
        }
    }
    
    patterns
}

// Parallel scoring mechanism
fn determine_action(changes: &[f64], min_length: usize, max_length: usize) -> (Vec<(Vec<f64>, Vec<usize>, i64)>, i64) {
    let dynamic_tolerance = compute_tolerance(changes, 0.3);
    
    // Use parallel iterator for pattern matching and scoring
    let matched: Vec<_> = get_efficient_patterns(changes, min_length, max_length)
        .into_par_iter()
        .filter_map(|pattern| {
            let result = pattern_matching_with_tolerance(changes, &pattern, dynamic_tolerance);
            
            // Filter valid matches with future changes
            let valid_result: Vec<usize> = result
                .into_iter()
                .filter(|&idx| idx + pattern.len() < changes.len())
                .collect();
            
            if valid_result.is_empty() {
                return None;
            }
            
            // Parallel scoring
            let score: i64 = valid_result.iter()
                .map(|&idx| if changes[idx + pattern.len()] > 0.0 { 1 } else { -1 })
                .sum();
            
            Some((pattern, valid_result, score))
        })
        .collect();
    
    let total_score = matched.iter().map(|&(_, _, score)| score).sum();
    
    (matched, total_score)
}

fn main() -> Result<(), Box<dyn Error>> {
    // Use buffered reader for efficient file reading
    let file = File::open("../futures_data.csv")?;
    let mut rdr = ReaderBuilder::new().from_reader(BufReader::new(file));

    // Pre-allocate with capacity for efficiency
    let mut open_prices = Vec::with_capacity(100000);
    let mut close_prices = Vec::with_capacity(100000);

    // Efficient deserialization
    for result in rdr.deserialize() {
        let record: Record = result?;
        open_prices.push(record.open);
        close_prices.push(record.close);
    }

    // Compute price changes with a single allocation
    let changes: Vec<f64> = close_prices.iter()
        .zip(open_prices.iter())
        .map(|(&c, &o)| ((c - o) / o) * 100.0)
        .collect();

    // Adjust these values as needed
    let min_length = 3;
    let max_length = 9;

    // Perform analysis
    let (actions, total_score) = determine_action(&changes, min_length, max_length);
    let tolerance = compute_tolerance(&changes, 0.3);
    println!("Tolerance: {}", tolerance);

    // Conditionally print detailed actions (commented out by default)
    // for (pattern, result, score) in &actions {
    //     println!("-----------------------------------------------------------");
    //     println!("Pattern {:?}\nfound at index {:?}, score: {}", pattern, result, score);
    //     println!("-----------------------------------------------------------");
    // }

    println!("Total Score: {}", total_score);

    // Decision logic
    match total_score.cmp(&0) {
        std::cmp::Ordering::Greater => println!("Decision: BUY"),
        std::cmp::Ordering::Less => println!("Decision: SELL"),
        std::cmp::Ordering::Equal => println!("Decision: None"),
    }

    Ok(())
}