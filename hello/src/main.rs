use std::error::Error;
use std::fs::File;
use std::io::BufReader;

use serde::Deserialize;
use csv::ReaderBuilder;

#[derive(Debug, Deserialize)]
struct Record {
    open: f64,
    close: f64,
}

fn compute_tolerance(changes: &[f64], factor: f64) -> f64 {
    let mean: f64 = changes.iter().copied().sum::<f64>() / (changes.len() as f64);
    let variance: f64 = changes.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (changes.len() as f64);
    let std_dev = variance.sqrt();
    std_dev * factor
}

fn within_tolerance(a: f64, b: f64, tolerance: f64) -> bool {
    (a - b).abs() <= tolerance
}

fn pattern_matching_with_tolerance(text: &[f64], pattern: &[f64], tolerance: f64) -> Vec<usize> {
    let m = pattern.len();
    let n = text.len();
    let mut matches = Vec::new();
    if m == 0 || m > n {
        return matches;
    }

    // A direct scan: O(N*m)
    // If m is small enough or data size is moderate, this is quite efficient in practice.
    for s in 0..=n - m {
        let mut matched = true;
        for j in 0..m {
            if !within_tolerance(text[s + j], pattern[j], tolerance) {
                matched = false;
                break;
            }
        }
        if matched {
            matches.push(s);
        }
    }

    matches
}

fn get_sublists_from_back<'a>(lst: &'a [f64], min_length: usize, max_length: usize) -> Vec<&'a [f64]> {
    let n = lst.len();
    // Instead of generating all sublists from the back, we limit the max_length to avoid O(N^2) blowup.
    // We only generate suffixes of length between min_length and max_length.
    // For example, if min_length=3 and max_length=10, we only get suffixes of length in [3..10].
    let mut sublists = Vec::new();
    for length in (min_length..=max_length).rev() {
        if length <= n {
            sublists.push(&lst[n - length..]);
        }
    }
    sublists
}

fn determine_action(changes: &[f64], min_length: usize, max_length: usize) -> (Vec<(Vec<f64>, Vec<usize>, i64)>, i64) {
    let dynamic_tolerance = compute_tolerance(changes, 0.3);
    let patterns = get_sublists_from_back(changes, min_length, max_length);
    let mut matched = Vec::new();
    let mut total_score = 0;

    // For each pattern, find matches and compute score.
    // Limiting max_length drastically reduces computation.
    for pat_slice in patterns {
        let pattern = pat_slice;
        let result = pattern_matching_with_tolerance(changes, pattern, dynamic_tolerance);

        let valid_result: Vec<usize> = result
            .into_iter()
            .filter(|&idx| idx + pattern.len() < changes.len())
            .collect();

        if valid_result.is_empty() {
            continue;
        }

        // Compute score based on future changes after each match
        let score: i64 = valid_result.iter().map(|&idx| {
            if changes[idx + pattern.len()] > 0.0 { 1 } else { -1 }
        }).sum();

        matched.push((pattern.to_vec(), valid_result, score));
        total_score += score;
    }

    (matched, total_score)
}

fn main() -> Result<(), Box<dyn Error>> {
    let file = File::open("../futures_data.csv")?;
    let mut rdr = ReaderBuilder::new().from_reader(BufReader::new(file));

    let mut open_prices = Vec::new();
    let mut close_prices = Vec::new();

    for result in rdr.deserialize() {
        let record: Record = result?;
        open_prices.push(record.open);
        close_prices.push(record.close);
    }

    let changes: Vec<f64> = close_prices.iter().zip(open_prices.iter())
        .map(|(&c, &o)| ((c - o) / o) * 100.0)
        .collect();

    // Adjust these values as needed
    let min_length = 3;
    let max_length = 10;

    let (actions, total_score) = determine_action(&changes, min_length, max_length);
    let tolerance = compute_tolerance(&changes, 0.3);
    println!("Tolerance: {}", tolerance);

    // If needed, uncomment to view the matched patterns and their scores:
    // for (pattern, result, score) in &actions {
    //     println!("-----------------------------------------------------------");
    //     println!("Pattern {:?}\nfound at index {:?}, score: {}", pattern, result, score);
    //     println!("-----------------------------------------------------------");
    // }

    println!("Total Score: {}", total_score);

    if total_score > 0 {
        println!("Decision: BUY");
    } else if total_score < 0 {
        println!("Decision: SELL");
    } else {
        println!("Decision: None");
    }

    Ok(())
}