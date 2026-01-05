use tracing::info;

#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    pub generation_times: Vec<f64>,      // Generation latencies in seconds
    pub scoring_times: Vec<f64>,        // API scoring latencies in seconds
    pub backprop_times: Vec<f64>,       // Backpropagation latencies in seconds
    pub generation_tokens: Vec<usize>,   // Tokens generated per sample
    pub scoring_tokens: Vec<usize>,      // API tokens used for scoring
    pub backprop_tokens: Vec<usize>,     // Tokens processed in backprop
}

impl PerformanceMetrics {
    pub fn new() -> Self {
        Self {
            generation_times: Vec::new(),
            scoring_times: Vec::new(),
            backprop_times: Vec::new(),
            generation_tokens: Vec::new(),
            scoring_tokens: Vec::new(),
            backprop_tokens: Vec::new(),
        }
    }

    pub fn record_generation(&mut self, time_sec: f64, tokens: usize) {
        self.generation_times.push(time_sec);
        self.generation_tokens.push(tokens);
    }

    pub fn record_scoring(&mut self, time_sec: f64, tokens: usize) {
        self.scoring_times.push(time_sec);
        self.scoring_tokens.push(tokens);
    }

    pub fn record_backprop(&mut self, time_sec: f64, tokens: usize) {
        self.backprop_times.push(time_sec);
        self.backprop_tokens.push(tokens);
    }

    /// Calculate percentile (0-100)
    fn percentile(data: &[f64], p: f64) -> f64 {
        if data.is_empty() {
            return 0.0;
        }
        let mut sorted = data.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let index = ((sorted.len() - 1) as f64 * p / 100.0).ceil() as usize;
        sorted[index.min(sorted.len() - 1)]
    }

    /// Calculate average
    fn average(data: &[f64]) -> f64 {
        if data.is_empty() {
            return 0.0;
        }
        data.iter().sum::<f64>() / data.len() as f64
    }

    /// Calculate tokens per second from times and token counts
    fn tokens_per_sec(times: &[f64], tokens: &[usize]) -> Vec<f64> {
        times.iter()
            .zip(tokens.iter())
            .map(|(time, &tokens)| {
                if *time > 0.0 {
                    tokens as f64 / *time
                } else {
                    0.0
                }
            })
            .collect()
    }

    pub fn print_summary(&self) {
        let separator = "=".repeat(80);
        info!("{}", separator);
        info!("PERFORMANCE METRICS SUMMARY");
        info!("{}", separator);

        // Generation Performance
        if !self.generation_times.is_empty() {
            let avg_time = Self::average(&self.generation_times);
            let p99_time = Self::percentile(&self.generation_times, 99.0);
            let gen_tps = Self::tokens_per_sec(&self.generation_times, &self.generation_tokens);
            let avg_tps = Self::average(&gen_tps);
            let p99_tps = Self::percentile(&gen_tps, 99.0);
            let total_tokens: usize = self.generation_tokens.iter().sum();

            info!("\n📊 Generation Performance:");
            info!("  Average Latency: {:.3}s", avg_time);
            info!("  P99 Latency:     {:.3}s", p99_time);
            info!("  Average Speed:   {:.2} tokens/sec", avg_tps);
            info!("  P99 Speed:       {:.2} tokens/sec", p99_tps);
            info!("  Samples:         {}", self.generation_times.len());
            info!("  Total Tokens:    {}", total_tokens);
        } else {
            info!("\n📊 Generation Performance: No data");
        }

        // API Scoring Performance
        if !self.scoring_times.is_empty() {
            let avg_time = Self::average(&self.scoring_times);
            let p99_time = Self::percentile(&self.scoring_times, 99.0);
            let scoring_tps = Self::tokens_per_sec(&self.scoring_times, &self.scoring_tokens);
            let avg_tps = Self::average(&scoring_tps);
            let p99_tps = Self::percentile(&scoring_tps, 99.0);
            let total_tokens: usize = self.scoring_tokens.iter().sum();

            info!("\n🎯 API Scoring Performance:");
            info!("  Average Latency: {:.3}s", avg_time);
            info!("  P99 Latency:     {:.3}s", p99_time);
            info!("  Average Speed:   {:.2} tokens/sec", avg_tps);
            info!("  P99 Speed:       {:.2} tokens/sec", p99_tps);
            info!("  Samples:         {}", self.scoring_times.len());
            info!("  Total API Tokens: {}", total_tokens);
        } else {
            info!("\n🎯 API Scoring Performance: No data");
        }

        // Backpropagation Performance
        if !self.backprop_times.is_empty() {
            let avg_time = Self::average(&self.backprop_times);
            let p99_time = Self::percentile(&self.backprop_times, 99.0);
            let backprop_tps = Self::tokens_per_sec(&self.backprop_times, &self.backprop_tokens);
            let avg_tps = Self::average(&backprop_tps);
            let p99_tps = Self::percentile(&backprop_tps, 99.0);
            let total_tokens: usize = self.backprop_tokens.iter().sum();

            info!("\n🔄 Backpropagation Performance:");
            info!("  Average Latency: {:.3}s", avg_time);
            info!("  P99 Latency:     {:.3}s", p99_time);
            info!("  Average Speed:   {:.2} tokens/sec", avg_tps);
            info!("  P99 Speed:       {:.2} tokens/sec", p99_tps);
            info!("  Samples:         {}", self.backprop_times.len());
            info!("  Total Tokens:    {}", total_tokens);
        } else {
            info!("\n🔄 Backpropagation Performance: No data");
        }

        let separator = "=".repeat(80);
        info!("{}", separator);
    }
}

impl Default for PerformanceMetrics {
    fn default() -> Self {
        Self::new()
    }
}

