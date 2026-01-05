use anyhow::Result;
use crate::config::RlaifConfig;
use crate::models::TeacherModel;
use crate::training::trainer::GeneratedSample;
use std::sync::Arc;

pub async fn compute_rewards(
    teacher: &Arc<TeacherModel>,
    samples: &[GeneratedSample],
    _config: &Arc<RlaifConfig>,
) -> Result<Vec<f64>> {
    let mut rewards = Vec::with_capacity(samples.len());

    // Parallel reward computation
    let reward_futures: Vec<_> = samples.iter()
        .map(|sample| {
            let teacher: Arc<TeacherModel> = Arc::clone(teacher);
            let sample = sample.clone();
            tokio::spawn(async move {
                teacher.score_code(&sample.code, &sample.prompt, &sample.language).await
            })
        })
        .collect();

    for future in reward_futures {
        let reward = future.await??;
        rewards.push(reward);
    }

    Ok(rewards)
}

