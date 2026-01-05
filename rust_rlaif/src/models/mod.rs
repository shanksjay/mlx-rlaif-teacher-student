pub mod teacher;
pub mod student;
pub mod dataset;
pub mod mlx_generator;
pub mod mlx_trainer;
pub mod code_extractor;

pub use teacher::TeacherModel;
pub use student::StudentModel;
pub use dataset::CodeDataset;
// MlxGenerator and MlxTrainer are used internally by StudentModel
// CodeSample is used internally, no need to export at module level

