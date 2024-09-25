mod path_finding;
use pyo3::prelude::*;

/// A Python module implemented in Rust.
#[pymodule]
fn theta_star_bi_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<path_finding::ThetaStarBi>()?;
    Ok(())
}
