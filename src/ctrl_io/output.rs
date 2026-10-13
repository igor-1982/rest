//! Keywords of the `[output]` block of the control input.
//!
//! The `[output]` block configures how the requested output files (the
//! `outputs` list in the `[ctrl]` block) are generated.

use serde::{Deserialize, Serialize};

/// Options of the `[output]` block of the control input.
#[derive(Debug,Clone,Serialize, Deserialize)]
#[serde(default)]
pub struct OutputKeywords {
    /// Dump the density matrices into the fchk file: "auto" (default) dumps
    /// every density that has been calculated (`Total/Spin SCF Density`, and
    /// for double-hybrid runs additionally `Total/Spin MP2 Density`), "scf"
    /// dumps only the SCF density, "false" dumps none.
    pub fchk_dm: String,
    /// Which writer generates the Gaussian-ordered content of the fchk file
    /// (MO coefficients, and the density layout they must agree with):
    /// "default" (the MOKIT-derived librest2fch library when compiled in, the
    /// native Rust writer otherwise), "librest2fch" or "rust". The density
    /// itself is always computed by REST; the librest2fch path preserves it
    /// (its own gen_density branch is not used, since it would rebuild the
    /// density from aufbau occupations and drop the UHF spin density). Note
    /// that omitting the density (`fchk_dm = "false"`) requires a librest2fch
    /// library that honors `gen_density = 0`; very old libraries ignore it
    /// and always dump the density.
    pub fchk_writer: String,
    /// Cube-file options for `outputs = ["cube_orb"]`. These keywords may also
    /// be given at their legacy `[ctrl]` location (still accepted, deprecated);
    /// a value set here takes precedence. `None` means "not set in `[output]`";
    /// the merge into the effective values on `InputKeywords` happens in
    /// `resolve_cube_keywords`.
    pub cube_orb_setting: Option<[f64;2]>,
    pub cube_orb_indices: Option<Vec<[usize;3]>>,
    pub cube_orb_type: Option<String>,
}

impl Default for OutputKeywords {
    fn default() -> Self {
        OutputKeywords { fchk_dm: String::from("auto"), fchk_writer: String::from("default"), cube_orb_setting: None, cube_orb_indices: None, cube_orb_type: None }
    }
}

pub fn parse_output_keywords(tmp_keys: &serde_json::Value) -> OutputKeywords {
    match tmp_keys.get("output") {
        Some(tmp_output) => serde_json::from_value(tmp_output.clone()).unwrap(),
        None => OutputKeywords::default(),
    }
}

/// Merge the cube keywords of the `[output]` block into the effective values
/// stored on `InputKeywords` (which `parse_ctrl_keywords` has already filled
/// with the legacy `[ctrl]` values, or the defaults).
///
/// Precedence: `[output]` > `[ctrl]`. The legacy `[ctrl]` location is still
/// accepted for backward compatibility; using it prints a deprecation hint,
/// and setting the same keyword in both blocks warns that `[output]` wins.
pub fn resolve_cube_keywords(input: &mut super::InputKeywords, tmp_keys: &serde_json::Value) {
    let ctrl_block = tmp_keys.get("ctrl");
    let in_ctrl = |name: &str| ctrl_block.and_then(|c| c.get(name)).is_some();
    let warn_moved = |name: &str, both: bool| {
        if both {
            println!("[WARN] Keyword '{name}' is set in both [ctrl] and [output]; the [output] value is used.");
        } else {
            println!("[WARN] Keyword '{name}' in [ctrl] has moved to the [output] block; the [ctrl] location is still accepted but deprecated.");
        }
    };

    if let Some(v) = input.output.cube_orb_type.take() {
        let both = in_ctrl("cube_orb_type");
        if both {
            warn_moved("cube_orb_type", true);
        }
        input.cube_orb_type = v.to_lowercase();
    } else if in_ctrl("cube_orb_type") {
        warn_moved("cube_orb_type", false);
    }

    if let Some(v) = input.output.cube_orb_setting.take() {
        if in_ctrl("cube_orb_setting") {
            warn_moved("cube_orb_setting", true);
        }
        input.cube_orb_setting = v;
    } else if in_ctrl("cube_orb_setting") {
        warn_moved("cube_orb_setting", false);
    }

    if let Some(v) = input.output.cube_orb_indices.take() {
        if in_ctrl("cube_orb_indices") {
            warn_moved("cube_orb_indices", true);
        }
        input.cube_orb_indices = v;
    } else if in_ctrl("cube_orb_indices") {
        warn_moved("cube_orb_indices", false);
    }
}

#[cfg(test)]
mod tests {
    use super::super::parse_ctl_from_json;

    fn parse(json: &str) -> super::super::InputKeywords {
        let mut keys: serde_json::Value = serde_json::from_str(json).unwrap();
        // the cube keywords under test live in [ctrl]/[output]; [geom] is only
        // required so that parse_ctl_from_json accepts the input at all
        if keys.get("geom").is_none() {
            keys["geom"] = serde_json::json!({
                "name": "H2", "unit": "angstrom",
                "position": "H 0.0 0.0 0.0\nH 0.0 0.0 0.7"
            });
        }
        if keys.get("ctrl").is_none() {
            keys["ctrl"] = serde_json::json!({});
        }
        let (input, _geom) = parse_ctl_from_json(&keys).unwrap();
        input
    }

    #[test]
    fn cube_keywords_default() {
        let input = parse(r#"{}"#);
        assert_eq!(input.cube_orb_setting, [3.0, 80.0]);
        assert!(input.cube_orb_indices.is_empty());
        assert_eq!(input.cube_orb_type, "wavefunction");
    }

    #[test]
    fn cube_keywords_legacy_ctrl_location_still_works() {
        let input = parse(r#"{
            "ctrl": {
                "cube_orb_setting": [4.0, 100.0],
                "cube_orb_indices": [[19, 21, 0]],
                "cube_orb_type": "Density"
            }
        }"#);
        assert_eq!(input.cube_orb_setting, [4.0, 100.0]);
        assert_eq!(input.cube_orb_indices, vec![[19, 21, 0]]);
        assert_eq!(input.cube_orb_type, "density");
    }

    #[test]
    fn cube_keywords_output_block_takes_precedence() {
        let input = parse(r#"{
            "ctrl": {
                "cube_orb_setting": [4.0, 100.0],
                "cube_orb_type": "density"
            },
            "output": {
                "cube_orb_setting": [5.0, 120.0],
                "cube_orb_indices": [[0, 2, 1]],
                "cube_orb_type": "wavefunction"
            }
        }"#);
        assert_eq!(input.cube_orb_setting, [5.0, 120.0]);
        assert_eq!(input.cube_orb_indices, vec![[0, 2, 1]]);
        assert_eq!(input.cube_orb_type, "wavefunction");
    }

    #[test]
    fn cube_keywords_output_block_partial_override() {
        let input = parse(r#"{
            "ctrl": { "cube_orb_indices": [[19, 21, 0]] },
            "output": { "cube_orb_type": "density" }
        }"#);
        assert_eq!(input.cube_orb_indices, vec![[19, 21, 0]]);
        assert_eq!(input.cube_orb_type, "density");
    }
}
