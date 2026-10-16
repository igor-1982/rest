//! Response-specific auxiliary basis of the `resp_auxbas_path` keyword.
//!
//! When set, the RI-JK response objects evaluate their low-precision (`prec = false`)
//! contractions — the CP-SCF response machinery — on decomposed ERIs freshly built on that
//! auxiliary basis, while the fock forms (`prec = true`) keep the SCF `rimatr`/`rimatr_sr`. This
//! is a response-side (A-tensor) substitution only: the energy-derivative (B-side) objects of
//! the hessian/multipole tasks keep the SCF auxiliary basis, so the CP-SCF solve becomes a
//! mixed-representation approximation — parallel to the coarser response grid of
//! `grid_level_cpscf`.
//!
//! Each energy contribution component that requires a decomposed ERI has its own definition
//! here: [`cderi_pair`] for the full-range RI-JK object, [`cderi_pair_sr`] for the RSH
//! short-range object. With the keyword set, the auxiliary-side `CInt` is regenerated on the
//! response basis (the SCF molecule is only read, never modified) and substituted into the rimatr
//! builder, yielding an owned tensor; with the keyword unset, the SCF tensors are borrowed as
//! before (zero copy).

use super::decompose::generate_rimatr_bare_on_cint;
use super::prelude_dev::*;
use super::util::{get_cint_aux_on_path, get_cint_mol};
use crate::analdrv::config::AnalDrvConfig;
use crate::SCF;

use log::debug;

/// Hard-error on the unsupported `even_tempered_basis` combination.
///
/// The even-tempered machinery would silently override the response basis, so the keyword
/// combination is rejected. Checked once up front by the task-level
/// [`validate_resp_auxbas`], and again at every response-object build
/// ([`cderi_pair`]/[`cderi_pair_sr`]), so paths that build the response objects directly (e.g. the
/// geomopt analytic hessian) are covered too.
pub fn assert_resp_auxbas_supported(scf_data: &SCF, config: &AnalDrvConfig) {
    if config.resp.auxbas_path.is_none() {
        return;
    }
    if scf_data.mol.ctrl.even_tempered_basis {
        panic!("`resp_auxbas_path` is not supported together with `even_tempered_basis`.");
    }
}

/// Validate `resp_auxbas_path` against this run's tasks, before any response object is built.
///
/// This is the task-level check: it warns when the keyword is set but no response (CP-SCF)
/// object will be built (`need_resp = false`), and enforces the hard error of
/// [`assert_resp_auxbas_supported`]. The per-component definitions call the latter themselves,
/// but only this function carries the warn (only the caller knows `need_resp`).
pub fn validate_resp_auxbas(scf_data: &SCF, config: &AnalDrvConfig, need_resp: bool) {
    if config.resp.auxbas_path.is_none() {
        return;
    }
    assert_resp_auxbas_supported(scf_data, config);
    if !need_resp {
        log::warn!(
            "`resp_auxbas_path` is set, but no task of this run uses the response (CP-SCF) object; the keyword will be ignored."
        );
    }
}

/// Decomposed ERIs of the full-range RI-JK response object, per precision, as a pair
/// `(cderi, cderi_resp)` of [`TsrCow`]s.
///
/// The first entry (the high-precision, `prec = true` resource of the fock forms) is always the
/// SCF `rimatr`, borrowed zero-copy. The second entry (the low-precision, `prec = false`
/// resource of the CP-SCF response contractions) is a freshly built tensor only when
/// `resp_auxbas_path` is set: the auxiliary-side `CInt` is regenerated from a local copy of the
/// keywords ([`get_cint_aux_on_path`](crate::ri_jk::util::get_cint_aux_on_path)) and, together
/// with the molecule-side one, handed to the CInt-level rimatr builder
/// ([`generate_rimatr_bare_on_cint`](crate::ri_jk::generate_rimatr_bare_on_cint)), so the SCF
/// molecule stays untouched and the tensor is owned. A `resp_auxbas_path` equal to the SCF
/// `auxbas_path` (exact match) is treated as unset when the SCF `rimatr` is available: the
/// borrowed tensor is used instead of rebuilding an identical one. With the keyword unset, the
/// second entry is `None` (the response path reuses the SCF tensor).
pub fn cderi_pair<'a>(scf_data: &'a SCF, config: &AnalDrvConfig) -> (TsrCow<'a>, Option<TsrCow<'a>>) {
    let device = DeviceBLAS::default();
    // a response basis equal to the SCF one, with the SCF tensor available, is treated as unset
    let resp_auxbas_path = match config.resp.auxbas_path.as_deref() {
        Some(path) if scf_data.mol.ctrl.auxbas_path == path && scf_data.rimatr.is_some() => {
            debug!("`resp_auxbas_path` equals the SCF auxiliary basis; treated as unset.");
            None
        },
        path => path,
    };
    let cderi_resp = match resp_auxbas_path {
        Some(resp_auxbas_path) => {
            assert_resp_auxbas_supported(scf_data, config);
            debug!("Building the response rimatr on `resp_auxbas_path` = \"{}\".", resp_auxbas_path);
            let mol = get_cint_mol(&scf_data.mol);
            let aux = get_cint_aux_on_path(&scf_data.mol, resp_auxbas_path);
            let cderi = generate_rimatr_bare_on_cint(
                &mol,
                &aux,
                scf_data.mol.ctrl.j2c_decomp,
                None,
                // deliberately unscreened: low-precision response path, small aux basis
                0.0,
                0.0,
            )
            .into_rstsr(&device);
            debug!(
                "RI-JK low-precision response path on the `resp_auxbas_path` decomposed ERI (owned, {} auxiliary functions).",
                cderi.shape()[1]
            );
            Some(cderi.into_cow())
        },
        None => None,
    };
    debug!("RI-JK fock forms of the response object on the SCF rimatr (borrowed, zero copy).");
    let (rimatr, _, _) = scf_data.rimatr.as_ref().expect(
        "This implementation requires cholesky decomposed ERI (or rimatr) to be available and stored in memory.",
    );
    let cderi = rimatr.to_rstsr_view(&device).into_cow();
    (cderi, cderi_resp)
}

/// Decomposed ERIs of the RSH short-range RI-JK response object, per precision, as a pair
/// `(cderi_sr, cderi_resp_sr)` of [`TsrCow`]s.
///
/// Same definition as [`cderi_pair`] on the short-range tensors: with `resp_auxbas_path` set,
/// the SR rimatr is freshly built (with `-omega` in libcint's convention, cf. the SCF integral
/// preparation) for the low-precision response path, and a path equal to the SCF `auxbas_path`
/// is treated as unset when the SCF `rimatr_sr` is available; the fock forms always borrow the
/// SCF `rimatr_sr`, and with the keyword unset the response path reuses it too (zero copy).
pub fn cderi_pair_sr<'a>(scf_data: &'a SCF, config: &AnalDrvConfig) -> (TsrCow<'a>, Option<TsrCow<'a>>) {
    let device = DeviceBLAS::default();
    // a response basis equal to the SCF one, with the SCF tensor available, is treated as unset
    let resp_auxbas_path = match config.resp.auxbas_path.as_deref() {
        Some(path) if scf_data.mol.ctrl.auxbas_path == path && scf_data.rimatr_sr.is_some() => {
            debug!("`resp_auxbas_path` equals the SCF auxiliary basis; treated as unset.");
            None
        },
        path => path,
    };
    let cderi_resp_sr = match resp_auxbas_path {
        Some(resp_auxbas_path) => {
            assert_resp_auxbas_supported(scf_data, config);
            // SR RI omega is negative in libcint's convention (cf. the SCF integral preparation)
            let omega = scf_data.mol.xc_data.omega().unwrap();
            debug!(
                "Building the short-range response rimatr for RSH (omega = {:.4}) on `resp_auxbas_path` = \"{}\".",
                omega, resp_auxbas_path
            );
            let mol = get_cint_mol(&scf_data.mol);
            let aux = get_cint_aux_on_path(&scf_data.mol, resp_auxbas_path);
            let cderi_sr = generate_rimatr_bare_on_cint(
                &mol,
                &aux,
                scf_data.mol.ctrl.j2c_decomp,
                Some(-omega),
                // deliberately unscreened: low-precision response path, small aux basis
                0.0,
                0.0,
            )
            .into_rstsr(&device);
            debug!("RSH short-range low-precision response path on the `resp_auxbas_path` decomposed ERI (owned).");
            Some(cderi_sr.into_cow())
        },
        None => None,
    };
    debug!("RSH short-range fock forms of the response object on the SCF rimatr_sr (borrowed, zero copy).");
    let (rimatr_sr, _, _) = scf_data.rimatr_sr.as_ref().expect(
        "The range-separated response requires the short-range ERI (rimatr_sr) to be built and stored in memory.",
    );
    let cderi_sr = rimatr_sr.to_rstsr_view(&device).into_cow();
    (cderi_sr, cderi_resp_sr)
}
