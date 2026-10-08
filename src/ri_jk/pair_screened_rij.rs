//! Basis-pair-screened in-core RI-J and RI-K kernels.
//!
//! The in-core RI path stores the metric-transformed three-center tensor `rimatr` with shape
//! `[nbaspar, naux]` (column-major, one row per AO pair, `nbaspar = nao(nao+1)/2`). Both Fock
//! contractions are linear in that row index, so a pair whose row cannot contribute may be
//! skipped:
//!
//! `text
//! J:  g[P] = sum_p w_p R[p,P]         (density contraction, w_p = 2 D_p, or D_p on the diagonal)
//!     J[p] = sum_P R[p,P] g[P]        (back contraction)
//!
//! K:  T[mu,i] = sum_lambda R_P[mu,lambda] C[lambda,i]   (per auxiliary function P, C = C_occ sqrt(occ))
//!     K      += T T^t
//! `
//!
//! Two screening levels act on that row index.
//!
//! **Storage level** ([`prune_rimatr`]) drops the rows of a freshly built tensor whose static
//! bound `q[p] = max_P |R[p,P]|` falls below `tol` times the largest bound. The criterion is
//! geometric, so the retained row set is fixed for a geometry and can be reused by every SCF
//! iteration; the compacted tensor carries its own row description, the [`PairMap`]. The
//! full-space tables `basbas2baspar` and `baspar2basbas` keep their meaning untouched, and the
//! pruned kernels address the tensor through the map.
//!
//! **Iteration level** ([`get_vj_pair_screened`], [`get_vk_pair_screened`]) drops the stored
//! rows that cannot contribute *at this density or with these orbitals*:
//!
//! - **J** drops a row of the density contraction when `|w_p| q[p]` falls below the threshold
//!   relative to `max_p |w_p| q[p]`, and a row of the back contraction when `q[p]` falls below
//!   the threshold relative to `max_p q[p]` (the quantity that contraction actually forms).
//! - **K** drops a row when `q[p] max(W_mu, W_nu)` falls below the threshold relative to its own
//!   maximum, with `W_mu = max_i |C[mu,i]|`. The retained rows define the *active* AO subset
//!   `A`; each auxiliary function is then contracted on that subset only, accumulated into a
//!   `|A| x |A|` `dsyrk` accumulator, and scattered back into the full symmetric matrix.
//!
//! Both kernels only ever *drop* contributions whose bound is below the threshold, and a
//! threshold of `0.0` keeps every stored row. The J kernel always returns the packed upper
//! triangle of the **full** pair space: dropped rows contribute zeros, so the caller keeps
//! assembling the Fock matrix with the unmodified `basbas2baspar`.

use super::prelude_dev::*;
use log::debug;
use tensors::matrix_blas_lapack::{_dsymm, _dsyrk};
use tensors::BasicMatrix;

/// Static per-pair data of the basis-pair screening of the in-core RI kernels.
///
/// Built once per geometry from a **full-space** `rimatr` tensor and reused by every SCF
/// iteration. See the module documentation for the meaning of the bounds. The compacted row
/// space of a storage-level pruned tensor is described by [`PairMap`] instead.
#[derive(Clone)]
pub struct RIMatrPairScreen {
    /// Row bound of every AO pair, `q[p] = max_P |R[p,P]|`.
    pub bounds: Vec<f64>,
    /// Whether the packed pair is a diagonal pair (mu = nu), which enters the J density
    /// contraction with weight 1 instead of 2.
    pub is_diagonal: Vec<bool>,
    /// Number of auxiliary functions of the bound tensor.
    pub num_auxbas: usize,
}

impl RIMatrPairScreen {
    /// Build the static screening data of one geometry.
    ///
    /// - `ri3fn`: the metric-transformed three-center tensor, shape `[nbaspar, naux]`.
    /// - `basbas2baspar`: map from an AO pair to its packed row index of `ri3fn`.
    /// - `nao`: number of AO basis functions.
    ///
    /// Under MPI the auxiliary functions are distributed over the ranks, so the row maxima are
    /// only partial: the caller has to reduce `bounds` to the global maximum before using them
    /// to prune, otherwise the ranks would keep different rows and the layout would diverge.
    pub fn build(ri3fn: &MatrixFull<f64>, basbas2baspar: &MatrixFull<usize>, nao: usize) -> Self {
        let num_baspar = ri3fn.size()[0];
        let num_auxbas = ri3fn.size()[1];
        let bounds = row_bounds(ri3fn);

        let mut is_diagonal = vec![false; num_baspar];
        for i in 0..nao {
            let p = basbas2baspar[[i, i]];
            if p < num_baspar {
                is_diagonal[p] = true;
            }
        }

        RIMatrPairScreen { bounds, is_diagonal, num_auxbas }
    }

    /// Number of pairs retained by the J screening at this density and threshold.
    ///
    /// This is a full-space diagnostic: it compares the packed density with the full-space
    /// bounds, and is not meaningful for the compacted rows of a [`PairMap`].
    pub fn count_retained_j(&self, dm_upper: &[f64], threshold: f64) -> usize {
        (0..self.bounds.len())
            .filter(|&p| {
                let w = if self.is_diagonal[p] { dm_upper[p] } else { 2.0 * dm_upper[p] };
                w.abs() * self.bounds[p] > threshold
            })
            .count()
    }

    /// Number of pairs retained by the K screening for the given per-AO orbital weights.
    ///
    /// Full-space diagnostic, see [`RIMatrPairScreen::count_retained_j`].
    pub fn count_retained_k(&self, w_ao: &[f64], baspar2basbas: &[[usize; 2]], threshold: f64) -> usize {
        (0..self.bounds.len())
            .filter(|&p| {
                let [mu, nu] = baspar2basbas[p];
                self.bounds[p] * w_ao[mu].max(w_ao[nu]) > threshold
            })
            .count()
    }
}

/// Per-AO weight `W_mu = max_i |C[mu,i]|` of a (possibly occupation-scaled) orbital coefficient
/// matrix, shape `[nao, nw]` column-major.
pub fn orbital_weights(coefficients: &MatrixFull<f64>) -> Vec<f64> {
    let nao = coefficients.size()[0];
    let nw = coefficients.size()[1];
    let data = coefficients.data_ref().unwrap();
    let mut w_ao = vec![0.0_f64; nao];
    w_ao.par_iter_mut().enumerate().for_each(|(mu, w)| {
        let mut bound = 0.0_f64;
        for i in 0..nw {
            let value = data[mu + nao * i].abs();
            if value > bound {
                bound = value;
            }
        }
        *w = bound;
    });
    w_ao
}

/// Weight of a stored row in the J density contraction.
///
/// Both triangles of a symmetric density contribute, so an off-diagonal pair counts twice. The
/// unscreened kernel realizes the same factor by halving the packed diagonal before a factor-2
/// GEMV.
#[inline]
fn j_row_weight(map: &PairMap, row: usize, dm: &MatrixFull<f64>) -> f64 {
    let (mu, nu) = map.pair_of(row);
    let d = dm[[mu, nu]];
    if mu == nu { d } else { 2.0 * d }
}

/// Screening scale of the J density contraction: the largest density-weighted row bound
/// `max_p |w_p| q[p]` of this iteration. The kernel drops rows below `threshold` times this
/// value, so the criterion is relative.
pub fn j_density_scale(map: &PairMap, dm: &MatrixFull<f64>) -> f64 {
    (0..map.len())
        .map(|p| j_row_weight(map, p, dm).abs() * map.bounds[p])
        .fold(0.0_f64, |acc, v| acc.max(v))
}

/// Screening scale of the J back contraction: the largest static row bound `max_p q[p]` of the
/// stored rows. The back contraction is bounded by it times a constant, and that constant
/// cancels in the relative criterion, so the criterion is static per geometry.
pub fn geometric_scale(map: &PairMap) -> f64 {
    map.bounds.iter().fold(0.0_f64, |acc, v| acc.max(*v))
}

/// Screening scale of the K contraction for the given per-AO weights:
/// `max_p q[p] max(W_mu, W_nu)`.
pub fn orbital_scale(map: &PairMap, w_ao: &[f64]) -> f64 {
    (0..map.len())
        .map(|p| {
            let (mu, nu) = map.pair_of(p);
            map.bounds[p] * w_ao[mu].max(w_ao[nu])
        })
        .fold(0.0_f64, |acc, v| acc.max(v))
}

/// Pair indexing of a storage-level pruned `rimatr`.
///
/// The pruned storage is addressed in a convention of its own, and the caller keeps the two
/// full-space tables of the unpruned tensor untouched:
///
/// - `rows`: stored row -> `(mu, nu)`, in ascending full-pair order. This is the hot table of
///   the J/K kernels, so it is stored as 32-bit indices (half the traffic of `[usize; 2]`).
/// - `full2row`: full packed pair index -> stored row, or [`PairMap::NOT_STORED`]. Indexing by
///   the full packed pair (what `basbas2baspar` returns) keeps the meaning of that table intact,
///   and no consumer has to learn a new sentinel inside a `[nao, nao]` matrix.
/// - `bounds`: static row bound `q[p] = max_P |R[p,P]|` of every stored row, in the same order.
///   It is the bound of the tensor that defined the pattern, which the iteration-level criteria
///   of the kernels above need.
///
/// `basbas2baspar` and `baspar2basbas` therefore keep their full-space semantics in every path,
/// pruned or not.
#[derive(Clone)]
pub struct PairMap {
    /// full packed pair index -> stored row, or [`PairMap::NOT_STORED`]
    pub full2row: Vec<u32>,
    /// stored row -> `(mu, nu)`, ascending full-pair order
    pub rows: Vec<[u32; 2]>,
    /// stored row -> static row bound `q[p]`, in the same order as `rows`
    pub bounds: Vec<f64>,
}

impl PairMap {
    /// Sentinel of [`PairMap::full2row`] for a pair that was not stored.
    pub const NOT_STORED: u32 = u32::MAX;

    /// Build the map of the retained rows. `kept` lists the full packed indices that are stored,
    /// ascending, and `bounds` are the full-space row bounds they are taken from.
    pub fn from_kept(num_baspar: usize, kept: &[usize], baspar2basbas: &[[usize; 2]], bounds: &[f64]) -> Self {
        let mut full2row = vec![Self::NOT_STORED; num_baspar];
        let mut rows = Vec::with_capacity(kept.len());
        let mut kept_bounds = Vec::with_capacity(kept.len());
        for (slot, &p) in kept.iter().enumerate() {
            let [mu, nu] = baspar2basbas[p];
            full2row[p] = slot as u32;
            rows.push([mu as u32, nu as u32]);
            kept_bounds.push(bounds[p]);
        }
        PairMap { full2row, rows, bounds: kept_bounds }
    }

    /// Number of stored rows.
    pub fn len(&self) -> usize {
        self.rows.len()
    }

    /// Whether the map is empty.
    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// `(mu, nu)` of a stored row.
    pub fn pair_of(&self, row: usize) -> (usize, usize) {
        let [mu, nu] = self.rows[row];
        (mu as usize, nu as usize)
    }

    /// Stored row of a full packed pair index, or `None` when that pair was dropped.
    pub fn row_of_full(&self, full_pair: usize) -> Option<usize> {
        match self.full2row[full_pair] {
            Self::NOT_STORED => None,
            row => Some(row as usize),
        }
    }

    /// Number of pairs of the full (unpruned) packed pair space.
    pub fn num_baspar_full(&self) -> usize {
        self.full2row.len()
    }

    /// Full packed pair indices of the stored rows, ascending.
    ///
    /// This is the inverse direction of [`PairMap::from_kept`], and it lets a second tensor that
    /// lives on the same AO-pair space (the short-range tensor of a range-separated hybrid, for
    /// instance) be compacted onto exactly the same rows.
    pub fn kept_of(&self, basbas2baspar: &MatrixFull<usize>) -> Vec<usize> {
        self.rows
            .iter()
            .map(|row| basbas2baspar[[row[0] as usize, row[1] as usize]])
            .collect()
    }

    /// Full packed pair indices of the stored rows, ascending.
    ///
    /// Same list as [`PairMap::kept_of`], read back from `full2row` instead of the pair table of a
    /// molecule. A consumer that has to project a packed AO-pair quantity onto the stored rows (the
    /// packed density of the derivative code, for instance) needs the packed index of every stored
    /// row, and this avoids rebuilding the `[nao, nao]` pair table of the molecule for that.
    pub fn kept_indices(&self) -> Vec<usize> {
        (0..self.full2row.len())
            .filter(|&p| self.full2row[p] != Self::NOT_STORED)
            .collect()
    }
}

/// Fail closed for a consumer that still indexes rimatr with the full-space pair tables.
///
/// A storage-level pruned tensor ([ctrl.ri_jk] pair_screen_threshold > 0) stores the retained
/// rows only, so a reader that walks the packed pair space with basbas2baspar reads *the wrong
/// rows* instead of failing, and returns a plausible number built from another pair. Every
/// consumer that has not been wired to [PairMap] yet calls this first and stops with the name of
/// the path that has to be wired.
pub fn require_unpruned_rimatr(map: &Option<PairMap>, consumer: &str) {
    if map.is_some() {
        panic!(
            "{} still reads the in-core RI tensor through the full-space pair tables, which is not \
             valid for a storage-level pruned tensor ([ctrl.ri_jk] pair_screen_threshold > 0). \
             Set pair_screen_threshold = 0.0 to run this path.",
            consumer
        );
    }
}

/// Fill the symmetric AO matrix of one auxiliary column from a stored tensor column.
///
/// The unpruned `rimatr` stores its rows in packed upper-triangular order, so a column can be
/// zipped straight onto the packed upper triangle. A storage-level pruned tensor stores only the
/// retained rows, and their `(mu, nu)` come from [`PairMap`]. The dropped rows stay at zero, which
/// is exactly the truncation the storage level applied, so every AO-to-MO consumer that reads the
/// column through this helper sees the same tensor the SCF kernels contract.
///
/// Callers allocate the output zeroed, so entries the map leaves out need no explicit clearing.
pub fn fill_ao_matrix_from_column(column: &[f64], map: Option<&PairMap>, out: &mut MatrixFull<f64>) {
    match map {
        None => {
            out.iter_matrixupper_mut()
                .unwrap()
                .zip(column.iter())
                .for_each(|(to, from)| *to = *from);
        }
        Some(map) => {
            debug_assert_eq!(
                column.len(),
                map.len(),
                "a compacted column carries one value per stored row"
            );
            for (row, value) in column.iter().enumerate() {
                let [mu, nu] = map.rows[row];
                let (i, j) = if mu <= nu {
                    (mu as usize, nu as usize)
                } else {
                    (nu as usize, mu as usize)
                };
                out[[i, j]] = *value;
            }
        }
    }
}

/// Result of the build-time (storage level) AO-pair pruning of `rimatr`.
///
/// `ri3fn` holds the retained rows only and [`PairMap`] carries the compact indexing and the
/// retained row bounds. The full-space tables of the tensor are not modified.
pub struct PrunedRIMatr {
    pub ri3fn: MatrixFull<f64>,
    pub map: PairMap,
}

/// Compact a `rimatr` onto the given rows, in place.
///
/// The tensor is consumed: the retained rows are moved to the front of every column and the
/// storage is shrunk, so no second copy of the tensor is ever allocated. That matters because the
/// full tensor is the dominant allocation of the in-core path, and a separate compacted copy would
/// raise the peak instead of lowering it.
///
/// `kept` lists the retained **full packed pair indices**, ascending. The in-place move is safe
/// because slot `i` of a column is written to position `i` while it is read from position
/// `kept[i] >= i`, so a write never lands on an entry that is still to be read.
pub fn compact_rimatr_rows(ri3fn: MatrixFull<f64>, kept: &[usize]) -> MatrixFull<f64> {
    let num_baspar = ri3fn.size()[0];
    let num_auxbas = ri3fn.size()[1];
    let num_kept = kept.len();
    assert!(num_kept <= num_baspar, "cannot keep more rows than the tensor has");
    assert!(kept.last().map_or(true, |&p| p < num_baspar), "retained row out of range");

    let mut data = ri3fn.data;
    if num_kept < num_baspar {
        for p_aux in 0..num_auxbas {
            let src = p_aux * num_baspar;
            let dst = p_aux * num_kept;
            for (slot, &p) in kept.iter().enumerate() {
                data[dst + slot] = data[src + p];
            }
        }
        data.truncate(num_kept * num_auxbas);
        data.shrink_to_fit();
    }

    MatrixFull::from_vec([num_kept, num_auxbas], data).unwrap()
}

/// A-priori keep list of a storage-level **pre-screen** of the AO-pair space.
///
/// The storage-level pruning of [`prune_rimatr`] can only run once the tensor exists, so the
/// three-centre integration and the metric transformation still pay for every pair. This function
/// produces a **conservative superset** of the rows that pruning will keep, from data available
/// before any three-centre integral is evaluated: the shell-pair Schwarz bound
/// `exp(q_cond) = sqrt(max_{mu in i, nu in j} |(mu nu|mu nu)|)` of [`PairList`], expanded to every
/// AO pair of the shell pair.
///
/// `tol` is a **relative** threshold on that bound. The caller is expected to pass a value well
/// below `[ctrl.ri_jk] pair_screen_threshold` (the production path uses a safety factor of ten):
/// the Schwarz bound only bounds the *untransformed* three-centre block, while pruning acts on the
/// row maximum of the *metric-transformed* tensor, so the two orderings differ and a cut of the
/// same nominal size is not guaranteed to be a superset.
///
/// Shell pairs that [`PairList::build`] drops through its diffuse-overlap pre-mask keep a bound of
/// zero here. That pre-mask uses `schwartz_overlap_tol2` (default 1e-24) on the overlap of the
/// most diffuse primitives, which is far below any relative cut this screen is used with, so a
/// pair it removes cannot be one the storage-level pruning would retain.
pub fn prescreen_pair_keep(
    cint_mol: &CInt,
    nao: usize,
    basbas2baspar: &MatrixFull<usize>,
    overlap_tol2: f64,
    tol: f64,
) -> Vec<usize> {
    let num_baspar = nao * (nao + 1) / 2;
    let pairs = super::PairList::build(cint_mol, overlap_tol2);

    let mut bound = vec![0.0_f64; num_baspar];
    for shell_pair in &pairs.pairs {
        let q = shell_pair.q_cond.exp();
        let (i0, i1) = (pairs.ao_loc[shell_pair.ish], pairs.ao_loc[shell_pair.ish + 1]);
        let (j0, j1) = (pairs.ao_loc[shell_pair.jsh], pairs.ao_loc[shell_pair.jsh + 1]);
        for mu in i0..i1 {
            for nu in j0..j1 {
                let p = basbas2baspar[[mu, nu]];
                if q > bound[p] {
                    bound[p] = q;
                }
            }
        }
    }

    let q_max = bound.iter().fold(0.0_f64, |acc, v| acc.max(*v));
    let cut = tol * q_max;
    (0..num_baspar).filter(|&p| bound[p] > cut).collect()
}

/// Row maxima `q[r] = max_P |R[r,P]|` of a tensor, in the row order of that tensor.
///
/// This is the per-row quantity both screening levels are built on.
/// [`RIMatrPairScreen::build`] augments it with the diagonal-pair flags of a full-space tensor; a
/// tensor that already lives on the row space of a pre-screen has no full-space pair table to
/// derive those from, so it uses this directly.
pub fn row_bounds(ri3fn: &MatrixFull<f64>) -> Vec<f64> {
    let num_rows = ri3fn.size()[0];
    let num_auxbas = ri3fn.size()[1];
    let data = ri3fn.data_ref().unwrap();

    // q[r] = max_P |R[r,P]|; column-major, so one row is a strided sweep over the columns
    let chunk = 512;
    let mut bounds = vec![0.0_f64; num_rows];
    bounds.par_chunks_mut(chunk).enumerate().for_each(|(ic, qs)| {
        let p0 = ic * chunk;
        for (io, q) in qs.iter_mut().enumerate() {
            let mut bound = 0.0_f64;
            let mut index = p0 + io;
            for _ in 0..num_auxbas {
                let value = data[index].abs();
                if value > bound {
                    bound = value;
                }
                index += num_rows;
            }
            *q = bound;
        }
    });
    bounds
}

/// Prune a tensor that already lives on the row space of a pre-screen.
///
/// `outer` gives the full packed pair index of every stored row, ascending, i.e. the row map that
/// [`prescreen_pair_keep`] produced for this geometry. The retained rows are chosen exactly as in
/// [`prune_rimatr`] (row bound above `tol` times the largest row bound), and the returned
/// [`PairMap`] describes them in the **full** packed pair space, so every consumer of the map sees
/// the layout it would have seen without the pre-screen.
pub fn prune_rimatr_prescreened(
    ri3fn: MatrixFull<f64>,
    bounds: &[f64],
    outer: &[usize],
    num_baspar_full: usize,
    baspar2basbas: &[[usize; 2]],
    tol: f64,
) -> PrunedRIMatr {
    let num_stored = ri3fn.size()[0];
    assert_eq!(bounds.len(), num_stored, "pair bounds do not match the pre-screened tensor");
    assert_eq!(outer.len(), num_stored, "pre-screen row map does not match the tensor");

    let q_max = bounds.iter().fold(0.0_f64, |acc, v| acc.max(*v));
    let cut = tol * q_max;
    let kept_local: Vec<usize> = (0..num_stored).filter(|&r| bounds[r] > cut).collect();

    // the map is addressed in the full packed pair space, so the bounds go back to that indexing
    let mut bounds_full = vec![0.0_f64; num_baspar_full];
    for (r, &p) in outer.iter().enumerate() {
        bounds_full[p] = bounds[r];
    }
    let kept_full: Vec<usize> = kept_local.iter().map(|&r| outer[r]).collect();
    let compacted = compact_rimatr_rows(ri3fn, &kept_local);
    let map = PairMap::from_kept(num_baspar_full, &kept_full, baspar2basbas, &bounds_full);
    PrunedRIMatr { ri3fn: compacted, map }
}

/// Drop the AO-pair rows of a freshly built `rimatr` whose static bound is negligible.
///
/// A row is kept when `q_p > tol * max_p q_p`, with `q_p` taken from `screen`. A tolerance
/// of `0.0` keeps every row with a nonzero bound, which reproduces the unpruned tensor.
///
/// **Under MPI the caller has to globalize `screen.bounds` first** (a `max` reduction plus a
/// broadcast): the auxiliary functions are distributed over the ranks, so a local bound is only a
/// partial maximum, and ranks that prune on local bounds would keep different row sets.
///
/// The peak memory of the build itself is unchanged, because the full tensor exists before the
/// pruning. Skipping the integration of the dropped shell pairs, which is what lowers the peak of
/// the build, is a later stage.
pub fn prune_rimatr(
    ri3fn: MatrixFull<f64>,
    screen: &RIMatrPairScreen,
    baspar2basbas: &[[usize; 2]],
    tol: f64,
) -> PrunedRIMatr {
    let num_baspar = ri3fn.size()[0];
    assert_eq!(screen.bounds.len(), num_baspar, "pair bounds do not match the RI tensor");

    let q_max = screen.bounds.iter().fold(0.0_f64, |acc, v| acc.max(*v));
    let cut = tol * q_max;
    let kept: Vec<usize> = (0..num_baspar).filter(|&p| screen.bounds[p] > cut).collect();

    // note the row bounds of the map are the **full-space** ones, gathered at the retained rows,
    // and the map itself is built before the tensor is consumed
    let map = PairMap::from_kept(num_baspar, &kept, baspar2basbas, &screen.bounds);
    let ri3fn_new = compact_rimatr_rows(ri3fn, &kept);

    PrunedRIMatr { ri3fn: ri3fn_new, map }
}

/// Compact a tensor that lives on the AO-pair space of an existing [`PairMap`].
///
/// The retained rows are the ones the map already describes, so the result shares the row space of
/// the map, and the row bounds of the map stay the screening bounds of the new tensor. Used for
/// the short-range tensor of a range-separated hybrid, which is indexed by the same AO pairs as
/// the full-range tensor: one shared selection keeps the two tensors of the SCF on one layout, and
/// the full-range bound is the conservative choice for the short-range operator.
pub fn prune_rimatr_to_map(
    ri3fn: MatrixFull<f64>,
    map: &PairMap,
    basbas2baspar: &MatrixFull<usize>,
) -> PrunedRIMatr {
    // `kept` holds full packed pair indices of this tensor, and the resulting row space is the one
    // the map already describes, so the map is carried over instead of being rebuilt: rebuilding it
    // from the stored-row bounds would index them with full-space pair numbers.
    let kept = map.kept_of(basbas2baspar);
    let ri3fn_new = compact_rimatr_rows(ri3fn, &kept);

    PrunedRIMatr { ri3fn: ri3fn_new, map: map.clone() }
}

/// Coalesce a sorted index list into maximal runs of consecutive indices.
///
/// The retained pairs of a screening step are sorted and strongly clustered (the shell pairs of
/// one atom occupy consecutive rows), so expressing them as runs turns both J passes into
/// contiguous slices instead of strided gathers.
pub fn sorted_runs(indices: &[usize]) -> Vec<Range<usize>> {
    let mut runs: Vec<Range<usize>> = Vec::new();
    for &p in indices {
        match runs.last_mut() {
            Some(run) if run.end == p => run.end = p + 1,
            _ => runs.push(p..p + 1),
        }
    }
    runs
}

/// Coulomb matrix J with basis-pair screening of a storage-level pruned RI tensor.
///
/// `map` describes the stored rows: `map.pair_of(row)` gives the AO pair and
/// `map.bounds[row]` its static bound. `dm` is the **full** density matrix, and the returned
/// vector is the packed upper triangle of J over the **full** pair space of `map` (length
/// `map.num_baspar_full()`): dropped rows contribute zeros, so the caller keeps the unmodified
/// `basbas2baspar` to assemble the Fock matrix. A threshold of `0.0` keeps every stored row.
pub fn get_vj_pair_screened(
    ri3fn: &MatrixFull<f64>,
    map: &PairMap,
    basbas2baspar: &MatrixFull<usize>,
    dm: &MatrixFull<f64>,
    threshold: f64,
    scale_density: f64,
    scale_geometric: f64,
) -> Vec<f64> {
    let num_stored = ri3fn.size()[0];
    let num_auxbas = ri3fn.size()[1];
    let num_baspar_full = map.num_baspar_full();
    assert_eq!(map.len(), num_stored, "pair map does not match the stored RI tensor");
    assert_eq!(map.bounds.len(), num_stored, "pair bounds do not match the stored RI tensor");
    let data = ri3fn.data_ref().unwrap();

    // pair weights of the density contraction, taken from the stored rows: both triangles of a
    // symmetric density contribute, so off-diagonal pairs count twice
    let weight: Vec<f64> = (0..num_stored).map(|p| j_row_weight(map, p, dm)).collect();

    // relative screening: a row is dropped when its bound falls below `threshold` times the
    // largest pair bound of this step, so the criterion is scale covariant and independent of the
    // absolute size of the density (the same reasoning applies to the K criterion below)
    let cut = threshold * scale_density;
    let retained: Vec<usize> = (0..num_stored).filter(|&p| weight[p].abs() * map.bounds[p] > cut).collect();
    let runs = sorted_runs(&retained);
    debug!(
        "pair-screened RI-J: {} of {} stored pairs in the density contraction (relative threshold {:.1e})",
        retained.len(),
        num_stored,
        threshold
    );

    // pass 1: auxiliary vectors g[P] = sum_p w_p R[p,P]
    let mut g = vec![0.0_f64; num_auxbas];
    g.par_iter_mut().enumerate().for_each(|(p_aux, g_p)| {
        let column = &data[p_aux * num_stored..(p_aux + 1) * num_stored];
        let mut acc = 0.0_f64;
        for run in &runs {
            let (a, b) = (run.start, run.end);
            for i in a..b {
                acc += column[i] * weight[i];
            }
        }
        *g_p = acc;
    });

    // pass 2: J[p] = sum_P R[p,P] g[P]; its bound is q[p] * max_P |g[P]|, and the constant factor
    // max|g| cancels in the relative criterion, so this selection is static per geometry
    let cut_out = threshold * scale_geometric;
    let retained_out: Vec<usize> = (0..num_stored).filter(|&p| map.bounds[p] > cut_out).collect();
    let runs_out = sorted_runs(&retained_out);
    let num_retained = retained_out.len();

    let chunks = crate::utilities::balancing(num_auxbas, rayon::current_num_threads());
    let partial: Vec<Vec<f64>> = chunks
        .par_iter()
        .map(|columns| {
            let mut local = vec![0.0_f64; num_retained];
            for p_aux in columns.clone() {
                let column = &data[p_aux * num_stored..(p_aux + 1) * num_stored];
                let g_p = g[p_aux];
                let mut offset = 0;
                for run in &runs_out {
                    let (a, b) = (run.start, run.end);
                    for (value, r) in local[offset..offset + (b - a)].iter_mut().zip(&column[a..b]) {
                        *value += r * g_p;
                    }
                    offset += b - a;
                }
            }
            local
        })
        .collect();

    let mut accumulator = vec![0.0_f64; num_retained];
    for local in &partial {
        for (acc, value) in accumulator.iter_mut().zip(local) {
            *acc += value;
        }
    }

    // scatter back into the full packed upper triangle; the dropped rows stay zero
    let mut j_upper = vec![0.0_f64; num_baspar_full];
    for (i, &p) in retained_out.iter().enumerate() {
        let (mu, nu) = map.pair_of(p);
        j_upper[basbas2baspar[[mu, nu]]] = accumulator[i];
    }

    j_upper
}

/// Exchange matrix K with basis-pair screening of a storage-level pruned RI tensor.
///
/// `map` describes the stored rows, see [`get_vj_pair_screened`]. `eigv_reduced` holds the
/// occupied orbitals with the square root of their occupation folded in, shape `[nao, nw]` (the
/// same object the unscreened kernel contracts). The returned matrix is the packed upper triangle
/// of K over the full AO space, in the `MatrixUpper` ordering of the RI tensor.
pub fn get_vk_pair_screened(
    ri3fn: &MatrixFull<f64>,
    map: &PairMap,
    eigv_reduced: &MatrixFull<f64>,
    w_ao: &[f64],
    threshold: f64,
    scale_orbital: f64,
) -> MatrixUpper<f64> {
    let num_stored = ri3fn.size()[0];
    let num_auxbas = ri3fn.size()[1];
    let nao = eigv_reduced.size()[0];
    let nw = eigv_reduced.size()[1];
    assert_eq!(map.len(), num_stored, "pair map does not match the stored RI tensor");
    let data = ri3fn.data_ref().unwrap();
    let coeff = eigv_reduced.data_ref().unwrap();

    // retained rows and the active AO subset they span; the criterion is relative to the
    // strongest pair bound of this iteration (see the J kernel)
    let paired_bound: Vec<f64> = (0..num_stored)
        .map(|p| {
            let (mu, nu) = map.pair_of(p);
            map.bounds[p] * w_ao[mu].max(w_ao[nu])
        })
        .collect();
    let cut = threshold * scale_orbital;
    let retained: Vec<(usize, usize, usize)> = (0..num_stored)
        .filter_map(|p| {
            if paired_bound[p] > cut {
                let (mu, nu) = map.pair_of(p);
                Some((p, mu, nu))
            } else {
                None
            }
        })
        .collect();

    if retained.is_empty() {
        // no stored row can contribute: the exchange matrix of this spin channel is zero, and the
        // zero-size BLAS calls below are better avoided
        return MatrixUpper::new(nao * (nao + 1) / 2, 0.0_f64);
    }

    let mut is_active = vec![false; nao];
    for &(_, mu, nu) in &retained {
        is_active[mu] = true;
        is_active[nu] = true;
    }
    let mut active_of = vec![usize::MAX; nao];
    let mut active: Vec<usize> = Vec::new();
    for (mu, flag) in is_active.iter().enumerate() {
        if *flag {
            active_of[mu] = active.len();
            active.push(mu);
        }
    }
    let num_active = active.len();
    debug!(
        "pair-screened RI-K: {} of {} stored pairs retained, {} of {} AO active (relative threshold {:.1e})",
        retained.len(),
        num_stored,
        num_active,
        nao,
        threshold
    );

    // occupied orbitals on the active subset
    let mut coeff_active = MatrixFull::new([num_active, nw], 0.0_f64);
    for (ia, &mu) in active.iter().enumerate() {
        for i in 0..nw {
            coeff_active[[ia, i]] = coeff[mu + nao * i];
        }
    }

    // retained rows in active-subset coordinates
    let retained_active: Vec<(usize, usize, usize)> =
        retained.iter().map(|&(p, mu, nu)| (p, active_of[mu], active_of[nu])).collect();

    let chunks = crate::utilities::balancing(num_auxbas, rayon::current_num_threads());
    let partial: Vec<MatrixFull<f64>> = chunks
        .par_iter()
        .map(|columns| {
            let mut sub = MatrixFull::new([num_active, num_active], 0.0_f64);
            let mut tmp = MatrixFull::new([num_active, nw], 0.0_f64);
            let mut accum = MatrixFull::new([num_active, num_active], 0.0_f64);
            for p_aux in columns.clone() {
                let column = &data[p_aux * num_stored..(p_aux + 1) * num_stored];
                // upper triangle of the retained rows only, dsymm reads the upper triangle
                for &(p, ia, ja) in &retained_active {
                    let (lo, hi) = if ia <= ja { (ia, ja) } else { (ja, ia) };
                    sub[[lo, hi]] = column[p];
                }
                // T = R_P C on the active subset, then K += T T^t
                _dsymm(&sub, &coeff_active, &mut tmp, 'L', 'U', 1.0, 0.0);
                _dsyrk(&tmp, &mut accum, 'U', 'N', 1.0, 1.0);
                for &(_, ia, ja) in &retained_active {
                    let (lo, hi) = if ia <= ja { (ia, ja) } else { (ja, ia) };
                    sub[[lo, hi]] = 0.0;
                }
            }
            accum
        })
        .collect();

    // scatter the active block into the full symmetric matrix and pack it
    let mut k_full = MatrixFull::new([nao, nao], 0.0_f64);
    for accum in &partial {
        for (ja, &nu) in active.iter().enumerate() {
            for (ia, &mu) in active.iter().enumerate().take(ja + 1) {
                k_full[[mu, nu]] += accum[[ia, ja]];
            }
        }
    }
    k_full.to_matrixupper()
}
