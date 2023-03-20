#![allow(clippy::many_single_char_names)]

use pyo3::prelude::*;

use std::collections::HashMap;
use std::f64::consts::PI;

use super::bragg;
use super::bse;
use super::parameters;
use statrs::function::gamma;

#[cfg(test)]
use super::comparison;

#[pyfunction]
pub fn radial_grid_kk(num_points: usize) -> (Vec<f64>, Vec<f64>) {
    let n = num_points as i32;
    let mut rws: Vec<_> = (1..=n).map(|i| kk_r_w(i, n)).collect();
    rws.reverse();
    rws.iter().cloned().unzip()
}

// https://doi.org/10.1063/1.475719, eqs. 9-13
fn kk_r_w(i: i32, n: i32) -> (f64, f64) {
    let pi = std::f64::consts::PI;

    let angle = ((i as f64) * pi) / ((n + 1) as f64);
    let s = angle.sin();
    let c = angle.cos();
    let t1 = ((n + 1 - 2 * i) as f64) / ((n + 1) as f64);
    let t2 = 2.0 / pi;
    let x = t1 + t2 * c * s * (1.0 + (2.0 / 3.0) * s * s);
    let f = 1.0 / 2.0_f64.ln();
    let r = f * (2.0 / (1.0 - x)).ln();
    let w = r * r * (f / (1.0 - x)) * (16.0 * s * s * s * s) / (3.0 * (n as f64) + 3.0);

    (r, w)
}

#[test]
fn test_radial_grid_kk() {
    let (rs, ws) = radial_grid_kk(99);
    assert!(comparison::floats_are_same(
        rs[98],
        27.520865062836048,
        1.0e-15
    ));
    assert!(comparison::floats_are_same(
        rs[97],
        22.522899235303480,
        1.0e-15
    ));
    assert!(comparison::floats_are_same(
        rs[0],
        7.4914976443367854e-9,
        1.0e-15
    ));
    assert!(comparison::floats_are_same(
        ws[98],
        5462.4446669497620,
        1.0e-15
    ));
    assert!(comparison::floats_are_same(
        ws[97],
        1828.2535433191961,
        1.0e-15
    ));
    assert!(comparison::floats_are_same(
        ws[0],
        2.1018140707490095e-24,
        1.0e-15
    ));
}

#[pyfunction]
pub fn radial_grid_lmg_bse(
    basis_set: &str,
    radial_precision: f64,
    proton_charge: i32,
) -> (Vec<f64>, Vec<f64>) {
    let (alpha_min, alpha_max) = bse::ang_min_and_max(basis_set, proton_charge as usize);

    radial_grid_lmg(alpha_min, alpha_max, radial_precision, proton_charge)
}

#[pyfunction]
pub fn radial_grid_lmg(
    alpha_min: HashMap<usize, f64>,
    alpha_max: f64,
    radial_precision: f64,
    proton_charge: i32,
) -> (Vec<f64>, Vec<f64>) {
    // factor 2.0 to match DIRAC code
    let r_inner = get_r_inner(radial_precision, alpha_max * 2.0);

    let mut h = std::f64::MAX;
    let mut r_outer: f64 = 0.0;

    // we need alpha_min sorted by l
    // at this point not sure why ... need to check again the literature
    let mut v: Vec<_> = alpha_min.into_iter().collect();
    v.sort_by(|x, y| x.0.cmp(&y.0));

    for (l, a) in v {
        if a > 0.0 {
            r_outer = r_outer.max(get_r_outer(
                radial_precision,
                a,
                l,
                4.0 * bragg::get_bragg_angstrom(proton_charge),
            ));
            assert!(r_outer > r_inner);
            h = h.min(get_h(radial_precision, l, 0.1 * (r_outer - r_inner)));
        }
    }
    assert!(r_outer > h);

    let c = r_inner / (h.exp() - 1.0);
    let num_points = ((1.0 + (r_outer / c)).ln() / h) as usize;

    let mut rs = Vec::new();
    let mut ws = Vec::new();

    for i in 1..=num_points {
        let r = c * (((i as f64) * h).exp() - 1.0);
        rs.push(r);
        ws.push((r + c) * r * r * h);
    }

    (rs, ws)
}

pub fn radial_grid_gc2nd (n: usize) -> (Vec<f64>, Vec<f64>)  {
    let ln2 = 1.0 / f64::ln(2.0);
    let fac = (16.0 / 3.0) / ((n+1) as f64);
    let mut x1 = vec![];
    let mut xi = vec![];

    for i in 1..n+1 {
        x1.push((i as f64)*PI / ((n+1) as f64));
    };

    for i in 1..n+1 {
        let a = ((n as isize)-1-2*((i-1) as isize)) as f64;
        let b = (1.0+2.0/3.0*f64::sin(x1[i-1]).powf(2.0))*f64::sin(2.0*x1[i-1]) / PI;
        xi.push(
            a / ((n+1) as f64) + b
        );
    };  // is there a mistake in xi? Should be (n+1-2i) not (n-1-2i)

    let xi_rev: Vec<&f64> = xi.iter().rev().collect();
    let xi_new: Vec<f64> = xi.iter().zip(xi_rev.iter()).map(|(xi, xi_rev)| (xi - *xi_rev) / 2.0 ).collect();
    let r: Vec<f64> = xi_new.iter().map(|xi_new| 1.0 - f64::ln(1.0 + xi_new) * ln2).collect();
    let w: Vec<f64> = xi_new.iter().zip(x1.iter()).map(|(xi_new, x1)| fac * f64::sin(*x1).powf(4.0) * ln2 / (1.0 + xi_new)).collect();
    return (r,w);

    /* 
    # Gauss-Chebyshev of the second kind,  and the transformed interval [0,\infty)
# Ref  Matthias Krack and Andreas M. Koster,  J. Chem. Phys. 108 (1998), 3226

 */

}

pub fn radial_grid_treutler(n: usize) -> (Vec<f64>, Vec<f64>) {
    //treutler_ahlrichs
    //Treutler-Ahlrichs [JCP 102, 346 (1995); DOI:10.1063/1.469408] (M4) radial grids
    let mut r: Vec<f64> = vec![];
    let mut w: Vec<f64> = vec![];
    let step = PI / ((n+1) as f64);
    let ln2 = 1.0 / f64::ln(2.0);
    for i in 1..n+1 {
        let x = f64::cos((i as f64)*step);
        r.push(-ln2*(1.0+x).powf(0.6)*f64::ln((1.0-x)/2.0));
        w.push(step*f64::sin((i as f64)*step)*ln2*(1.0+x).powf(0.6)*(-0.6/(1.0+x)*f64::ln((1.0-x)/2.0)+1.0/(1.0-x)))
    }
    let r_rev: Vec<f64> = r.into_iter().rev().collect();
    let w_rev: Vec<f64> = w.into_iter().rev().collect();
    return (r_rev, w_rev)

}
/* 
def treutler_ahlrichs(n, *args, **kwargs):
    '''

    '''
    r = numpy.empty(n)
    dr = numpy.empty(n)
    step = numpy.pi / (n+1)
    ln2 = 1 / numpy.log(2)
    for i in range(n):
        x = numpy.cos((i+1)*step)
        r [i] = -ln2*(1+x)**.6 * numpy.log((1-x)/2)
        dr[i] = step * numpy.sin((i+1)*step) \
                * ln2*(1+x)**.6 *(-.6/(1+x)*numpy.log((1-x)/2)+1/(1-x))
    return r[::-1], dr[::-1]
treutler = treutler_ahlrichs
 */






// TCA 106, 178 (2001), eq. 25
// we evaluate r_inner for s functions
fn get_r_inner(max_error: f64, alpha_inner: f64) -> f64 {
    let d = 1.9;

    let mut r = d - (1.0 / max_error).ln();
    r *= 2.0 / 3.0;
    r = r.exp() / alpha_inner;
    r = r.sqrt();

    r
}

#[test]
fn test_get_r_inner() {
    assert!(comparison::floats_are_same(
        get_r_inner(1.0e-12, 2.344e4),
        0.0000012304794589759454,
        1.0e-15
    ));
    assert!(comparison::floats_are_same(
        get_r_inner(1.0e-12, 2.602e1),
        0.000036931719276091795,
        1.0e-15
    ));
}

// TCA 106, 178 (2001), eq. 19
fn get_r_outer(max_error: f64, alpha_outer: f64, l: usize, guess: f64) -> f64 {
    let m = (2 * l) as f64;
    let mut r_old = std::f64::MAX;
    let mut step = 0.5;
    let mut sign = 1.0;
    let mut r = guess;

    while (r_old - r).abs() > parameters::SMALL {
        let c = gamma::gamma((m + 3.0) / 2.0);
        let a = (alpha_outer * r * r).powf((m + 1.0) / 2.0);
        let e = (-alpha_outer * r * r).exp();
        let f = c * a * e;

        let sign_old = sign;
        sign = if f > max_error { 1.0 } else { -1.0 };
        if r < 0.0 {
            sign = 1.0
        }
        if sign * sign_old < 0.0 {
            step *= 0.1;
        }

        r_old = r;
        r += sign * step;
    }

    r
}

#[test]
fn test_get_r_outer() {
    assert!(comparison::floats_are_same(
        get_r_outer(1.0e-12, 0.3023, 0, 2.4),
        9.827704700468292,
        1.0e-15
    ));
    assert!(comparison::floats_are_same(
        get_r_outer(1.0e-12, 0.2753, 1, 2.4),
        10.97632862649149,
        1.0e-15
    ));
    assert!(comparison::floats_are_same(
        get_r_outer(1.0e-12, 1.1850, 2, 2.4),
        5.656909461147809,
        1.0e-15
    ));
    assert!(comparison::floats_are_same(
        get_r_outer(1.0e-12, 0.1220, 0, 1.4),
        15.470033591458682,
        1.0e-15
    ));
    assert!(comparison::floats_are_same(
        get_r_outer(1.0e-12, 0.7270, 1, 1.4),
        6.75449681779016,
        1.0e-15
    ));
    assert!(comparison::floats_are_same(
        get_r_outer(1.0e-12, 0.1220, 0, 1.4),
        15.470033591458682,
        1.0e-15
    ));
    assert!(comparison::floats_are_same(
        get_r_outer(1.0e-12, 0.7270, 1, 1.4),
        6.75449681779016,
        1.0e-15
    ));
}

// TCA 106, 178 (2001), eqs. 17 and 18
fn get_h(max_error: f64, l: usize, guess: f64) -> f64 {
    let m = (2 * l) as f64;
    let mut h_old = std::f64::MAX;
    let mut h = guess;
    let mut step = 0.1 * guess;
    let mut sign = -1.0;

    let pi = std::f64::consts::PI;

    while (h_old - h).abs() > parameters::SMALL {
        let c0 = 4.0 * (2.0 as f64).sqrt() * pi;
        let cm = gamma::gamma(3.0 / 2.0) / gamma::gamma((m + 3.0) / 2.0);
        let p0 = 1.0 / h;
        let e0 = (-pi * pi / (2.0 * h)).exp();
        let pm = (pi / h).powf(m / 2.0);
        let rd0 = c0 * p0 * e0;
        let f = cm * pm * rd0;

        let sign_old = sign;
        sign = if f > max_error { -1.0 } else { 1.0 };
        if h < 0.0 {
            sign = 1.0
        }
        if sign * sign_old < 0.0 {
            step *= 0.1;
        }

        h_old = h;
        h += sign * step;
    }

    h
}

#[test]
fn test_get_h() {
    assert!(comparison::floats_are_same(
        get_h(1.0e-12, 0, 0.9827703),
        0.1523549756417546,
        1.0e-15
    ));
    assert!(comparison::floats_are_same(
        get_h(1.0e-12, 1, 1.0976330),
        0.1402889524647394,
        1.0e-15
    ));
    assert!(comparison::floats_are_same(
        get_h(1.0e-12, 2, 1.0976330),
        0.13136472924710518,
        1.0e-15
    ));
    assert!(comparison::floats_are_same(
        get_h(1.0e-12, 0, 1.5470000),
        0.15235497564177505,
        1.0e-15
    ));
    assert!(comparison::floats_are_same(
        get_h(1.0e-12, 1, 1.5470000),
        0.14028895246475018,
        1.0e-15
    ));
    assert!(comparison::floats_are_same(
        get_h(1.0e-12, 0, 1.5470000),
        0.15235497564177505,
        1.0e-15
    ));
    assert!(comparison::floats_are_same(
        get_h(1.0e-12, 1, 1.5470000),
        0.14028895246475018,
        1.0e-15
    ));
}
