use std::fs;
use std::io::{BufWriter, Write};
use rest_libcint::{CINTR2CDATA, CintType};
use regex::Regex;
use crate::constants::{SPECIES_INFO};
#[cfg(feature = "librest2fch")]
use crate::external_libs::py2fch;
use crate::scf_io::SCF;
use crate::scf_io::SCFType;
use crate::x2c::RelativisticMethod;
use tensors::MatrixFull;

// ============================================================================
// fchk numeric layout
//
// A real field is a fixed-width Fortran column, five columns per line, with the
// exponent re-padded to at least two digits (`E+00` / `E-05`). The historical
// pipeline pushed every value through `format!`, a `Regex::new` compiled *inside*
// the conversion function, and a freshly allocated `String`, then wrote it with
// one `write!` per value into an unbuffered `File`. For a job with a few thousand
// basis functions the density sections alone are millions of values, i.e.
// millions of regex compilations, millions of heap allocations and millions of
// `write(2)` calls. The helpers below emit exactly the same bytes with no heap
// allocation and one `write_all` per line.
// ============================================================================

/// Real columns per line in the fchk layout.
const FCHK_REAL_PER_LINE: usize = 5;
/// Field width of the density / MO / orbital-energy / basis columns.
const FCHK_REAL_WIDTH: usize = 16;
/// Decimals of the density / MO / orbital-energy / basis columns.
const FCHK_REAL_DECIMALS: usize = 8;
/// Field width of the `SCF Energy` / `Total Energy` values.
const FCHK_ENERGY_WIDTH: usize = 27;
/// Decimals of the `SCF Energy` / `Total Energy` values.
const FCHK_ENERGY_DECIMALS: usize = 15;
/// Write buffer size; a whole density section then costs a handful of syscalls.
const FCHK_BUFFER_SIZE: usize = 1 << 20;

/// A `fmt::Write` sink over a fixed stack buffer — the allocation-free stand-in
/// for `format!`.
struct StackSink<'a> {
    buf: &'a mut [u8],
    len: usize,
}

impl std::fmt::Write for StackSink<'_> {
    fn write_str(&mut self, s: &str) -> std::fmt::Result {
        let bytes = s.as_bytes();
        if self.len + bytes.len() > self.buf.len() {
            return Err(std::fmt::Error);
        }
        self.buf[self.len..self.len + bytes.len()].copy_from_slice(bytes);
        self.len += bytes.len();
        Ok(())
    }
}

/// Render `x` into `out` (whose length is the field width), reproducing the
/// historical `r2f(&format!("{:width.decimals$E}", x))` pipeline byte for byte:
/// the mantissa keeps `decimals` decimals, the exponent is zero-padded to at
/// least two digits with an explicit sign, and the content is right-aligned in
/// `out.len()` characters — truncated from the left when it does not fit.
///
/// The one intentional difference is that an exponent of three digits
/// (|exp| >= 100) is now written in full instead of being truncated to two
/// digits, which made the historical `out_str[n_len - o_len..]` slice underflow
/// and abort the job on a legitimate finite value.
fn fchk_format_real(x: f64, decimals: usize, out: &mut [u8]) {
    let width = out.len();
    if !x.is_finite() {
        // The historical pipeline panicked here too (the regex did not match
        // "NaN"/"inf"); keep failing loudly rather than writing garbage into a
        // file consumed by external tooling.
        panic!("fchk: cannot write the non-finite value {x:?} to the formatted checkpoint");
    }
    let mut scratch = [0u8; 48];
    let len = {
        let mut sink = StackSink { buf: &mut scratch, len: 0 };
        // `std::io::Write` is the trait in scope for the file sinks, so the
        // formatting sink is driven through `std::fmt::Write` explicitly.
        std::fmt::Write::write_fmt(&mut sink, format_args!("{:.*E}", decimals, x))
            .expect("fchk real scratch buffer is too small");
        sink.len
    };
    let rendered = std::str::from_utf8(&scratch[..len]).expect("a formatted real is ASCII");
    let epos = rendered
        .as_bytes()
        .iter()
        .position(|&b| b == b'E')
        .expect("the upper-exponential format always emits an exponent");
    let (mantissa, exponent) = rendered.split_at(epos);
    let exponent: i32 = exponent[1..]
        .parse()
        .expect("the upper-exponential format always emits an integer exponent");
    let (sign, mut magnitude) = if exponent < 0 {
        (b'-', (-exponent) as u32)
    } else {
        (b'+', exponent as u32)
    };

    // `mantissaE±dd`, unpadded, in a second stack buffer.
    let mut content = [0u8; 40];
    let mut n = mantissa.len();
    content[..n].copy_from_slice(mantissa.as_bytes());
    content[n] = b'E';
    content[n + 1] = sign;
    n += 2;
    let mut digits = [0u8; 3];
    let mut ndigits = 0;
    loop {
        digits[ndigits] = b'0' + (magnitude % 10) as u8;
        ndigits += 1;
        magnitude /= 10;
        if magnitude == 0 {
            break;
        }
    }
    while ndigits < 2 {
        digits[ndigits] = b'0';
        ndigits += 1;
    }
    for i in (0..ndigits).rev() {
        content[n] = digits[i];
        n += 1;
    }

    if n >= width {
        out.copy_from_slice(&content[n - width..n]);
    } else {
        let pad = width - n;
        out[..pad].fill(b' ');
        out[pad..].copy_from_slice(&content[..n]);
    }
}

/// Write `data` as fchk real columns, five per line — the buffered,
/// allocation-free replacement for the `dump_real_r2f!` per-element loop.
fn write_real_fields<W: Write>(input: &mut W, data: &[f64]) -> std::io::Result<()> {
    let mut line = [b' '; (FCHK_REAL_WIDTH + 1) * FCHK_REAL_PER_LINE];
    for chunk in data.chunks(FCHK_REAL_PER_LINE) {
        let mut n = 0;
        for &x in chunk {
            fchk_format_real(x, FCHK_REAL_DECIMALS, &mut line[n..n + FCHK_REAL_WIDTH]);
            n += FCHK_REAL_WIDTH;
        }
        line[n] = b'\n';
        input.write_all(&line[..n + 1])?;
    }
    Ok(())
}

/// Write a single fchk real value in a `width`-character column (the
/// `SCF Energy` / `Total Energy` lines).
fn write_real_value<W: Write>(input: &mut W, x: f64) -> std::io::Result<()> {
    let mut field = [b' '; 32];
    fchk_format_real(x, FCHK_ENERGY_DECIMALS, &mut field[..FCHK_ENERGY_WIDTH]);
    input.write_all(&field[..FCHK_ENERGY_WIDTH])
}

macro_rules! dump_real_r2f {
    ($input:expr, $data:expr) => {
        write_real_fields(&mut $input, &$data).unwrap()
    }
}

macro_rules! dump_int {
    ($input:expr, $data:expr) => {
    let mut i_index = 0;
    $data.iter().for_each(|x| {
        if (i_index + 1)%6 == 0 {
            write!($input, "{:12}\n", x)
        } else {
            write!($input, "{:12}", x)
        };
        i_index += 1;
    });
    if i_index % 6 != 0 {write!($input, "\n");}
}
}


impl SCF {
    pub fn save_fchk_of_gaussian(&self) {
        println!("\nSave fch file");
        self.create_fchk_head();
        self.fchk_write_mo();
        // the DH density dump (see `SCF::dh_rdm1_resp`); controlled by
        // `[output] fchk_dm`: with the default "auto" it is dumped whenever
        // it has been calculated, "scf" and false omit it
        if let Some(sections) = self.dh_rdm1_resp.as_ref() {
            if self.mol.ctrl.output.fchk_dm.as_str() == "auto" {
                let labels = ["Total MP2 Density", "Spin MP2 Density"];
                for (label, dm) in labels.iter().zip(sections.iter()) {
                    self.fchk_append_density_ao(label, dm);
                }
            }
        }
    }

    pub fn create_fchk_head(&self) {
        let re_basis = Regex::new(r"/?(?P<basis>[^/]*)/?$").unwrap();
        let cap = re_basis.captures(&self.mol.ctrl.basis_path).unwrap();
        let basis_name = cap.name("basis").unwrap().as_str().to_string();
        let file = fs::File::create(&format!("{}.fchk",&self.mol.geom.name)).unwrap();
        let mut input = BufWriter::with_capacity(FCHK_BUFFER_SIZE, file);
        write!(input, "{:} generated by REST\n", &self.mol.geom.name);
        if let SCFType::ROHF = self.scftype {
            write!(input, "SP        RO{:-58}{:-20}\n",
                self.mol.ctrl.xc.to_uppercase(),
                basis_name.to_uppercase()
            );
        } else if self.mol.spin_channel==1 {
            write!(input, "SP        R{:-59}{:-20}\n",
                self.mol.ctrl.xc.to_uppercase(),
                basis_name.to_uppercase()
            );
        } else {
            write!(input, "SP      U{:-59}{:-20}\n",
                self.mol.ctrl.xc.to_uppercase(),
                basis_name.to_uppercase()
            );
        };
        let natom = self.mol.geom.elem.len();
        write!(input, "Number of atoms                            I {:16}\n", self.mol.geom.elem.len());
        let mut route_tokens: Vec<String> = vec![String::from("#p")];
        route_tokens.push(format!("{}/{}",
            self.mol.ctrl.xc.to_uppercase(),
            basis_name.to_uppercase()
        ));
        if let Some(disp) = &self.mol.ctrl.empirical_dispersion {
            let em = match disp.as_str() {
                "d3" => "GD3",
                "d3bj" => "GD3BJ",
                "d4" => "GD4",
                other => other,
            };
            route_tokens.push(format!("em={}", em));
        }
        match self.mol.ctrl.rel {
            RelativisticMethod::SFX2C => route_tokens.push(String::from("int(nobasistransform,X2C)")),
            _ => route_tokens.push(String::from("int=nobasistransform")),
        };
        write!(input, "Route                                      C   N={:12}\n", route_tokens.len());
        write!(input, "{}\n", route_tokens.join(" "));
        write!(input, "Charge                                     I {:16}\n", self.mol.ctrl.charge as i32);
        write!(input, "Multiplicity                               I {:16}\n", self.mol.ctrl.spin as i32);
        write!(input, "Number of electrons                        I {:16}\n", self.mol.num_elec[0] as i32);
        write!(input, "Number of alpha electrons                  I {:16}\n", self.mol.num_elec[1] as i32);
        write!(input, "Number of beta electrons                   I {:16}\n", self.mol.num_elec[2] as i32);
        write!(input, "Number of basis functions                  I {:16}\n", self.mol.num_basis);
        write!(input, "Number of independent functions            I {:16}\n", self.mol.num_state);
        // ==============================
        // Now for atomic numbers
        // ==============================
        write!(input, "Atomic numbers                             I   N={:12}\n", natom);
        let mut i_index = 0;
        self.mol.geom.elem.iter().for_each(|x| {
            let (mass, charge) = SPECIES_INFO.get(x.as_str()).unwrap();
            if (i_index + 1)%6 == 0 {
                write!(input, "{:12}\n",*charge as i32)
            } else {
                write!(input, "{:12}",*charge as i32)
            };
            i_index += 1;
        });
        if i_index % 6 != 0 {write!(input, "\n");}
        // ==============================
        // Now for Nuclear charges
        // ==============================
        write!(input, "Nuclear charges                            R   N={:12}\n", natom);
        let nuclear_charges: Vec<f64> = self.mol.geom.elem.iter()
            .map(|x| SPECIES_INFO.get(x.as_str()).unwrap().1)
            .collect();
        write_real_fields(&mut input, &nuclear_charges).unwrap();
        // ==============================
        // Now for geometry coordinates
        // ==============================
        write!(input, "Current cartesian coordinates              R   N={:12}\n", natom*3);
        let mut coordinates: Vec<f64> = Vec::with_capacity(natom * 3);
        self.mol.geom.position.iter_columns_full().for_each(|x_position| {
            coordinates.extend(x_position.iter().copied());
        });
        write_real_fields(&mut input, &coordinates).unwrap();
        // ==============================
        // Now for atomic weights
        // ==============================
        write!(input, "Real atomic weights                        R   N={:12}\n", natom);
        let atomic_weights: Vec<f64> = self.mol.geom.elem.iter()
            .map(|x| SPECIES_INFO.get(x.as_str()).unwrap().0)
            .collect();
        write_real_fields(&mut input, &atomic_weights).unwrap();
        // ==============================
        // Now for MicOpt
        // ==============================
        write!(input, "MicOpt                                     I   N={:12}\n", natom);
        let mut i_index = 0;
        self.mol.geom.elem.iter().for_each(|x| {
            if (i_index + 1)%6 == 0 {
                write!(input, "{:12}\n", -1)
            } else {
                write!(input, "{:12}", -1)
            };
            i_index += 1;
        });
        if i_index % 6 != 0 {write!(input, "\n");}
        //================================
        // Now for basis set info.
        //================================
        let mut num_contract = 0;
        let mut num_primitiv = 0;
        let mut num_d_shell = 0;
        let mut num_f_shell = 0;
        let mut max_ang = 0;
        let mut max_contract = 0;
        let mut shell_type: Vec<i32> = vec![];
        let mut num_primitiv_vec: Vec<usize> = vec![];
        let mut shell_to_atom_map: Vec<usize> = vec![];
        let mut primitive_exp: Vec<f64> = vec![];
        let mut coord_each_shell: Vec<f64> = vec![];
        let mut coeff_vec:Vec<f64> = vec![];
        let shell_type_fac = match self.mol.cint_type {
            CintType::Spheric => -1,
            CintType::Cartesian => 1,
            // NOTE: Spinor is not yet implemented properly, IGOR 2024-08-18
            CintType::Spinor => 1,
        };
        self.mol.basis4elem.iter().enumerate().for_each(|(i_atom,ibas)| {
            ibas.electron_shells.iter().for_each(|ibascell| {
                //let mut tmp_coeff_vec =  ibascell.coefficients.clone();
                ibascell.coefficients.iter().for_each(|coeff_i| {
                    // De-normalization
                    let mut tmp_coeff_i = coeff_i.clone();
                    let tmp_ang = ibascell.angular_momentum[0];
                    tmp_coeff_i.iter_mut().zip(ibascell.exponents.iter()).for_each(|(i, e)| {
                        *i /= CINTR2CDATA::gto_norm(tmp_ang, *e);
                    });
                    coeff_vec.extend(tmp_coeff_i.iter());
                    num_contract += 1;
                    num_primitiv += ibascell.exponents.len();
                    max_ang = std::cmp::max(max_ang, tmp_ang);
                    max_contract = std::cmp::max(max_contract, ibascell.exponents.len());
                    if ibascell.angular_momentum[0] == 2 {num_d_shell += 1};
                    if ibascell.angular_momentum[0] == 3 {num_f_shell += 1};
                    let tmp_shell_type = if ibascell.angular_momentum[0] == 1 {
                        ibascell.angular_momentum[0]
                    } else {
                        ibascell.angular_momentum[0]*shell_type_fac
                    };
                    shell_type.push(tmp_shell_type);
                    num_primitiv_vec.push(ibascell.exponents.len());
                    shell_to_atom_map.push(i_atom+1);
                    primitive_exp.extend(ibascell.exponents.iter());
                    coord_each_shell.extend(self.mol.geom.rg_position.iter_column(i_atom));
                });
            });
        });
        write!(input, "Int Atom Types                             I   N={:12}\n", natom);
        let mut i_index = 0;
        self.mol.geom.elem.iter().for_each(|x| {
            if (i_index + 1)%6 == 0 {
                write!(input, "{:12}\n", 0)
            } else {
                write!(input, "{:12}", 0)
            };
            i_index += 1;
        });
        if i_index % 6 != 0 {write!(input, "\n");}
        write!(input, "Number of contracted shells                I {:16}\n",num_contract);
        write!(input, "Number of primitive shells                 I {:16}\n",num_primitiv);
        let basis_type = match self.mol.cint_type {
            CintType::Spheric => {
                0
            },
            CintType::Cartesian => {
                1
            },
            CintType::Spinor => {
                panic!("Spinor basis is not yet implemented.")
            }
        };
        write!(input, "Pure/Cartesian d shells                    I {:16}\n", basis_type);
        write!(input, "Pure/Cartesian f shells                    I {:16}\n", basis_type);
        write!(input, "Highest angular momentum                   I {:16}\n", max_ang);
        write!(input, "Largest degree of contraction              I {:16}\n", max_contract);
        // ==============================
        // Now for shell infor.
        // ==============================
        write!(input, "Shell types                                I   N={:12}\n", num_contract);
        dump_int!(input, shell_type);
        write!(input, "Number of primitives per shell             I   N={:12}\n", num_contract);
        dump_int!(input, num_primitiv_vec);
        write!(input, "Shell to atom map                          I   N={:12}\n", num_contract);
        dump_int!(input, shell_to_atom_map);
        write!(input, "Primitive exponents                        R   N={:12}\n", primitive_exp.len());
        dump_real_r2f!(input, primitive_exp);
        write!(input, "Contraction coefficients                   R   N={:12}\n", coeff_vec.len());
        dump_real_r2f!(input, coeff_vec);
//        write!(input, "P(S=P) Contraction coefficients            R   N={:12}\n", coeff_vec.len());
//        let mut i_index = 0;
//        coeff_vec.iter().for_each(|x| {
//            let sdd = format!("{:16.8E}", 0);
//            if (i_index + 1)%5 == 0 {
//                write!(input, "{}\n",r2f(&sdd))
//            } else {
//                write!(input, "{}",r2f(&sdd))
//            };
//            i_index += 1;
//        });
//        if i_index % 5 != 0 {write!(input, "\n");}
        write!(input, "Coordinates of each shell                  R   N={:12}\n", coord_each_shell.len());
        dump_real_r2f!(input, coord_each_shell);
        // ==============================
        // Now for total and orb energies
        // ==============================
        let mut ilsw:Vec<i32> = vec![0;100];
        if self.mol.spin_channel==2 && !matches!(self.scftype, SCFType::ROHF) {
            ilsw[0] = 1
        }
        ilsw[4] = 2;
        ilsw[11] = -1;
        ilsw[12] = 5;
        ilsw[24] = 1;
        ilsw[25] = 1;
        ilsw[32] = 100000;
        ilsw[34] = -1;
        ilsw[45] = 1;
        ilsw[49] = -2000000000;
        ilsw[50] = 1;
        ilsw[56] = 4;
        ilsw[57] = 52;
        ilsw[69] = natom as i32;
        ilsw[71] = 1;  
        write!(input, "Num ILSW                                   I     {:12}\n", 100);
        write!(input, "ILSW                                       I   N={:12}\n", 100);
        dump_int!(input, ilsw);
        // TODO: ECP
        write!(input, "SCF Energy                                 R ").unwrap();
        write_real_value(&mut input, self.scf_energy).unwrap();
        write!(input, "\n").unwrap();
        write!(input, "Total Energy                               R ").unwrap();
        write_real_value(&mut input, self.scf_energy).unwrap();
        write!(input, "\n").unwrap();
        for i_spin in 0..self.mol.spin_channel {
            if i_spin ==0  {
                write!(input, "Alpha Orbital Energies                     R   N={:12}\n", self.eigenvalues[i_spin].len());
            } else {
                write!(input, "Beta Orbital Energies                      R   N={:12}\n", self.eigenvalues[i_spin].len());
            }
            dump_real_r2f!(input, self.eigenvalues[i_spin]);
            if let SCFType::ROHF = self.scftype { break } // ROHF: alpha == beta
        }
        // ==============================
        // Now for orbital coefficients
        // ==============================
        let nbf = self.mol.num_basis;
        //let nif = self.mol.num_state;
        for i_spin in 0..self.mol.spin_channel {
            if i_spin ==0  {
                write!(input, "Alpha MO coefficients                      R   N={:12}\n", self.eigenvectors[i_spin].data.len());
            } else {
                write!(input, "Beta MO coefficients                       R   N={:12}\n", self.eigenvectors[i_spin].data.len());
            }
            if let SCFType::ROHF = self.scftype { break } // ROHF: alpha == beta
        }
        // ==============================
        // Now for orbital occupations
        // ==============================
        if let SCFType::ROHF = self.scftype {
            // ROHF: merge alpha + beta into single occupation array
            let occ_merged: Vec<f64> = self.occupation[0].iter()
                .zip(self.occupation[1].iter())
                .map(|(a, b)| a + b)
                .collect();
            write!(input, "Alpha Orbital Occupation                   R   N={:12}\n", occ_merged.len());
            dump_real_r2f!(input, occ_merged);
        } else {
            for i_spin in 0..self.mol.spin_channel {
                if i_spin == 0 {
                    write!(input, "Alpha Orbital Occupation                   R   N={:12}\n", self.occupation[i_spin].len());
                } else {
                    write!(input, "Beta Orbital Occupation                    R   N={:12}\n", self.occupation[i_spin].len());
                }
                dump_real_r2f!(input, self.occupation[i_spin]);
            }
        }
        // leave MOs blank; they are filled afterwards either by librest2fch
        // (feature `librest2fch`) or natively by `fchk_write_mo_rust` (default)
        // Total SCF density: the lower-triangular, column-packed AO density, permuted to
        // Gaussian's AO order so it agrees with the MO coefficients written by librest2fch. For
        // open-shell UHF runs the alpha/beta split is written in Gaussian's convention as
        // `Total SCF Density` (alpha + beta) plus `Spin SCF Density` (alpha - beta); a zero
        // placeholder is kept for ROHF (its fractional occupations have no alpha/beta split).
        // librest2fch only requires the section header to exist and copies the section content
        // verbatim when regenerating the MO coefficients.
        // Controlled by `[output] fchk_dm` (default: "auto"); "scf" and
        // "false" both omit the MP2 densities, "scf" keeps the SCF density.
        if self.mol.ctrl.output.fchk_dm.as_str() != "false" {
            let n_dm = nbf*(nbf+1)/2;
            let packed_dm = match self.scftype {
                SCFType::UHF => {
                    let dm_total = self.density_matrix[0].clone() + self.density_matrix[1].clone();
                    self.fchk_pack_density_ao(&dm_total)
                },
                _ if self.mol.spin_channel == 1 => self.fchk_pack_density_ao(&self.density_matrix[0]),
                _ => vec![0.0f64; n_dm],
            };
            write!(input, "Total SCF Density                          R   N={:12}\n", n_dm);
            dump_real_r2f!(input, packed_dm);
            if let SCFType::UHF = self.scftype {
                let dm_spin = self.density_matrix[0].clone() - self.density_matrix[1].clone();
                let packed_spin = self.fchk_pack_density_ao(&dm_spin);
                write!(input, "Spin SCF Density                           R   N={:12}\n", n_dm);
                dump_real_r2f!(input, packed_spin);
            }
        }
        input.flush().unwrap();
        input.into_inner().unwrap().sync_all().unwrap();
    }

    /// Write the MO coefficients of the fchk file. The writer is selected by
    /// the `[output] fchk_writer` keyword: "default" resolves to the
    /// MOKIT-derived Fortran library (`librest2fch`) when the feature of the
    /// same name is compiled in, and to the native Rust writer otherwise;
    /// "librest2fch" and "rust" force one of the two explicitly. The density
    /// sections are always generated by REST itself (see `create_fchk_head`);
    /// in the librest2fch path the library copies them verbatim.
    pub fn fchk_write_mo(&self) {
        let writer = self.mol.ctrl.output.fchk_writer.as_str();
        #[cfg(feature = "librest2fch")]
        {
            let use_lib = match writer {
                "default" | "librest2fch" => true,
                "rust" => false,
                other => panic!("Unknown fchk_writer '{}'. Available: default, librest2fch, rust.", other),
            };
            if use_lib {
                let nbf = self.mol.num_basis;
                let nif = self.mol.num_state;

                for i_spin in 0..self.mol.spin_channel {
                    if i_spin ==0  {
                        py2fch(format!("{}.fchk", self.mol.geom.name), nbf, nif, &self.eigenvectors[i_spin].data, 'a', &self.eigenvalues[i_spin], 0, 0);
                    } else {
                        py2fch(format!("{}.fchk", self.mol.geom.name), nbf, nif, &self.eigenvectors[i_spin].data, 'b', &self.eigenvalues[i_spin], 0, 0);
                    }
                    if let SCFType::ROHF = self.scftype { break } // ROHF: alpha == beta
                }
                return;
            }
        }
        #[cfg(not(feature = "librest2fch"))]
        {
            match writer {
                "librest2fch" => panic!("fchk_writer = \"librest2fch\" requires a REST binary built with `--features librest2fch`."),
                "default" | "rust" => {},
                other => panic!("Unknown fchk_writer '{}'. Available: default, rust (librest2fch is not compiled in).", other),
            }
        }
        self.fchk_write_mo_rust();
    }

    /// Fill the blank `Alpha/Beta MO coefficients` sections of the fchk file in
    /// pure Rust (no librest2fch). The coefficients are permuted from REST's
    /// (libcint) AO order to Gaussian's order with the same table the library
    /// applies ([`SCF::gaussian_ao_permutation`]; the `norm` factors of
    /// `py2fch` are 1.0 for spherical shells), then inserted after the section
    /// headers that `create_fchk_head` left blank, in the same column-major
    /// layout (AO index fastest) and the same `ES15.8`-style 5-per-line format.
    fn fchk_write_mo_rust(&self) {
        let perm = self.gaussian_ao_permutation();
        let nbf = self.mol.num_basis;
        let fname = format!("{}.fchk", self.mol.geom.name);
        let tmpname = format!("{}.t", fname);
        let content = fs::read_to_string(&fname).unwrap();
        // The rewrite is line-by-line, so the sink has to be buffered: an
        // unbuffered `File` costs one `write(2)` per output line, and a job with
        // a few thousand basis functions rewrites a file of several hundred MB.
        let file = fs::File::create(&tmpname).unwrap();
        let mut output = BufWriter::with_capacity(FCHK_BUFFER_SIZE, file);
        for line in content.lines() {
            writeln!(output, "{}", line).unwrap();
            let i_spin = if line.starts_with("Alpha MO coefficients") {
                Some(0)
            } else if line.starts_with("Beta MO coefficients") {
                Some(1)
            } else {
                None
            };
            if let Some(i_spin) = i_spin {
                let nif = self.mol.num_state;
                let mut mo: Vec<f64> = Vec::with_capacity(nbf*nif);
                for i_mo in 0..nif {
                    perm.iter().for_each(|&rest_pos| {
                        mo.push(self.eigenvectors[i_spin].data[rest_pos + i_mo*nbf]);
                    });
                }
                dump_real_r2f!(output, mo);
            }
        }
        output.flush().unwrap();
        output.into_inner().unwrap().sync_all().unwrap();
        fs::rename(&tmpname, &fname).unwrap();
    }

    /// AO-position permutation from REST's (libcint) AO order to Gaussian's internal order,
    /// as `perm[gaussian_position] = rest_position`, consistent with the permutation that
    /// librest2fch (`py2fch`) applies to the MO coefficients: spherical shells with `l >= 2`
    /// are reordered from libcint's m = -l ... +l to Gaussian's m = 0, +1, -1, +2, -2, ...
    /// (the MOKIT `py2fch_permute_5d/7f/9g/11h` tables); s and p shells carry identical
    /// orders in both conventions, so they are copied through.
    ///
    /// Any density matrix dumped to the fchk file (e.g. the `Total MP2 Density` section)
    /// must be permuted with this table to agree with the MO coefficients of the same file.
    pub fn gaussian_ao_permutation(&self) -> Vec<usize> {
        if !matches!(self.mol.cint_type, CintType::Spheric) {
            panic!("The Gaussian AO permutation is currently implemented for spherical shells only.")
        }
        let mut perm: Vec<usize> = Vec::new();
        let mut base = 0usize;
        self.mol.basis4elem.iter().for_each(|ibas| {
            ibas.electron_shells.iter().for_each(|ibascell| {
                let l = ibascell.angular_momentum[0];
                let n_func = (2 * l + 1) as usize;
                let n_cont = ibascell.coefficients.len();
                // Gaussian m sequence 0, +1, -1, +2, -2, ...; libcint's m-ascending
                // position of m is (m + l) within the shell
                let mut ms: Vec<i32> = vec![0];
                if l > 1 {
                    let mut k = 1i32;
                    while ms.len() < n_func {
                        ms.push(k);
                        if ms.len() < n_func {
                            ms.push(-k);
                        }
                        k += 1;
                    }
                }
                // libcint lays out a contracted shell contraction-major, each contraction
                // contributing one block of (2l+1) AOs (see Molecule::build_cint/build_fdqc)
                for _ic in 0..n_cont {
                    if l <= 1 {
                        (0..n_func).for_each(|i| perm.push(base + i));
                    } else {
                        ms.iter().for_each(|&m| perm.push(base + (m + l as i32) as usize));
                    }
                    base += n_func;
                }
            })
        });
        perm
    }

    /// Append a symmetric density section to the Gaussian fchk file: `label` followed by the
    /// lower-triangular (column-packed) density in Gaussian's AO order — the same layout as
    /// the `Total SCF Density` section. The file must already exist (i.e.
    /// [`SCF::save_fchk_of_gaussian`] must have run, so the appended section comes after the
    /// librest2fch-written MO coefficients, which are untouched). `label` should follow
    /// Gaussian's section naming ("Total MP2 Density", the name Gaussian itself writes for
    /// the relaxed MP2 density with `Density=MP2`), so that downstream readers (Gaussian
    /// tooling, Multiwfn, MOKIT, ...) recognize the section.
    pub fn fchk_append_density_section(&self, label: &str, packed_lower: &[f64]) {
        println!("\nAppend density section \"{label}\" to fchk file");
        let file = fs::OpenOptions::new()
            .append(true)
            .open(format!("{}.fchk", &self.mol.geom.name))
            .unwrap();
        let mut input = BufWriter::with_capacity(FCHK_BUFFER_SIZE, file);
        write!(input, "{label:<43}R   N={:12}\n", packed_lower.len());
        dump_real_r2f!(input, packed_lower);
        input.flush().unwrap();
        input.into_inner().unwrap().sync_all().unwrap();
    }

    /// Permute an AO-basis density matrix into Gaussian's AO order (consistent with the MO
    /// coefficients of the same fchk file) and pack its lower triangle by column — the layout
    /// of the fchk density sections (same as `Total SCF Density`).
    fn fchk_pack_density_ao(&self, dm_ao: &MatrixFull<f64>) -> Vec<f64> {
        let nbf = dm_ao.size[0];
        assert_eq!(dm_ao.size[1], nbf, "The packed AO density must be square.");
        let perm = self.gaussian_ao_permutation();
        assert_eq!(perm.len(), nbf, "AO permutation size mismatch.");
        let mut packed = Vec::with_capacity(nbf * (nbf + 1) / 2);
        for pj in 0..nbf {
            let j = perm[pj];
            for pi in 0..=pj {
                packed.push(dm_ao.data[perm[pi] + j * nbf]);
            }
        }
        packed
    }

    /// Append a symmetric AO-basis density as the Gaussian fchk density section `label`
    /// (e.g. `Total MP2 Density`): the density is permuted into Gaussian's AO order, packed
    /// lower-triangular by column and appended through [`SCF::fchk_append_density_section`]
    /// (which documents the file/label requirements).
    pub fn fchk_append_density_ao(&self, label: &str, dm_ao: &MatrixFull<f64>) {
        let packed = self.fchk_pack_density_ao(dm_ao);
        self.fchk_append_density_section(label, &packed);
    }

}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utilities::convert_scientific_notation_to_fortran_format as legacy_r2f;

    /// The historical per-element writer, kept verbatim as the reference the
    /// buffered helpers have to reproduce.
    fn legacy_dump(data: &[f64]) -> String {
        let mut out = String::new();
        let mut i_index = 0;
        data.iter().for_each(|x| {
            let sdd = format!("{:16.8E}", x);
            if (i_index + 1) % 5 == 0 {
                out.push_str(&format!("{}\n", legacy_r2f(&sdd)))
            } else {
                out.push_str(&legacy_r2f(&sdd))
            };
            i_index += 1;
        });
        if i_index % 5 != 0 {
            out.push('\n');
        }
        out
    }

    /// Values the legacy pipeline could actually format: finite and with an
    /// exponent of at most two digits (three digits used to underflow the
    /// `out_str[n_len - o_len..]` slice and abort the job).
    fn legacy_compatible(x: f64) -> Option<String> {
        if !x.is_finite() {
            return None;
        }
        let rendered = format!("{:16.8E}", x);
        let epos = rendered.find('E')?;
        let exponent: i32 = rendered[epos + 1..].parse().ok()?;
        if exponent.abs() > 99 {
            return None;
        }
        Some(rendered)
    }

    fn sample_values() -> Vec<f64> {
        let mut samples: Vec<f64> = vec![
            0.0, -0.0, 1.0, -1.0, 0.5, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7,
            1e-8, 1e-9, 1e-10, 1e-15, 1e-30, 1e1, 1e2, 1e10, 1e20, 1e99,
            1.23456789e-5, -1.23456789e-5, 9.99999999e-1, 3.0, 1234.5678,
            -6660.1565868236, -6848.4926793607, 2.862028e6,
        ];
        // Dense sweep over every two-digit exponent, both signs.
        let mantissa = 1.23456789_f64;
        for e in -99..=99 {
            samples.push(mantissa * 10f64.powi(e));
            samples.push(-mantissa * 10f64.powi(e));
            samples.push(9.999999999_f64 * 10f64.powi(e));
        }
        // Pseudo-random bit patterns (xorshift), keeping the finite ones.
        let mut state = 0x2545_F491_4F6C_DD1D_u64;
        for _ in 0..50_000 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let x = f64::from_bits(state);
            if x.is_finite() {
                samples.push(x);
            }
        }
        samples
    }

    /// Byte-for-byte equivalence of the whole numeric layout: field content,
    /// five columns per line, and the trailing-newline rule.
    #[test]
    fn fchk_real_layout_matches_legacy_writer() {
        let samples = sample_values();
        let compatible: Vec<f64> = samples
            .iter()
            .copied()
            .filter(|x| legacy_compatible(*x).is_some())
            .collect();
        assert!(compatible.len() > 1000, "the sample set should be large");

        let expected = legacy_dump(&compatible);
        let mut got: Vec<u8> = Vec::new();
        write_real_fields(&mut got, &compatible).unwrap();
        assert_eq!(String::from_utf8(got).unwrap(), expected);

        // Partial final line, and a single value.
        for len in [1usize, 2, 3, 4, 5, 6, 7, 11] {
            let slice = &compatible[..len];
            let expected = legacy_dump(slice);
            let mut got: Vec<u8> = Vec::new();
            write_real_fields(&mut got, slice).unwrap();
            assert_eq!(String::from_utf8(got).unwrap(), expected, "mismatch at len {len}");
        }
    }

    /// The 27-column energy field follows the same rule.
    #[test]
    fn fchk_energy_field_matches_legacy_pipeline() {
        for &energy in &[
            -6660.1565868236_f64,
            -6848.4926793607,
            0.0,
            1.5,
            -0.0003356,
            1.563e-16,
            1.0e-99,
        ] {
            let rendered = format!("{:27.15E}", energy);
            let expected = legacy_r2f(&rendered);
            let mut got = [b' '; 32];
            fchk_format_real(energy, FCHK_ENERGY_DECIMALS, &mut got[..FCHK_ENERGY_WIDTH]);
            assert_eq!(
                std::str::from_utf8(&got[..FCHK_ENERGY_WIDTH]).unwrap(),
                expected,
                "mismatch for {energy:e}"
            );
        }
        // Four-digit exponents made the legacy slice underflow; the field is
        // still the full width, right-aligned.
        let mut got = [b' '; 32];
        fchk_format_real(-1.0e-100, FCHK_ENERGY_DECIMALS, &mut got[..FCHK_ENERGY_WIDTH]);
        assert_eq!(
            std::str::from_utf8(&got[..FCHK_ENERGY_WIDTH]).unwrap(),
            "    -1.000000000000000E-100"
        );
    }

    /// Three-digit exponents used to abort the job; they must now be written in
    /// full, still inside the fixed-width column.
    #[test]
    fn fchk_real_writes_three_digit_exponents() {
        for (value, expected) in [
            (1e-100_f64, " 1.00000000E-100"),
            (-1e-100_f64, "-1.00000000E-100"),
            (1e100_f64, " 1.00000000E+100"),
            (5e-324_f64, " 4.94065646E-324"),
        ] {
            let mut got = [b' '; FCHK_REAL_WIDTH];
            fchk_format_real(value, FCHK_REAL_DECIMALS, &mut got);
            assert_eq!(
                std::str::from_utf8(&got).unwrap(),
                expected,
                "mismatch for {value:e}"
            );
            assert_eq!(std::str::from_utf8(&got).unwrap().len(), FCHK_REAL_WIDTH);
        }
    }

    /// Opt-in micro-benchmark of the numeric writer. It lives with the code it
    /// measures (rather than in `tests/`) because the helpers are module
    /// private, and it doubles as a large-scale byte-identity check:
    ///
    /// `REST_BENCH_FCHK=1 cargo test --release -p rest --lib bench_fchk -- --nocapture`
    ///
    /// The scale is the `Total SCF Density` section of a ~2400-basis-function
    /// open-shell job (2392*2393/2 values). Three pipelines are timed so the two
    /// independent costs can be told apart:
    ///
    /// * `original` — the historical writer: `format!`, a `Regex::new` compiled
    ///   inside the conversion function, and one `write!` per value into an
    ///   unbuffered `File`.
    /// * `hoisted`  — the same writer with only the regex hoisted out of the
    ///   function (isolates the regex-compilation cost).
    /// * `current`  — stack formatting plus one `write_all` per line behind a
    ///   `BufWriter`.
    ///
    /// Run it under `strace -c -e trace=write` to also see the syscall count:
    /// the `write(2)` count is what makes the original scale with the number of
    /// values rather than with the file size, and is the part most affected by a
    /// networked parallel filesystem.
    #[test]
    fn bench_fchk_real_writer() {
        use std::time::Instant;
        if std::env::var("REST_BENCH_FCHK").is_err() {
            return;
        }
        // The original conversion, verbatim: the regex is rebuilt on every call.
        fn original_r2f(n: &String) -> String {
            let re = Regex::new(r"(?P<num> *-?\d.\d*)[E|e](?P<exp>-?\d{1,2})").unwrap();
            let o_len = n.len();
            if let Some(cap) = re.captures(n) {
                let main_part = cap["num"].to_string();
                let exp: i32 = cap["exp"].to_string().parse().unwrap();
                let out_str = if exp >= 0 {
                    format!("{}E+{:0>2}", main_part, exp)
                } else {
                    format!("{}E-{:0>2}", main_part, exp.abs())
                };
                let n_len = out_str.len();
                return out_str[n_len - o_len..n_len].to_string();
            }
            panic!("Error: the input string is not a standard scientific notation")
        }

        /// Write `data` with the per-element macro loop, converting each value
        /// with `convert`, into an unbuffered file.
        fn legacy_write(path: &std::path::Path, data: &[f64], convert: impl Fn(&String) -> String) -> Vec<u8> {
            let mut out = fs::File::create(path).unwrap();
            let mut i_index = 0;
            for x in data {
                let sdd = format!("{:16.8E}", x);
                if (i_index + 1) % 5 == 0 {
                    write!(out, "{}\n", convert(&sdd)).unwrap();
                } else {
                    write!(out, "{}", convert(&sdd)).unwrap();
                }
                i_index += 1;
            }
            if i_index % 5 != 0 {
                write!(out, "\n").unwrap();
            }
            out.sync_all().unwrap();
            fs::read(path).unwrap()
        }

        const N: usize = 2_862_028;
        let data: Vec<f64> = (0..N)
            .map(|i| ((i as f64) * 0.618_033_988_749_895).sin() * 1e-2)
            .collect();
        let original_path = std::env::temp_dir().join("rest_bench_fchk_original.fchk");
        let hoisted_path = std::env::temp_dir().join("rest_bench_fchk_hoisted.fchk");
        let current_path = std::env::temp_dir().join("rest_bench_fchk_current.fchk");

        let t0 = Instant::now();
        let original_bytes = legacy_write(&original_path, &data, original_r2f);
        let original = t0.elapsed();

        let t0 = Instant::now();
        let hoisted_bytes = legacy_write(&hoisted_path, &data, legacy_r2f);
        let hoisted = t0.elapsed();

        let t0 = Instant::now();
        {
            let file = fs::File::create(&current_path).unwrap();
            let mut out = BufWriter::with_capacity(FCHK_BUFFER_SIZE, file);
            write_real_fields(&mut out, &data).unwrap();
            out.flush().unwrap();
            out.into_inner().unwrap().sync_all().unwrap();
        }
        let current = t0.elapsed();
        let current_bytes = fs::read(&current_path).unwrap();

        assert!(
            original_bytes == current_bytes && hoisted_bytes == current_bytes,
            "all three writers must stay byte-identical ({} / {} / {} bytes)",
            original_bytes.len(),
            hoisted_bytes.len(),
            current_bytes.len()
        );

        println!("values                                    : {N}");
        println!("bytes                                     : {}", current_bytes.len());
        println!("original format! + regex/call + unbuffered : {original:?}");
        println!("hoisted  format! + shared regex + unbuffered: {hoisted:?}");
        println!("current  stack fmt + BufWriter per line    : {current:?}");
        println!("speed-up (original -> current)             : {:.1}x", original.as_secs_f64() / current.as_secs_f64());
        println!("speed-up (hoisted  -> current)             : {:.1}x", hoisted.as_secs_f64() / current.as_secs_f64());
        for p in [original_path, hoisted_path, current_path] {
            fs::remove_file(p).unwrap();
        }
    }
}
