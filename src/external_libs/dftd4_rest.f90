subroutine calc_dftd4_rest(num, num_size, xyz, charge, uhf, method_c, method_len, energy, final_gradient, final_sigma)
    use mctc_env
    use mctc_io
    use dftd4
    implicit none
 
    integer, intent(in) :: num_size
    integer, intent(in) :: num(num_size)
    real(wp), intent(in) :: xyz(3, num_size)
    real(wp), intent(in) :: charge
    integer, intent(in) :: uhf
    character(len=100), intent(in) :: method_c
    integer, intent(in) :: method_len
    character(len=100) :: method
 
    type(structure_type) :: mol
    type(d4_model) :: disp
    class(damping_param), allocatable :: param
 
    real(wp), allocatable :: xyz_m(:, :)
 
    real(wp), intent(out) :: energy
    real(wp), intent(out) :: final_gradient(3, num_size)
    real(wp), intent(out) :: final_sigma(3, num_size)
 
    !strip the method and corr
    method = method_c(1:method_len)
 
    !turn the float array into wp array
    xyz_m = reshape(xyz, [3, num_size])
 
    call new(mol, num, xyz_m, charge, uhf)
 
    call new_d4_model(disp, mol)
 
    call get_rational_damping(method, param, s9=1.0_wp)
 
    call get_dispersion(mol, disp, param, realspace_cutoff(), energy, final_gradient, final_sigma)
 
end subroutine calc_dftd4_rest


