subroutine calc_dftd3_atm_rest(num, num_size, xyz, charge, uhf, method_c, method_len, &
                             & energy, final_gradient, final_sigma, corr_c, corr_len)
    use mctc_env
    use mctc_io
    use dftd3
    implicit none
 
    integer, intent(in) :: num_size
    integer, intent(in) :: num(num_size)
    real(wp), intent(in) :: xyz(3 * num_size)
    real(wp), intent(in) :: charge
    integer, intent(in) :: uhf
    character(len=100) :: method_c, corr_c
    integer, intent(in) :: method_len, corr_len
    character(len=100) :: method, corr
 
    type(error_type), allocatable :: error
    type(d3_model) :: disp
    type(d3_param) :: inp
    class(rational_damping_param), allocatable :: rational_param
    class(zero_damping_param), allocatable :: zero_param
    type(structure_type) :: mol
    real(wp), allocatable :: gradient(:,:)
    real(wp), allocatable :: sigma(:,:)
 
    real(wp), allocatable :: xyz_m(:, :)
 
    real(wp), intent(out) :: energy
    real(wp), intent(out) :: final_gradient(3, num_size)
    real(wp), intent(out) :: final_sigma(3, num_size)
 
    !strip the method and corr
    method = method_c(1:method_len)
    corr = corr_c(1:corr_len)
 
    !turn the float array into wp array
    xyz_m = reshape(xyz, [3, num_size])
 
    call new(mol, num, xyz_m, charge, uhf)
 
    call new_d3_model(disp, mol)
 
    allocate(gradient(3, num_size), sigma(3, num_size))
 
    !change model by input
    select case (trim(corr))
    case ('d3bj')
        call get_rational_damping(inp, method, error, s9=1.0_wp)
        if (allocated(error)) return
        allocate(rational_param)
        call new_rational_damping(rational_param, inp)
        call get_dispersion(mol, disp, rational_param, realspace_cutoff(), energy, gradient, sigma)
    case ('d3')
        call get_zero_damping(inp, method, error, s9=1.0_wp)
        if (allocated(error)) return
        allocate(zero_param)
        call new_zero_damping(zero_param, inp)
        call get_dispersion(mol, disp, zero_param, realspace_cutoff(), energy, gradient, sigma)
    end select
 
    if (allocated(gradient)) then
        final_gradient = gradient
    endif
 
    if (allocated(sigma)) then
        final_sigma = sigma
    endif

end subroutine calc_dftd3_atm_rest
