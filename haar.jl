using SparseArrays
using LinearAlgebra
using Arpack
using Statistics
using Random
using DelimitedFiles
using NPZ
using ExpmV
using Dates

Random.seed!(Dates.now().instant.periods.value)

function Hamiltonian(L)
    """
    Jx_list = ones(Float64, L)
    Jy_list = ones(Float64, L)
    Jz_list = ones(Float64, L)

    
    # set the last element to what you want
    Jx_list[end] = 1e-5
    Jy_list[end] = 1e-5
    Jz_list[end] = 1e-5
    """

    #delta = 0.5

    # Define Pauli matrices as complex sparse matrices
    id = sparse(ComplexF64[1 0; 0 1])
    sx = sparse(ComplexF64[0 1; 1 0])
    sy = sparse(ComplexF64[0 -im; im 0])
    sz = sparse(ComplexF64[1 0; 0 -1])


    # Preallocate vectors of operators with correct type
    sx_list = Vector{SparseMatrixCSC{ComplexF64, Int}}(undef, L)
    sy_list = Vector{SparseMatrixCSC{ComplexF64, Int}}(undef, L)
    sz_list = Vector{SparseMatrixCSC{ComplexF64, Int}}(undef, L)

    for i_site in 1:L
        x_ops = fill(id, L)
        y_ops = fill(id, L)
        z_ops = fill(id, L)
        x_ops[i_site] = sx
        y_ops[i_site] = sy
        z_ops[i_site] = sz

        # Build the full operator by tensoring
        X = x_ops[1]
        Y = y_ops[1]
        Z = z_ops[1]
        for j in 2:L
            X = kron(X, x_ops[j])
            Y = kron(Y, y_ops[j])
            Z = kron(Z, z_ops[j])
        end

        sx_list[i_site] = X
        sy_list[i_site] = Y
        sz_list[i_site] = Z
    end

    """
    dim = 2^L
    H = spzeros(ComplexF64, dim, dim)

    for i in 1:L-1
        ip1 = (i + 1)  # Open boundary
        H -= Jx_list[i] * (sx_list[i] * sx_list[ip1])
        H -= Jy_list[i] * (sy_list[i] * sy_list[ip1])
        H -= (Jz_list[i]) * (sz_list[i] * sz_list[ip1])
    end
    """

    return sx_list, sz_list
end

"""
function ck(Nc, a, t)
    k = 0:Nc-1
    return (-im) .^ k .* besselj.(k, a * t)
end

function Time_evo(psi, Nc, Ht, a, b, t)
    c = ck(Nc, a, t)
    psi_km0 = Ht * psi
    psi_km1 = psi
    psi_t = c[1] * psi_km1 + 2 * c[2] * psi_km0  # Julia is 1-indexed

    for k in 3:Nc
        psi_k = 2 * Ht * psi_km0 - psi_km1
        psi_t += 2 * c[k] * psi_k
        psi_km1 = psi_km0
        psi_km0 = psi_k
    end

    return exp(-im * b * t) * psi_t
end

function H_twi(L)
    H, sx_list, sz_list = Hamiltonian(L)
    GS, _ = eigs(H; nev=1, which=:SR, ritzvec=false)  # smallest eigenvalue
    SS, _ = eigs(H; nev=1, which=:LR, ritzvec=false)  # largest eigenvalue
    a = (SS[1] - GS[1]) / 2
    b = (SS[1] + GS[1]) / 2
    dim = 2^L
    Ht = (H - b * sparse(I, dim, dim)) / a
    return Ht, a, b, sx_list, sz_list
end
"""

function calculate_entropy(psi::Vector{ComplexF64}, L::Int, subsystem_sites::Vector{Int})
    psi = psi / norm(psi)

    # Reshape the state vector into a tensor of shape (2, 2, ..., 2) with L indices
    psi_tensor = reshape(psi, ntuple(i -> 2, L))

    # Sort the subsystem sites and determine environment sites
    subsystem_sites = sort(subsystem_sites)
    all_sites = collect(1:L)
    environment_sites = setdiff(all_sites, subsystem_sites)

    # Permute the tensor to bring subsystem indices to the front
    permute_order = vcat(subsystem_sites, environment_sites)
    permuted_tensor = permutedims(psi_tensor, permute_order)

    # Reshape into a matrix suitable for SVD
    dim_subsystem = 2^length(subsystem_sites)
    dim_environment = 2^(L - length(subsystem_sites))
    psi_matrix = reshape(permuted_tensor, dim_subsystem, dim_environment)

    # Perform SVD
    s = svdvals(psi_matrix)

    # Compute the entropy
    s_squared = s .^ 2
    s_squared ./= sum(s_squared)  # Normalize
    s_squared_nonzero = s_squared[s_squared .> 1e-15]
    entropy = -sum(s_squared_nonzero .* log.(s_squared_nonzero))

    return entropy
end

function I3(psi::Vector{ComplexF64})
    L = Int(round(log2(length(psi))))

    A = collect(1:(L ÷ 4))
    B = collect((L ÷ 4) + 1 : (L ÷ 2))
    C = collect((L ÷ 2) + 1 : (3L ÷ 4))
    D = collect((3L ÷ 4) + 1 : L)

    AB = vcat(A, B)
    BC = vcat(B, C)
    AC = vcat(A, C)

    SA = calculate_entropy(psi, L, A)
    SB = calculate_entropy(psi, L, B)
    SC = calculate_entropy(psi, L, C)
    SABC = calculate_entropy(psi, L, D)
    SAB = calculate_entropy(psi, L, AB)
    SAC = calculate_entropy(psi, L, AC)
    SBC = calculate_entropy(psi, L, BC)

    return SA + SB + SC + SABC - SAB - SAC - SBC
end

function random_product_state(L::Int)
    product_state = ComplexF64[1.0]  # Start with scalar 1 as base state

    for _ in 1:L
        θ = rand() * π        # Polar angle
        ϕ = rand() * 2π       # Azimuthal angle

        α = cos(θ / 2)                # Coefficient for |0⟩
        β = sin(θ / 2) * cis(ϕ)       # Coefficient for |1⟩, using cis(ϕ) = exp(iϕ)

        temp_state = ComplexF64[α, β]

        product_state = kron(product_state, temp_state)
    end

    return product_state / norm(product_state)
end

function Entropy_t(L::Int, T::Float64, dt::Float64, p::Float64, direction::String, shot::Int)
    Random.seed!(shot)  # Set random seed
    s_t = random_product_state(L)
    #s_t = neel_spinhalf(L) ## Neel state
    S_list = Float64[]

    # Define Hamiltonian and local observables
    #Ht, a, b, sx_list, sz_list = H_twi(L)
    
    sx_list, sz_list = Hamiltonian(L)

    if direction == "X"
    sm_list = sx_list
    elseif direction == "Z"
        sm_list = sz_list
    else
        error("Invalid direction: $direction. Choose \"X\" or \"Z\".")
    end


    steps = Int(floor(T / dt))

    for _ in 1:steps
        #push!(S_list, calculate_entropy(s_t, L, collect(1:L-1)))
        push!(S_list, calculate_entropy(s_t, L, collect(1:L÷2)))
        #push!(S_list, I3(s_t))

        # Time evolution
        s_t = time_evolution(s_t, L)
        #s_t = Time_evo(s_t, Nc, Ht, a, b, dt)
        s_t ./= sqrt(real(s_t' * s_t))  # Normalize

        # Measurements
        if p != 0
            for l in 1:L
                x = rand()
                if x < p
                    m_op = sm_list[l]
                    p_m = 0.5 + 0.5 * real(s_t' * (m_op * s_t))
                    if rand() < p_m
                        s_t = (s_t + m_op * s_t) / (2 * sqrt(p_m))
                    else
                        s_t = (s_t - m_op * s_t) / (2 * sqrt(1 - p_m))
                    end
                end
            end
        end
    end

    # Save result to disk
    folder = "/Users/uditvarma/Documents/haar_data/data_hc"
    mkpath(folder)
    filename_i3 = joinpath(folder, "L$(L),T$(T),dt$(dt),p$(p),dir$(direction),s$(shot)_hc.npy")
    npzwrite(filename_i3, S_list)

    return S_list
end

function time_evolution(ψ::Vector{ComplexF64}, L)

    U_odd  = odd_layer(L)
    U_even = even_layer(L)

    # One full brick-wall step:
    U_step = U_even * U_odd
    ψ_new = U_step * ψ

    return normalize!(ψ_new)
end

# Define spin-1/2 basis vectors
function spinhalf_vector(state::String)
    if state == "Up"
        return [1.0, 0.0]
    elseif state == "Dn"
        return [0.0, 1.0]
    else
        error("State must be \"Up\" or \"Dn\"")
    end
end

function neel_spinhalf(N::Int)
    # Construct the Néel pattern: Up, Dn, Up, Dn...
    neel_state = [isodd(j) ? "Up" : "Dn" for j in 1:N]
    
    # Start with first spin
    psi = spinhalf_vector(neel_state[1])
    
    # Build the full tensor product
    for j in 2:N
        psi = kron(psi, spinhalf_vector(neel_state[j]))
    end
    
    # Convert to complex vector
    psi_complex = ComplexF64.(psi)
    
    return psi_complex
end


function odd_layer(L::Int)
    id = sparse(ComplexF64[1 0; 0 1])
    U = haar_unitary()
    for i in 3:2:L-1
        U = kron(U, haar_unitary())
    end
    if isodd(L)
        U = kron(U, id)   # pad last site if odd number of spins
    end
    return U
end

function even_layer(L::Int)
    id = sparse(ComplexF64[1 0; 0 1])
    U = id                          # start with a free spin at site 1
    for _ in 2:2:L-1
        U = kron(U, haar_unitary())
    end
    if isodd(L)
        # if L is even, we’ve already used up all spins
        return U
    else
        # if L is odd, pad last site
        return kron(U, id)
    end
end

function haar_unitary()
    # Complex Ginibre ensemble (Gaussian random matrix)
    Z = randn(ComplexF64, 4, 4) ./ sqrt(2)
    
    # QR decomposition
    Q, R = qr(Z)
    Q = Matrix(Q)
    
    # Normalize phases (so Q is Haar distributed)
    d = diag(R)
    ph = d ./ abs.(d)
    Q .* ph'
end