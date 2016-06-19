#Pkg.update()
#Pkg.add("Optim")
using Optim


function calcA1(it,jt,p,q)
    A1 = zeros(2*p,2*p)
    A1[it,it] = 1
    A1[jt+q,jt+q] = 1
    A1[p+it,p+it] = 1
    A1[p+jt+q,p+jt+q] = 1
    sA1 = .5*(A1+A1')
    return sA1
end
function calcA2(it,jt,p,q)
    A2 = zeros(2*p,2*p)
    A2[it,jt+q] = 1
    A2[p+it,p+jt+q] = -1
    sA2 = .5*(A2+A2')
    return sA2
end
function calcA3(it,jt,p,q)
    A3 = zeros(2*p,2*p)
    A3[it,jt+q] = -1
    A3[p+it,p+jt+q] = 1
    sA3 = .5*(A3+A3')
    return sA3
end
function calcA4(p)
  sA4 = eye(2*p)
  return sA4
end

function applyUpdate(it, jt, Xt_1, Lt, eta, beta, tau,p,q)
    #Given the previous matrix, a location where a prediction was made,
    #a loss matrix derived from the previous prediction and its associated loss function
    #calculate the new matrix

    #transform into log space, subtract the gradient
    #and exponentiate
    Y = expm(logm(Xt_1) - eta*Lt)
    #precompute logY for multiple uses
    logY = logm(Y)

    #calculate the coefficients for the alpha's
    #in the dual problem. These represent linear
    #constraints
    sA1 = calcA1(it,jt,p,q)
    sA2 = calcA2(it,jt,p,q)
    sA3 = calcA3(it,jt,p,q)
    sA4 = calcA4(p)
    b1 = 4*beta
    b2 = 1.0
    b3 = 1.0
    b4 = tau

    function dual_f(x::Vector)
        #dual function to minimize
        sum_alpha_A = x[1]*sA1 + x[2]*sA2 + x[3]*sA3 + x[4]*sA4
        sum_alpha_b = x[1]*b1 + x[2]*b2 + x[3]*b3 + x[4]*b4
        project = expm(logY - sum_alpha_A)
        fval = trace(project) + sum_alpha_b
        return fval
    end
    function dual_g!(x::Vector, storage)
        #gradient of dual function to minimize
        sum_alpha_A = x[1]*sA1 + x[2]*sA2 + x[3]*sA3 + x[4]*sA4
        sum_alpha_b = x[1]*b1 + x[2]*b2 + x[3]*b3 + x[4]*b4
        project = expm(logY - sum_alpha_A)
        storage[1] = trace(-sA1*project) + b1
        storage[2] = trace(-sA2*project) + b2
        storage[3] = trace(-sA3*project) + b3
        storage[4] = trace(-sA4*project) + b4
    end
    function dual_fg!(x::Vector, storage)
        #gradient of dual function to minimize
        #as well as returned value of function
        sum_alpha_A = x[1]*sA1 + x[2]*sA2 + x[3]*sA3 + x[4]*sA4
        sum_alpha_b = x[1]*b1 + x[2]*b2 + x[3]*b3 + x[4]*b4
        project = expm(logY - sum_alpha_A)
        fval = trace(project) + sum_alpha_b
        storage[1] = trace(-sA1*project) + b1
        storage[2] = trace(-sA2*project) + b2
        storage[3] = trace(-sA3*project) + b3
        storage[4] = trace(-sA4*project) + b4
        return fval
    end
    d4 = DifferentiableFunction(dual_f,dual_g!,dual_fg!)

    #lower bounds of 0
    l = [0.0, 0.0,0.0,0.0]
    #no upper bounds
    u = [Inf, Inf,Inf, Inf]
    x0 = [0.1, 0.1, 0.1, 0.1]
    res = optimize(d4,x0,  l,u, Fminbox())
    oa = Optim.minimizer(res)
    #reconstruct matrix from dual
    sum_optimal_alpha_A = oa[1]*sA1 + oa[2]*sA2 + oa[3]*sA3 + oa[4]*sA4
    Xt = expm(logY - sum_optimal_alpha_A)
    return Xt
end


function calcLt(Lt, g, it,jt,p,q)
    #build loss matrix from gradient
    Lt[:,:] = 0
    Lt[it,jt+q] = g
    Lt[jt+q,it] = g
    Lt[p+it,p+jt+q] = -g
    Lt[p+jt+q,p+it] = -g
end

function calcLoss(it,jt,prediction, true_mtx, loss_type = "square")
    #only G-lipshitz loss functions allowed
    if loss_type == "square"
        return (prediction - true_mtx[it,jt])^2, 2*prediction - 2*true_mtx[it,jt]
    elseif loss_type == "absolute"
        diff = prediction - true_mtx[it,jt]
        lt = .5*abs(diff)
        grad = .5*sign(diff)
        return lt, grad
    elseif loss_type == "logistic"
        expon = exp(-prediction*true_mtx[it,jt])
        lt = log(1+expon)/log(2)
        grad = (-true_mtx[it,jt]*expon)/(log(2)*(1+expon))
        return lt, grad
    else
        throw(ArgumentError("loss type $loss_type not implemented yet"))
    end
end

function randomIndices(dim1,dim2)
    mat_indices = []
    for i=1:dim1
        for j=1:dim2
            push!(mat_indices,(i,j))
        end
    end
    shuffle!(mat_indices)
    mat_indices
end

function runOLO(true_mtx,T, β,loss_type="square")
    m,n = size(true_mtx)
    # τ = Θ(√(nm))
    τ = √(n*m)
    p = m+n
    q = m
    if loss_type == "square"
        G = 4.0
    elseif loss_type == "absolute"
        G = .5
    elseif loss_type == "logistic"
        G = 1/(2*log(2))
    else
        throw(ArgumentError("loss type $loss_type not implemented yet"))
    end

    γ = 4*(G^2)
    N = 2*p
    #the following η was used to prove a theoretical result
    #in Hazan et al. but performs worse than just setting it to 1
    #η = τ*log(N)/(β*γ*T)
    η = 1.0

    #initialize matrix (it is a symmetric version ([0 W; W 0]) of the implicit
    #matrix W we're predicting, but factored into pxp matrices P and N using
    #(β,τ)-decomposability of matrices. X becomes [P 0; 0 N], a 2p x 2p matrix
    #For more explanation see Hazan et al

    Xt = (τ/N)*eye(N)

    #pick random ordering of online iterations,
    #this could be chosen by an adversary
    ordering = randomIndices(m,n)

    #initialize Lt with zeros,
    #this will be updated by calcLt before it is used
    Lt = zeros(2*p,2*p)


    losses = zeros(1,T)
    running_avg_losses = zeros(1,T-4)
    for t=1:T
        tic()
        it,jt = ordering[t]
        prediction = Xt[it,jt+q] - Xt[p+it,p+jt+q]
        loss,g = calcLoss(it,jt,prediction, true_mtx, loss_type)
        calcLt(Lt, g,it,jt,p,q)
        #@show it,jt
        #@show prediction

        Xt = applyUpdate(it, jt, Xt, Lt, η, β, τ, p,q)
        #can reconstruct original matrix W if we want
        #symW = Xt[1:p,1:p] - Xt[p+1:end,p+1:end]
        #W = symW[1:m,m+1:m+n]

        losses[t] = loss
        if t > 4
          running_avg_loss = sum(losses[t-4:t])/5
          running_avg_losses[t-4] = running_avg_loss
          if t % 20 == 0
            @show running_avg_loss
          end
        end
        toc()

    end
    @show running_avg_losses
end

function runExample()
  tic()
  m = 100
  n = 80
  true_mtx = rand([-1,1],m,n)
  T = round(Int,(m*n)/2)
  β = 1
  W = runOLO(true_mtx, T, β)
  toc()
end

runExample()
