### A Pluto.jl notebook ###
# v0.14.8

using Markdown
using InteractiveUtils

# ╔═╡ 51561b8e-d201-11eb-2923-ff55efeb2aae
using Pkg

# ╔═╡ 24a086ca-4f94-4219-b679-5c8beec9d1e5
using Flux

# ╔═╡ 0a4c72eb-c4d7-4fbf-b745-4be75d4092b0
using Flux.Data: DataLoader

# ╔═╡ db7c77b7-54a6-4d60-a618-1e65afc9a7d3
using Flux.Optimise: Optimiser, WeightDecay

# ╔═╡ e95c4aab-c96d-4fb5-978a-55dd2d88b92d
using Flux: onehotbatch, onecold, logitcrossentropy

# ╔═╡ fc5ad24b-c529-438b-96d4-863e6af213d8
using Statistics, Random

# ╔═╡ f4a0b00e-af8b-41ad-ad4e-d43701f8a887
using Parameters: @with_kw

# ╔═╡ 7b010992-4308-4fdb-a257-007e557be8bb
using Logging: with_logger, global_logger

# ╔═╡ 483fc590-f6e9-4c2c-92c0-fe2d36bb485e
using TensorBoardLogger: TBLogger, tb_overwrite, set_step!, set_step_increment!

# ╔═╡ 4d969c13-883d-494c-9f3c-fc929790ec4e
using CUDAapi

# ╔═╡ c111e216-35ed-4c9d-9f18-f48bfd4a0df4


# ╔═╡ 99cbbd36-3ffd-4ab3-8b6d-f56efa2f7d68
import ProgressMeter

# ╔═╡ 4525e2df-5c2c-4b20-9dcc-97ba28d5c70c
import MLDatasets

# ╔═╡ 3548f18c-f0b9-4d20-bc68-120a5b4a6124
import DrWatson: savename, struct2dict

# ╔═╡ 59b3049e-1d99-41d6-baba-1d22e067bd3b
import BSON

# ╔═╡ 1b0d88a6-1d43-4980-869c-082560b46247
#LeNet Implementation shuffling

function LeNet5(; imgsize=(28,28,1), nclasses=10) 
    out_conv_size = (imgsize[1]÷4 - 3, imgsize[2]÷4 - 3, 16)
    
    return Chain(
            x -> reshape(x, imgsize..., :),
            Conv((5, 5), imgsize[end]=>6, relu),
            MaxPool((2, 2)),
            Conv((5, 5), 6=>16, relu),
            MaxPool((2, 2)),
            x -> reshape(x, :, size(x, 4)),
            Dense(prod(out_conv_size), 120, relu), 
            Dense(120, 84, relu), 
            Dense(84, nclasses)
          )
end

# ╔═╡ c786b922-66b9-48e3-a599-eb55e3592ba5
function get_data(args)
    xtrain, ytrain = MLDatasets.MNIST.traindata(Float32, dir=args.datapath)
    xtest, ytest = MLDatasets.MNIST.testdata(Float32, dir=args.datapath)

    xtrain = reshape(xtrain, 28, 28, 1, :)
    xtest = reshape(xtest, 28, 28, 1, :)

    ytrain, ytest = onehotbatch(ytrain, 0:9), onehotbatch(ytest, 0:9)

    train_loader = DataLoader(xtrain, ytrain, batchsize=args.batchsize, shuffle=true)
    test_loader = DataLoader(xtest, ytest,  batchsize=args.batchsize)
    
    return train_loader, test_loader
end

# ╔═╡ 7fb9bf36-9b47-414b-9662-02f529ce4575
loss(ŷ, y) = logitcrossentropy(ŷ, y)

# ╔═╡ 58c495c8-7ff5-4675-9ca3-1f5a6fc194f1
num_params(model) = sum(length, Flux.params(model))

# ╔═╡ 6150bf77-62bc-4c08-9ac9-2cbe6018b1c4
round4(x) = round(x, digits=4)

# ╔═╡ c77453db-7a32-48a8-b899-ea9518884f07
function eval_loss_accuracy(loader, model, device)
    l = 0f0
    acc = 0
    ntot = 0
    for (x, y) in loader
        x, y = x |> device, y |> device
        ŷ = model(x)
        l += loss(ŷ, y) * size(x)[end]        
        acc += sum(onecold(ŷ |> cpu) .== onecold(y |> cpu))
        ntot += size(x)[end]
    end
    return (loss = l/ntot |> round4, acc = acc/ntot*100 |> round4)
end

# ╔═╡ f1b5669e-ff8d-4728-8b3c-18945f44d1da
@with_kw mutable struct Args
    η = 3e-4             # learning rate
    λ = 0                # L2 regularizer param, implemented as weight decay
    batchsize = 128      # batch size
    epochs = 20          # number of epochs
    seed = 0             # set seed > 0 for reproducibility
    cuda = true          # if true use cuda (if available)
    infotime = 1 	     # report every `infotime` epochs
    checktime = 5        # Save the model every `checktime` epochs. Set to 0 for no checkpoints.
    tblogger = false      # log training with tensorboard
    savepath = nothing    # results path. If nothing, construct a default path from Args. If existing, may overwrite
    datapath = joinpath(homedir(), "Datasets", "MNIST") # data path: change to your data directory 
end

# ╔═╡ 36ec5004-ffd7-4faa-bdb5-2e1b3209db93
function train(; kws...)
    args = Args(; kws...)
    args.seed > 0 && Random.seed!(args.seed)
    use_cuda = args.cuda && CUDAapi.has_cuda_gpu()
    if use_cuda
        device = gpu
        @info "Training on GPU"
    else
        device = cpu
        @info "Training on CPU"
    end

    ## DATA
    train_loader, test_loader = get_data(args)
    @info "Dataset MNIST: $(train_loader.nobs) train and $(test_loader.nobs) test examples"

    ## MODEL AND OPTIMIZER
    model = LeNet5() |> device
    @info "LeNet5 model: $(num_params(model)) trainable params"    
    
    ps = Flux.params(model)  

    opt = ADAM(args.η) 
    if args.λ > 0 
        opt = Optimiser(opt, WeightDecay(args.λ))
    end
    
    ## LOGGING UTILITIES
    if args.savepath == nothing
        experiment_folder = savename("lenet", args, scientific=4,
                    accesses=[:batchsize, :η, :seed, :λ]) # construct path from these fields
        args.savepath = joinpath("runs", experiment_folder)
    end
    if args.tblogger 
        tblogger = TBLogger(args.savepath, tb_overwrite)
        set_step_increment!(tblogger, 0) # 0 auto increment since we manually set_step!
        @info "TensorBoard logging at \"$(args.savepath)\""
    end
    
    function report(epoch)
        train = eval_loss_accuracy(train_loader, model, device)
        test = eval_loss_accuracy(test_loader, model, device)        
        println("Epoch: $epoch   Train: $(train)   Test: $(test)")
        if args.tblogger
            set_step!(tblogger, epoch)
            with_logger(tblogger) do
                @info "train" loss=train.loss  acc=train.acc
                @info "test"  loss=test.loss   acc=test.acc
            end
        end
    end
    
    ## TRAINING
    @info "Start Training"
    report(0)
    for epoch in 1:args.epochs
        p = ProgressMeter.Progress(length(train_loader))

        for (x, y) in train_loader
            x, y = x |> device, y |> device
            gs = Flux.gradient(ps) do
                ŷ = model(x)
                loss(ŷ, y)
            end
            Flux.Optimise.update!(opt, ps, gs)
            ProgressMeter.next!(p)   # comment out for no progress bar
        end
        
        epoch % args.infotime == 0 && report(epoch)
        if args.checktime > 0 && epoch % args.checktime == 0
            !ispath(args.savepath) && mkpath(args.savepath)
            modelpath = joinpath(args.savepath, "model.bson") 
            let model=cpu(model), args=struct2dict(args)
                BSON.@save modelpath model epoch args
            end
            @info "Model saved in \"$(modelpath)\""
        end
    end
end

# ╔═╡ cfc4ca2e-2909-4399-a59d-42b7963a6345
if abspath(PROGRAM_FILE) == @__FILE__ 
    train()
end

# ╔═╡ ebd310a0-092f-4e61-a598-b9e226f0f3fe


# ╔═╡ 4ffe5da9-d937-457b-9093-57dd0f14ec0f


# ╔═╡ Cell order:
# ╠═51561b8e-d201-11eb-2923-ff55efeb2aae
# ╠═c111e216-35ed-4c9d-9f18-f48bfd4a0df4
# ╠═24a086ca-4f94-4219-b679-5c8beec9d1e5
# ╠═0a4c72eb-c4d7-4fbf-b745-4be75d4092b0
# ╠═db7c77b7-54a6-4d60-a618-1e65afc9a7d3
# ╠═e95c4aab-c96d-4fb5-978a-55dd2d88b92d
# ╠═fc5ad24b-c529-438b-96d4-863e6af213d8
# ╠═f4a0b00e-af8b-41ad-ad4e-d43701f8a887
# ╠═7b010992-4308-4fdb-a257-007e557be8bb
# ╠═483fc590-f6e9-4c2c-92c0-fe2d36bb485e
# ╠═99cbbd36-3ffd-4ab3-8b6d-f56efa2f7d68
# ╠═4525e2df-5c2c-4b20-9dcc-97ba28d5c70c
# ╠═3548f18c-f0b9-4d20-bc68-120a5b4a6124
# ╠═59b3049e-1d99-41d6-baba-1d22e067bd3b
# ╠═4d969c13-883d-494c-9f3c-fc929790ec4e
# ╠═1b0d88a6-1d43-4980-869c-082560b46247
# ╠═c786b922-66b9-48e3-a599-eb55e3592ba5
# ╠═7fb9bf36-9b47-414b-9662-02f529ce4575
# ╠═c77453db-7a32-48a8-b899-ea9518884f07
# ╠═58c495c8-7ff5-4675-9ca3-1f5a6fc194f1
# ╠═6150bf77-62bc-4c08-9ac9-2cbe6018b1c4
# ╠═f1b5669e-ff8d-4728-8b3c-18945f44d1da
# ╠═36ec5004-ffd7-4faa-bdb5-2e1b3209db93
# ╠═cfc4ca2e-2909-4399-a59d-42b7963a6345
# ╠═ebd310a0-092f-4e61-a598-b9e226f0f3fe
# ╠═4ffe5da9-d937-457b-9093-57dd0f14ec0f
