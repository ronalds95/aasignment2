### A Pluto.jl notebook ###
# v0.14.8

using Markdown
using InteractiveUtils

# ╔═╡ 95eb99c0-d0a2-11eb-22b1-6b4e66df3cc1
using Pkg

# ╔═╡ d1c4a564-e192-4414-a51b-dcecbfa90eb9
Pkg.activate("Project.toml")

# ╔═╡ c25ab905-9f57-4abf-8c00-faf1c757c07f
using Markdown

# ╔═╡ 8550e7d7-c0a6-46c4-83f4-15e8a90e1682
using InteractiveUtils

# ╔═╡ 49562c01-5f1a-4f4d-9e98-56d4cc480864
using PlutoUI

# ╔═╡ de1d08ef-f2bd-4436-ab5e-bf815df080f2
using Random

# ╔═╡ 3e09e0de-2840-4aed-8577-1b8cf4bd8a69
using Statistics

# ╔═╡ f6f6b05a-fe19-4e8e-b015-04823dd854ac
using DataStructures

# ╔═╡ e2a6c2c4-a3b2-412e-b110-ee64f865d26d
using Images

# ╔═╡ 0e1c4f4c-bb9b-48f8-9c90-53203045bd95
using FFTW

# ╔═╡ 18ed3010-1791-4027-8519-9d9c835098fe
using Flux

# ╔═╡ e1f27306-edb5-4e61-b49b-fb240b5af65e
using Plots

# ╔═╡ 9014c807-c656-42f5-b330-2941211b7d45
using Knet, JLD2, FileIO

# ╔═╡ 3ef39061-f8ee-49db-abc4-d703846ec81c
using Statistics: mean

# ╔═╡ ed6712e9-6fed-4d34-b140-3c241cae8ef6
KAGGLE_FILE_PATH_NORMAL = "C:\\Users\\ronal\\Desktop\\Home\\_\\aig\\kaggle_photos\\chest_xray\\train\\NORMAL"

# ╔═╡ 2dca15f3-2e5b-43d8-9b1d-ac4918b5a1ae
KAGGLE_FILE_PATH_PNEUMONIA = "C:\\Users\\ronal\\Desktop\\Home\\_\\aig\\kaggle_photos\\chest_xray\\train\\PNEUMONIA"

# ╔═╡ 30162845-d722-44ec-a74a-2bf38e27bd1d
TARGET_NORMAL_DIR = "Dataset/Normal"

# ╔═╡ 7de666c6-6ae2-4969-9a43-7bd4881d0afe
TARGET_PNEUMONIA_DIR = "Dataset/Pneumonia"

# ╔═╡ 60be666f-71fc-4c26-b081-c1d1c3e55028
struct image_names
	nn
    x0
end

# ╔═╡ dd5c1d49-e275-4587-a614-6bb07fb582cf
struct image_names2
	nn
    x0
end

# ╔═╡ 2beaae77-806b-4af1-8ea3-7e1c11f25394
NORMAL = readdir(KAGGLE_FILE_PATH_NORMAL)

# ╔═╡ d69d42b4-ee54-486b-89ab-ea28fc90840c
PNEUMONIA = readdir(KAGGLE_FILE_PATH_PNEUMONIA)

# ╔═╡ c2696f8a-16cc-4c50-9118-1884ad1ddff5
NORMAL

# ╔═╡ 173294e6-6be1-476e-a9eb-a3dfcbe5f535
PNEUMONIA

# ╔═╡ b88e64ab-afcd-43ad-a216-018e531dd91c
Random.shuffle(NORMAL)

# ╔═╡ 85953084-df5f-4ca7-96ec-95cf5e1d0172
Random.shuffle(PNEUMONIA)

# ╔═╡ e1664357-acab-4a02-97b0-c087e0baff03


# ╔═╡ 780dd573-5a11-43d0-836a-09d8bb70467f
module Net5

# ╔═╡ c18c9d64-e2fe-4de2-bc54-800b2c55b627
import ..load_model

# ╔═╡ 42ca98f2-ddcd-4400-ac4e-9815a3792465
struct LeNet5C
    conv1_w; conv1_b
    conv2_w; conv2_b
    conv3_w; conv3_b
    fc4_w; fc4_b
    fc5_w; fc5_b
end

# ╔═╡ 0e76c807-9634-4d8a-ab40-df43617f2fa8
struct LeNet5Iterator
    nn
    x0
end

# ╔═╡ 28d7760c-069a-4150-b351-d7276c9d6397
function Base.iterate(it::LeNet5Iterator, state)
    x, layer = state
    nn = it.nn
    if layer == :conv1
        # convolution with 5x5 kernel, channel 1 -> 6
        x = conv4(nn.conv1_w, x; padding=0, stride=1, mode=0) .+ nn.conv1_b
        x = tanh.(x)
        return ((:conv1, x), (x, :pool1))
    elseif layer == :pool1
        # average pooling
        x = pool(x; window=2, padding=0, stride=2, mode=1)
        return ((:pool1, x), (x, :conv2))
    elseif layer == :conv2
        # convolution with 5x5 kernel, channel 6 -> 16
        x = conv4(nn.conv2_w, x; padding=0, stride=1, mode=0) .+ nn.conv2_b
        x = tanh.(x)
        return ((:conv2, x), (x, :pool2))
    elseif layer == :pool2
        # average pooling
        x = pool(x; window=2, padding=0, stride=2, mode=1)
        return ((:pool2, x), (x, :conv3))
    elseif layer == :conv3
        # convolution with 5x5 kernel, channel 16 -> 120
        x = conv4(nn.conv3_w, x; padding=0, stride=1, mode=0) .+ nn.conv3_b
        x = tanh.(x)
        return ((:conv3, x), (x, :fc4))
    elseif layer == :fc4
        x = reshape(x,:,size(x,4))
        x = nn.fc4_w * x .+ nn.fc4_b
        x = tanh.(x)
        return ((:fc4, x), (x, :fc5))
    elseif layer == :fc5
        x = nn.fc5_w * x .+ nn.fc5_b
        return ((:fc5, x), (x, :prob))
    elseif layer == :prob
        x = softmax(x)
        return ((:prob, x), nothing)
    else
        throw(Exception("Layer Not Valid"))
    end
end

# ╔═╡ c06008ea-9d7a-4a23-9a54-caf887711f38
Base.iterate(it::LeNet5Iterator, ::Nothing) = nothing

# ╔═╡ 8c4a3bb7-6e07-4644-ba5b-b61342a4bbe6
function Base.iterate(it::LeNet5Iterator)
    Base.iterate(it, (it.x0, :conv1))
end

# ╔═╡ b32c7bb9-2b97-4f00-b9d5-8c023e78fc52
Base.IteratorSize(it::LeNet5Iterator) = Base.HasLength()

# ╔═╡ 539c64c6-de81-4b86-ba63-848d61610b8e
Base.length(it::LeNet5Iterator) = 8

# ╔═╡ 7084b271-0d8d-4690-a11a-a9cc1348b571
Base.last(it::LeNet5Iterator) = Iterators.drop(it,7) |> collect |> last |> last

# ╔═╡ 2bf276ff-08f8-4d18-986b-c9433dd3713e
begin
	function load_model(::Type{LeNet5}, path; atype=nothing)::LeNet5
	    if atype === nothing
	        atype = (Knet.gpu() >= 0 ? KnetArray{Float32} : Array{Float32})
	    end
	    arr(a) = convert(atype, a)
	    jldopen(path, "r") do file
	        LeNet5(
	            file["lenet5/conv1_w"] |> arr,
	            file["lenet5/conv1_b"] |> arr,
	            file["lenet5/conv2_w"] |> arr,
	            file["lenet5/conv2_b"] |> arr,
	            file["lenet5/conv3_w"] |> arr,
	            file["lenet5/conv3_b"] |> arr,
	            file["lenet5/fc4_w"]   |> arr,
	            file["lenet5/fc4_b"]   |> arr,
	            file["lenet5/fc5_w"]   |> arr,
	            file["lenet5/fc5_b"]   |> arr,
	        )
	    end
	end
	
	end
end

# ╔═╡ a85eed3d-2be4-4f6f-b2d0-0191e04ec47a
#Lenet 5 implementation ends

# ╔═╡ 043389e7-8d1e-40db-8f74-0fdb34be68b3


# ╔═╡ 3185b538-0210-4c9c-a58e-e6a4c311fd88


# ╔═╡ 5c54e3d3-2929-4c34-9535-995ff48bef9c
combinationsolution

# ╔═╡ 0107c90f-682a-4ff2-bda3-506fd6d906bf
begin
	normalpneunomiaxraycombined = [NORMAL; PNEUMONIA]
	
	combinationCalculate = [fill(0, size(NORMAL)); fill(1, size(PNEUMONIA))];
	
	combinationsolution
end

# ╔═╡ 3b95a757-bfab-461c-882d-e3431e2decd2
#Backpropagation training model

# ╔═╡ ded6f420-4af1-4135-a228-89d7045f5e79
begin
	?SDG
end

# ╔═╡ 2fff6667-ac67-4b55-8535-00637ef12922
begin
	?Flux.train!
end

# ╔═╡ 560f0dc9-5666-4f7e-bbff-53b66d06dec2
model.w

# ╔═╡ c3acdaf0-1114-414c-a650-62c7cadc585a
model.b

# ╔═╡ 44ad95a5-2e38-4b3e-9d71-205bdaf713d3
typeof(model.w)

# ╔═╡ f035f76a-33dd-403d-8253-7302748affb0
model(normalpneunomiaxraycombined[end])

# ╔═╡ 96baf013-d727-45cc-99ad-d4bb7abd0c08
begin
	using Flux.Tracker
	back!(loss)
end

# ╔═╡ bd2404a7-30f8-44e3-b33b-0f471b8bd572
model.w

# ╔═╡ 713ff8cf-5ef4-423c-a40c-ce207a606bca
model.b.grad

# ╔═╡ 19377677-a5a5-49ee-b85b-b761d454025f
model.w.grad

# ╔═╡ 67fb45b0-9c9b-46bb-a717-3f89db34c800
#data before computing

model(normalpneunomiaxraycombined[1])

# ╔═╡ c6a8dc01-2091-476a-9392-fe717958485d
#gradient descent function to train and get closer to zero

for step in 1:1000
	i = rand(1:length(normalpneunomiaxraycombined))
ƞ =0.01
loss = Flux.mos(model(normalpneunomiaxraycombined[1]), combinationCalculate[1])
back!(loss)
model.w.data .-= model.w.grad * ƞ
model.b.data .-= model.b.grad * ƞ
end

# ╔═╡ a5071141-dc79-4633-8def-bcbe0a80c43c
#data after computing decreasing and getting closer to zero, 

model(normalpneunomiaxraycombined[end-1])

# ╔═╡ 1171cc1d-7899-4581-9db5-9146aff278c8
#training algorithm
for step in 1:100
Flux.train!(L, zip(normalpneunomiaxraycombined,combinationCalculate), opt)
end

# ╔═╡ 60ce7383-9b2b-4641-8507-ce923e80a0aa


# ╔═╡ 44582174-6baf-49c1-a10e-a11c63f525d8


# ╔═╡ dcc9f0d3-1a27-45db-ad20-7812a7af77bc
params(model)

# ╔═╡ 4e8bb2c7-2dab-4dd9-b602-350f371bf4c6


# ╔═╡ 70b7326d-5ea3-4ab3-86a1-7fc0fda19d0f
begin
	##visualizing result
	
	contour(0:.1:1, 0:.1.1, (NORMAL, PNEUMONIA) -> model([NORMAL, PNEUMONIA])[].data, fill=true)
	scatter!(first.(NORMAL), last(PNEUMONIA), label ="Normal")
	scatter!(first.(PNEUMONIA), last(NORMAL), label ="Pneumonia")
	xlabel!("Normal x-ray patients")
	ylabel!("Pneumonia x-ray patients")
end

# ╔═╡ b22adec3-6b6f-439d-bb87-bc2556ac576f


# ╔═╡ 2c205d2e-6791-4541-8be2-3b6a59b49db4
begin
	model = Dense(2, 1, σ)
	
	
	L(NORMAL,PNEUMONIA) = Flux.mse(model(NORMAL), PNEUMONIA)
	
	opt = SDG(params(model))
	
	Flux.train!(L, zip(normalpneunomiaxraycombined,combinationCalculate), opt)
end

# ╔═╡ d1b27784-5e97-436d-a9dd-05759a0cc2b7
model = Dense(2, 1, σ)

# ╔═╡ a27ae224-e12c-4335-a671-8ccaf6615d4c
function loss(nn::LeNet5, x, y)
    #mean(sum((nn(x) .- y) .^ 2, dims=1))
    x = last(nn(x))
    e = eps(eltype(x))
    (-y .* log.(x .+ e) .- (1 .- y) .* log.(1 .- x .+ e)) |> a->sum(a,dims=1) |> mean
end

# ╔═╡ da23fdcd-5280-4114-b8fb-4df56ef1e5a9
import .Net5: LeNet5

# ╔═╡ 253da97e-4838-40c3-829d-e8cf1a16df7a
model = Dense(2, 1, σ)

# ╔═╡ 131cabc5-7ad2-42bc-9def-65c6e76f03f3
function LeNet5()
    LeNet5(
        param(5,5,1,6), param0(1,1,6,1),
        param(5,5,6,16), param0(1,1,16,1),
        param(5,5,16,120), param0(1,1,120,1),
        param(84,120), param0(84),
        param(10,84), param0(10),
    )
end

# ╔═╡ 7ad16c6a-83ba-4259-b551-c3f035577409
loss = Flux.mos(model(normalpneunomiaxraycombined[1]), combinationCalculate[1])

# ╔═╡ 332bfb67-dd36-4b13-bcd3-32ec2c68780c
function fit(nn::LeNet5, data)
    sgd((x,y) -> loss(nn,x,y), data; lr=0.01, params=params(nn))
end

# ╔═╡ a7754f9f-8473-47d3-8834-5296be41f5b6
import ..loss

# ╔═╡ 48146e67-712e-4af6-beab-92c7582fc5d4
function (nn::LeNet5)(x)
    # check input
    if ndims(x) == 2
        w, h = size(x)
        x = reshape(x, (w, h, 1, 1))
    elseif ndims(x) == 3
        w, h, n = size(x)
        x = reshape(x, (w, h, 1, n))
    end
    @assert ndims(x) == 4
    LeNet5Iterator(nn, x)
end

# ╔═╡ 32fcf48d-7fc4-4e60-972e-1c36c8053cc0
import ..save_model

# ╔═╡ f889f6c3-6980-4504-a5e5-f6ba11d8b6de
import ..fit

# ╔═╡ 8d662bc9-9794-4e4b-9df1-08393699b158
function save_model(path, model::LeNet5)
    arr(a) = convert(Array{Float32}, a)
    jldopen(path, "w") do file
        file["lenet5/conv1_w"] = model.conv1_w |> value |> arr
        file["lenet5/conv1_b"] = model.conv1_b |> value |> arr
        file["lenet5/conv2_w"] = model.conv2_w |> value |> arr
        file["lenet5/conv2_b"] = model.conv2_b |> value |> arr
        file["lenet5/conv3_w"] = model.conv3_w |> value |> arr
        file["lenet5/conv3_b"] = model.conv3_b |> value |> arr
        file["lenet5/fc4_w"]   = model.fc4_w   |> value |> arr
        file["lenet5/fc4_b"]   = model.fc4_b   |> value |> arr
        file["lenet5/fc5_w"]   = model.fc5_w   |> value |> arr
        file["lenet5/fc5_b"]   = model.fc5_b   |> value |> arr
    end;
end

# ╔═╡ Cell order:
# ╠═c25ab905-9f57-4abf-8c00-faf1c757c07f
# ╠═8550e7d7-c0a6-46c4-83f4-15e8a90e1682
# ╠═95eb99c0-d0a2-11eb-22b1-6b4e66df3cc1
# ╠═d1c4a564-e192-4414-a51b-dcecbfa90eb9
# ╠═49562c01-5f1a-4f4d-9e98-56d4cc480864
# ╠═de1d08ef-f2bd-4436-ab5e-bf815df080f2
# ╠═3e09e0de-2840-4aed-8577-1b8cf4bd8a69
# ╟─f6f6b05a-fe19-4e8e-b015-04823dd854ac
# ╟─e2a6c2c4-a3b2-412e-b110-ee64f865d26d
# ╟─0e1c4f4c-bb9b-48f8-9c90-53203045bd95
# ╠═18ed3010-1791-4027-8519-9d9c835098fe
# ╠═e1f27306-edb5-4e61-b49b-fb240b5af65e
# ╠═d1b27784-5e97-436d-a9dd-05759a0cc2b7
# ╠═560f0dc9-5666-4f7e-bbff-53b66d06dec2
# ╠═c3acdaf0-1114-414c-a650-62c7cadc585a
# ╠═44ad95a5-2e38-4b3e-9d71-205bdaf713d3
# ╠═ed6712e9-6fed-4d34-b140-3c241cae8ef6
# ╠═2dca15f3-2e5b-43d8-9b1d-ac4918b5a1ae
# ╠═30162845-d722-44ec-a74a-2bf38e27bd1d
# ╠═7de666c6-6ae2-4969-9a43-7bd4881d0afe
# ╠═60be666f-71fc-4c26-b081-c1d1c3e55028
# ╠═dd5c1d49-e275-4587-a614-6bb07fb582cf
# ╠═2beaae77-806b-4af1-8ea3-7e1c11f25394
# ╠═d69d42b4-ee54-486b-89ab-ea28fc90840c
# ╠═c2696f8a-16cc-4c50-9118-1884ad1ddff5
# ╠═173294e6-6be1-476e-a9eb-a3dfcbe5f535
# ╠═b88e64ab-afcd-43ad-a216-018e531dd91c
# ╠═85953084-df5f-4ca7-96ec-95cf5e1d0172
# ╠═e1664357-acab-4a02-97b0-c087e0baff03
# ╠═780dd573-5a11-43d0-836a-09d8bb70467f
# ╠═9014c807-c656-42f5-b330-2941211b7d45
# ╠═3ef39061-f8ee-49db-abc4-d703846ec81c
# ╠═c18c9d64-e2fe-4de2-bc54-800b2c55b627
# ╠═32fcf48d-7fc4-4e60-972e-1c36c8053cc0
# ╠═a7754f9f-8473-47d3-8834-5296be41f5b6
# ╠═f889f6c3-6980-4504-a5e5-f6ba11d8b6de
# ╠═da23fdcd-5280-4114-b8fb-4df56ef1e5a9
# ╠═42ca98f2-ddcd-4400-ac4e-9815a3792465
# ╠═131cabc5-7ad2-42bc-9def-65c6e76f03f3
# ╠═48146e67-712e-4af6-beab-92c7582fc5d4
# ╠═0e76c807-9634-4d8a-ab40-df43617f2fa8
# ╠═8c4a3bb7-6e07-4644-ba5b-b61342a4bbe6
# ╠═28d7760c-069a-4150-b351-d7276c9d6397
# ╠═c06008ea-9d7a-4a23-9a54-caf887711f38
# ╠═b32c7bb9-2b97-4f00-b9d5-8c023e78fc52
# ╠═539c64c6-de81-4b86-ba63-848d61610b8e
# ╠═7084b271-0d8d-4690-a11a-a9cc1348b571
# ╠═a27ae224-e12c-4335-a671-8ccaf6615d4c
# ╠═332bfb67-dd36-4b13-bcd3-32ec2c68780c
# ╠═8d662bc9-9794-4e4b-9df1-08393699b158
# ╠═2bf276ff-08f8-4d18-986b-c9433dd3713e
# ╠═a85eed3d-2be4-4f6f-b2d0-0191e04ec47a
# ╠═043389e7-8d1e-40db-8f74-0fdb34be68b3
# ╠═3185b538-0210-4c9c-a58e-e6a4c311fd88
# ╠═5c54e3d3-2929-4c34-9535-995ff48bef9c
# ╠═0107c90f-682a-4ff2-bda3-506fd6d906bf
# ╠═253da97e-4838-40c3-829d-e8cf1a16df7a
# ╠═f035f76a-33dd-403d-8253-7302748affb0
# ╠═7ad16c6a-83ba-4259-b551-c3f035577409
# ╠═3b95a757-bfab-461c-882d-e3431e2decd2
# ╠═bd2404a7-30f8-44e3-b33b-0f471b8bd572
# ╠═713ff8cf-5ef4-423c-a40c-ce207a606bca
# ╠═96baf013-d727-45cc-99ad-d4bb7abd0c08
# ╠═19377677-a5a5-49ee-b85b-b761d454025f
# ╠═67fb45b0-9c9b-46bb-a717-3f89db34c800
# ╠═c6a8dc01-2091-476a-9392-fe717958485d
# ╠═a5071141-dc79-4633-8def-bcbe0a80c43c
# ╠═ded6f420-4af1-4135-a228-89d7045f5e79
# ╠═2fff6667-ac67-4b55-8535-00637ef12922
# ╠═2c205d2e-6791-4541-8be2-3b6a59b49db4
# ╠═1171cc1d-7899-4581-9db5-9146aff278c8
# ╠═60ce7383-9b2b-4641-8507-ce923e80a0aa
# ╠═44582174-6baf-49c1-a10e-a11c63f525d8
# ╠═dcc9f0d3-1a27-45db-ad20-7812a7af77bc
# ╠═4e8bb2c7-2dab-4dd9-b602-350f371bf4c6
# ╠═70b7326d-5ea3-4ab3-86a1-7fc0fda19d0f
# ╠═b22adec3-6b6f-439d-bb87-bc2556ac576f
