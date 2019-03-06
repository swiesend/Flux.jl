using Test
using Flux
using Flux: onehot, onehotbatch

alphabet = map(c -> convert(Char, c), 0:255)
data = onehotbatch("Hello, world!", alphabet)

@testset "LSTM" begin
    m = Chain(
        LSTM(256,10),
        LSTM(10,256))
    
    @test size(m(data)) == (256, 13)
end

@testset "peephole LSTM" begin
    m = Chain(
        PLSTM(256,10),
        PLSTM(10,256))

    @test size(m(data)) == (256, 13)
end

@testset "fully connected LSTM" begin
    m = Chain(
        FCLSTM(256,10),
        FCLSTM(10,256))

    @test size(m(data)) == (256, 13)
end

@testset "convolutional LSTM" begin
    # 5 batches of 1 Channel blank 28x28 grayscale images
    r = rand(Float32, 28, 28, 1, 5)
    
    m = Chain(
        Conv((2, 2), 1=>3, relu))
    @test size(m(r)) == (27, 27, 3, 5)

    m = Chain(ConvLSTM(28, 28, relu))
    @show m(r)
    # @test size(m(r)) == (10, 5)
end
