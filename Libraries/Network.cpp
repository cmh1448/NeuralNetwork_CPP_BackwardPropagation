//
// Created by cmh on 2017. 11. 4..
//

#include "Network.h"

AffineLayer::AffineLayer(size_t output_size) {//생성자
    Output_Size = output_size;
}

void AffineLayer::SetUpSize(size_t Input_size) {//가중치와 편향의 사이즈 설정하기
    W.resizeRandomly(Input_size,Output_Size);
    b.resizeRandomly(1, Output_Size);
}

void AffineLayer::SetUpSizeWithStd(size_t Input_size, int node_num) {//He초기값으로 자동 설정
    W.resizeRandomlyWithStd(Input_size, Output_Size, node_num);
    b.resizeRandomly(1, Output_Size);
}

Matrix AffineLayer::feedforward(const Matrix &X) {
  //AffineLayer의 구조를 말하자면 X -> AffineLayer -> Y 이런식이다.
    //Affine은 기하학에서 행렬의 내적을 뜻한다.
  //Last_y와 Last_x는 이 X와 Y를 등록하는 것이다.
    Last_y = (X^W) + b;
    Last_x = X;

    return Last_y;
}

Matrix &AffineLayer::GetW() {
    return W;
}

Matrix& AffineLayer::GetB() {
    return b;
}

Matrix& AffineLayer::GetDw() {
    return Dw;
}

Matrix& AffineLayer::GetDb() {
    return Db;
}

size_t AffineLayer::GetOutputSize() {
    return Output_Size;
}

int AffineLayer::GetNumOfNodes() {
    return (int)W.size().first;//가중치의 세로 길이가 뉴런의 개수와 같다.
}

Matrix AffineLayer::feedbackward(const Matrix &input) {//역전파법
  //Affine 식 : (X^W) + b
  //dX = input & W(T) (T는 Transpose, 즉 전치행렬이란 뜻이다.)
  //dW = X(T) ^ input

    Dw = Last_x.GetInverse() ^ input;//가중치의 미분값은 Last_x의 전치행렬과 input과의 내적과 같다.
    Db.resize(1, input.size().second);//편향의 크기는 1, input의 가로 길이이다.

    for (int i = 0; i < input.size().second; ++i) {
        for (int j = 0; j < input.size().first; ++j) {
            Db[0][i] += input[j][i];//입력값의 각 축의 총 합이 편향의 기울기이다.
        }
    }
    return input ^ W.GetInverse();//거슬러올라 d_input/d_X의 값을 반환해야 한다.
}

Matrix &AffineLayer::GetLast_x() {
    return Last_x;
}

Matrix &AffineLayer::GetLast_y() {
    return Last_y;
}

Matrix SigmoidLayer::feedforward(const Matrix &X) {//시그모이드 앞으로 먹이기
    Last_y = AIMath::ActivationFunctions::Sigmoid(X);
    //Last_y가 이름은 AffineLayer의 Last_y와 같지만 이건 SigmoidLayer에서의 Last_y즉, 출력값이란 소리다.
    return Last_y;
}

Matrix SigmoidLayer::feedbackward(const Matrix &input) {
    return input * (Last_y * (1 - Last_y));//Sigmoid함수의 도함수 = y(1-y)
    //합성함수의 미분때문에 하위함수들의 미분값의 곱들인 input을 곱한뒤 돌려주는 것이다.
}

Matrix &SigmoidLayer::GetLast_y() {
    return Last_y;
}


NeuralNetwork::NeuralNetwork(size_t Input_Size) {
    InputSize = Input_Size;
}

NeuralNetwork &NeuralNetwork::operator<<(Block &layer) {
    Blocks.push_back(layer);//Blocks에 해당 layer추가(형타입이 Block인데 이름이 layer라니 좀 이상하지만 넘어가도록 하자)
    OptimizeSizes();//가중치와 편향 크기 최적화
    return *this;
}

Matrix NeuralNetwork::predict(Matrix X) {
    if(X.size().first==1){}//입력값 검사
    else if(X.size().second == 1){
        X.Inverse();//가로 크기가 1이되도록 만들어 준다.
    }else
        throw "다층 퍼셉트론 신경망은 입력값이 벡터입니다";

    Matrix out = X;
    for (auto &Block : Blocks) {//C++11에서 추가된 foreach : Blocks의 요소들을 하나씩 돌려가며 반복한다. Block이 Block[i]와 같은거임
        out = Block.feedforward(out);//out값 갱신
    }
    return out;//out반환
}

double NeuralNetwork::loss(Matrix &x, Matrix &t) {//손살함수
    Matrix y = predict(x);//일단 predict해준다음
    Last_loss = LossLayer.feedforward(y, t);//손실 레이어에 feedforward해준다.
    return Last_loss;//그러면 손실함수까지 거친 손실값이 나오게 된다.
}

Grads NeuralNetwork::numerical_gradient(Matrix x, Matrix t) {//수치미분법

    std::function<double(Matrix)> loss_W = [&](Matrix w)->double{ return loss(x,t); };//람다 함수, AIMath::numerical_gradient에서 첫번째 인자가 함수 포인터이기 때문에 파라미터 형태를 맞추어 주기 위하여
    //람다로 재정의를 한다.
    //람다에 관한 이야기는 검색을 통해 알면 된다.

    Grads Rtrn;//{가중치의 기울기, 편향의 기울기}의 배열

    //modify later
    for (auto &block : Blocks) {//Blocks의 요소 수만큼 반복
        Rtrn.first.push_back(AIMath::NumericalGradient(loss_W, block.GetAffine().GetW()));//loss_W에 W를 대입하고 수치미분하여 나온 미분값을 대입
        Rtrn.second.push_back(AIMath::NumericalGradient(loss_W, block.GetAffine().GetB()));//loss_W에 b를 대입하고 수치미분하여 나온 미분값을 대입
    }

    return Rtrn;
}

void NeuralNetwork::OptimizeSizes() {//값 최적화
    Blocks[0].GetAffine().SetUpSizeWithStd(InputSize, (int)InputSize);//첫번째 블럭의 가중치값을 사이즈를 맞추어 준다.
    for (int i = 1; i < Blocks.size(); ++i) {
        Blocks[i].GetAffine().SetUpSizeWithStd(Blocks[i-1].GetAffine().GetOutputSize(), Blocks[i-1].GetAffine().GetNumOfNodes());//앞쪽의 뉴런 개수를 고려해가며 사이즈를 맞추어 준다.
    }
}

void NeuralNetwork::Train(std::pair<Matrix, Matrix> Train_Data, double Rate) {
    Grads Result = gradient(Train_Data.first, Train_Data.second);//오차 역전파법으로 기울기를 산출한다.
    //Grads Result = numerical_gradient(Train_Data.first, Train_Data.second);으로 수치미분으로 산출하는 방법도 있다.

    for (int i = 0; i < Blocks.size(); ++i) {//SGD방식으로 현재 가중치값을 기울기에 학습률을 곱한것만큼 빼 갱신한다.
        Blocks[i].GetAffine().GetW() -= (Result.first[i] * Rate);
        Blocks[i].GetAffine().GetB() -= (Result.second[i] * Rate);
    }


}

void NeuralNetwork::Trains(std::vector<std::pair<Matrix, Matrix>> Train_Datas, double Rate) {//여러가지 값을 한꺼번에 학습한다.
    for(const auto &Train_Data : Train_Datas)
    {
        Train(Train_Data, Rate);
    }
}

Grads NeuralNetwork::gradient(Matrix x, Matrix t) {//오차 역전파법으로 기울기 구하기
    loss(x,t);//한번 LossLayer까지 포함해서 feedforward를 싹 시켜준다.(Last_y, Last_x등 변수들이 갱신된다.)

    Matrix dout;
    dout = LossLayer.feedbackward(Blocks[Blocks.size() - 1].GetActL().GetLast_y(), t);//LossLayer를 역전파 시킨다. 이때 y값은 끝에서 2번째의 레이어, 즉 Blocks의 마지막 레이어의 SigmoidLayer에서의 Last_y값과 같으므로
    //이렇게 매개변수를 입력시켜 준다.
    for (int i = 0; i <= Blocks.size() - 1; ++i) {//Blocks.size-1이 될때까지 반복해가며 역전파시킨다.
        dout = Blocks[Blocks.size() - 1 - i].feedbackward(dout); //마지막에는 Blocks[0].feedbackward가 호출되며 for문에서 탈출한다.
    }


    Grads Rtrn;
    for (auto &Block : Blocks) {//Block들의 Dw에 기록되어있는 가중치들의 기울기들을 모아 Rtrn에 기록한다.
        Rtrn.first.emplace_back(Block.GetAffine().GetDw());//emplace_back은 push_back과 같은거다.
        Rtrn.second.emplace_back(Block.GetAffine().GetDb());
    }

    return Rtrn;//Rtrn바놘
}

void NeuralNetwork::Diff(std::pair<Matrix, Matrix> Train_Data) {//오차역전파법이 제대로 동작하는지 수치미분의 미분값과 비교할수 있다.
    Grads Result = numerical_gradient(Train_Data.first, Train_Data.second);//수치미분
    Grads backR = gradient(Train_Data.first, Train_Data.second);//오차역전파

    Result.second[0].print();//수치미분의 0번블록의 편향의 기울기를 출력한다.
    backR.second[0].print();//역전파의 0번블록의 편향의기울기를 출력한다.

    std::cout<<(backR.second[0] - Result.second[0]).GetAbs().GetAverage()<<std::endl;//두 기울기의 차이의 절대값의 평균을 출력한다. 0에 가까울수록 더 차이가 적은것이다.
}

Matrix &NeuralNetwork::GetLast_t() {
    return Last_t;
}

AffineLayer &Block::GetAffine() {
    return Affine;
}

ActivationLayer &Block::GetActL() {
    return *ActL;
}

Matrix Block::feedforward(const Matrix &X) {//블록단위의 feedforward이다.
    Matrix out;
    out = Affine.feedforward(X);//AffineLayer먼저 feedforward시켜준뒤
    out = ActL->feedforward(out);//ActivationLayer까지 거쳐준다.
    return out;
}

Matrix Block::feedbackward(const Matrix &output) {//블록단위의 feedbackward
    Matrix out = ActL->feedbackward(output);//ActivationLayer먼저 역전파 시켜준뒤
    return Affine.feedbackward(out);;//AffineLayer를 역전파 시켜준다.
}

Matrix Block::feedbackward(double input) {//double형일때의 feedbackward
    Matrix temp;
    temp.resize(GetAffine().GetLast_y().size().first, GetAffine().GetLast_y().size().second, input);

    return feedbackward(temp);
}


double NeuralNetwork::lossLayer::feedforward(const Matrix &x, const Matrix &t) {//lossLayer의 feedforward
    return AIMath::ErrorFunctions::MSE(x, t);//MSE를 거친 값만 출력한다.
}

Matrix NeuralNetwork::lossLayer::feedbackward(const Matrix &Last_y, const Matrix &Last_t) {//lossLayer의 feedbackward
    return (2 / (Last_y.size().first * Last_y.size().second)) * (Last_y - Last_t);//MSE를 미분한 도함수에 값을 대입하여 출력한다.
}
