//
// Created by cmh on 2017. 11. 4..
//

#ifndef IMPORVEDNEURALNETWORK_NETWORK_H
#define IMPORVEDNEURALNETWORK_NETWORK_H

#include "Matrix.hpp"
#include <functional>
#include <utility>

typedef std::pair< std::vector<Matrix>, std::vector<Matrix> > Grads;//{가중치의 기울기, 편향의 기울기}들의 배열(vector)를 담은 형식을 정의한다.

class Block;//위에 쓸까봐 미리 선언부터 해놓음
class NeuralNetwork;// ''

enum ActivationFuncs{//활성화 함수들 종류를 기록해놓은 열거형
                                      //사실 여기서 쓰는건 Sigmoid밖에.....읍읍!
    None,
    Sigmoid,
    Step_Function,
    SoftMax,
    ReLU
};

enum ErrorFuncs{//오차함수 종류
    ACE, //크로스 엔트로피 오차
    MSE //평균 오차 제곱
};

class ActivationLayer{//활성화 함수를 이루는 ActivationLayer의 상속에서 Base가 되는 가상 클래스이다.
public:
  //순수 가상함수들
    virtual Matrix feedforward(const Matrix &X) = 0;
    virtual Matrix feedbackward(const Matrix &input) = 0;
    virtual Matrix& GetLast_y() = 0;
};

class AffineLayer{
public:
    explicit AffineLayer(size_t output_size);//생성자

    Matrix feedforward(const Matrix &X);//앞에서 맥이기
    Matrix feedbackward(const Matrix &input);//뒤에서 맥이기

    void SetUpSize(size_t Input_size);//사이즈 관리
    void SetUpSizeWithStd(size_t Input_size, int node_num);//지능적으로 사이즈 관리

    Matrix& GetW();
    Matrix& GetB();
    Matrix& GetDw();
    Matrix& GetDb();
    Matrix& GetLast_x();
    Matrix& GetLast_y();
    size_t GetOutputSize();
    int GetNumOfNodes();

private:
    Matrix W;
    Matrix b;
    size_t Output_Size;

    //역전파법
    Matrix Dw;
    Matrix Db;
    Matrix Last_x;
    Matrix Last_y;
};

class SigmoidLayer : public ActivationLayer{//ActivationLayer를 상속하는 SigmoidLayer
public:
    Matrix feedforward(const Matrix &X) override;
    Matrix feedbackward(const Matrix &input) override;
    Matrix& GetLast_y() override ;

private:
    Matrix Last_y;
};

class SoftMaxLayer : public ActivationLayer{

};

class Block{//블록이다. AffineLayer와 ActivationLayer를 가진다.
public:
    Block(AffineLayer layer, ActivationFuncs Acts) : Affine(std::move(layer))
    {
        switch (Acts)
        {
            case Sigmoid:
                ActL =new SigmoidLayer();
                break;
            default://현  신경망망에서는 Sigmoid활성화함수만 사용하므로 다른 활성화함수는 구현해놓지 않았다.
                break;
        }
    }


    Matrix feedforward(const Matrix &X);//블럭단위로 feedforward시키기
    Matrix feedbackward(const Matrix &output);//블럭단위로 feedbackward시키기
    Matrix feedbackward(double input);//사용한적은 없다. 만약을 위해 오버로딩해놓음.

    AffineLayer& GetAffine();//AffineLayer얻기
    ActivationLayer& GetActL();//ActivationLayer얻기
private:
    AffineLayer Affine;//자체로 가지고 있다.
    ActivationLayer* ActL;//포인터로 지정
};

class NeuralNetwork{//네트워크 구조
public:
    explicit NeuralNetwork(size_t Input_Size);//생성자

    NeuralNetwork &operator<<(Block &layer);//블럭을 vector<Block> Blocks에 추가시킨다.

    Matrix predict(Matrix X);//예측하기, 네트워크 단위의 feedforward를 말한다.

    double loss(Matrix &x, Matrix &t);//손실함수
    Grads gradient(Matrix x, Matrix t);//오차역전파법 전용 미분(기울기 산출)
    Grads numerical_gradient(Matrix x, Matrix t);//수치미분

    void Train(std::pair<Matrix, Matrix> Train_Data, double Rate);//학습하기
    void Trains(std::vector<std::pair<Matrix, Matrix>> Train_Datas, double Rate);//학습데이터 여러개를 한꺼번에 학습
    void Diff(std::pair<Matrix, Matrix> Train_Data);//오차역전파법과 수치미분으로 나온 2개의 기울기를 비교해 본다.

    double Last_loss = 0; // for debug
    Matrix& GetLast_t();//마지막으로 학습하기 위해 입력된 t값을 얻기 Train_Data.second에 해당한다.
private:

    class lossLayer{//역전파법을 위하여 lossLayer를 따로 Network안에 구비하였다.
    public:
        double feedforward(const Matrix &x, const Matrix &t);
        Matrix feedbackward(const Matrix &Last_y, const Matrix &Last_t);//Last_y와 Last_t가 입력된다는 점에 주의하자

    };

    Matrix Last_t;//마지막 테스트 정답 레이블
    size_t InputSize;//들어가는 사이즈
    void OptimizeSizes();//하위 Block들의 가중치의 크기들을 사이즈에 맞게 최적화 시킨다.
    std::vector<Block> Blocks;//블럭들의 리스트
    lossLayer LossLayer;//손실레이어

};
#endif //IMPORVEDNEURALNETWORK_NETWORK_H
