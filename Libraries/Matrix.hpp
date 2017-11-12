//
//  Network.hpp
//  NeuralNetwork
//
//  Created by cmh on 2017. 10. 1..
//  Copyright © 2017년 cmh. All rights reserved.
//

#ifndef Matrix_hpp
#define Matrix_hpp


#include <iostream>
#include <vector>//vector와 pair를 쓸수있게 한다.
#include <cmath>//exp나 log, pow같은 함수들이 있다.
#include <random>//랜덤함수 만들때 필요한 include
#include <ctime>//시간관련
#include <functional>//람다써서 그걸 함수포인터로 넘길때 필요한
//람다(Lambda)는 C++11에 추가된 기능인데, 함수 안에서 함수를 만들수 있게 한다.
typedef std::vector<std::vector<double>> MATRIX;//MATRIX정의




class Matrix
{
public:

    Matrix(); //기본 생성자 안이 비어있는 행렬 선언
    explicit Matrix(MATRIX copy); //복사 생성자, 클래스단위로 복사시켜준다.
    explicit Matrix(std::vector<double> copy);//Matrix와 vector와의 호환성을 위한 복사 생성자. n사이즈의 vector(배열)을 n X 1사이즈의 행렬로 만든다.

    //Functions
    void resize(size_t rows, size_t colums);//자신을 0으로 가득찬 행렬로 만든다. rows는 세로, colums는 가로 길이를 뜻한다.(원소 개수)
    void resize(size_t rows, size_t colums, double value);//위와 동일하지만, 0대신 value를 대입하며 resize를 실행한다.
    void resize(const Matrix& target);//편의상 만든 함수, 다른 행렬의 사이즈만 가져와 resize한다. 그러니까,0으로 가득찬 크기만 같은 행렬로 자신을 만드는 것이다.
    void resize(const Matrix& target, double value);//0대신 value를 쓴다.
    void resizeRandomly(size_t rows, size_t colums);//-1과 1사이의 랜덤한 값으로 행렬을 resize한다.
    void resizeRandomlyWithStd(size_t rows,size_t colums, int node_num);//이건 딥러닝용으로 만든건데, 앞계층의 뉴런의 개수를 입력하면 최적화된 초기값으로 만들어진 행렬을 만들어준다.
    void print() const; //행렬을 출력한다.

    std::pair<size_t, size_t> size() const; //현재 행렬의 사이즈를 출력한다.

    Matrix GetInverse() const;//전치행렬을 출력한다
    void Inverse();//자신을 전치행렬로 만든다.

    double GetSum() const; //모든 원소의 합을 구한다.
    double GetAverage() const;//모든 원소의 평균을 구한다.

    Matrix GetAbs() const; //모든 원소를 절대값으로 만든 행렬을 구한다.
    //operator Overloading
    std::vector<double>& operator[](size_t idx);//값을 수정할때 사용하는 []연산자 오버로딩(반환형이 참조자(레퍼런스)이다.)
    std::vector<double> operator[](size_t idx) const;//값을 구할떄 사용하는 []연산자 오버로딩(반환형이 그냥 일반 변수이다. )

    Matrix operator^ (const Matrix &B) const;//내적을 구할때 사용하는 연산자 오버로딩

    Matrix& operator= (const MATRIX &matrix);//대입 연산자 오버로딩

    Matrix& operator= (const std::vector<double> &vector);//대입 연산자 오버로딩 2

    Matrix& operator-=(const Matrix &mtrx);
    Matrix& operator+=(const Matrix &mtrx);
    Matrix& operator*=(const Matrix &mtrx);
    Matrix& operator/=(const Matrix &mtrx);


    Matrix operator+(const Matrix &B) const;//덧셈 연산자 오버로딩 행렬 + 행렬
    Matrix operator+(const double &B) const;//위와 같지만 행렬 + double형 변수의 연산을 위해 만들었다.
//이하는 같으므로 설명을 생략한다.
    Matrix operator-(const Matrix &B) const;
    Matrix operator-(const double &B) const;


    Matrix operator*(const Matrix &B) const;
    Matrix operator*(const double &B) const;

    Matrix operator/(const Matrix &B) const;
    Matrix operator/(const double &B) const;

    double Max() const;//현재 가지고 있는 값중 가장 큰 값

    friend Matrix operator-(const double &copy, const Matrix &Mtrx);//교환법칙/역차의 연산을 성립시키기 위해 만든 오버로딩이다.
    friend Matrix operator+(const double &copy, const Matrix &Mtrx);//다시말하면, 행렬 + double 뿐만아니라 double + 행렬 연산도 지원하기 위해 만들었다.
    friend Matrix operator*(const double &copy, const Matrix &Mtrx);
    friend Matrix operator/(const double &copy, const Matrix &Mtrx);
private:

    MATRIX Mtrx;//Matrix는 vector< vector<double> >이다. 배열의 배열이므로 이중배열이라고 할수 있다. 그래서 행렬이 되는 것이다.
    //하지만 이렇게 만들어놓으면 관리하기가 힘들고, 따로 연산이 정의되어 있지 않으므로 편리성을 위해서
    //이를 관리하는 클래스인 Matrix클래스를 만들어 Mtrx변수를 관리하게 하는 것이다.

};



namespace AIMath { //설명은 main.cpp에 나와있다.
    namespace ErrorFunctions//손실함수들
    {
        double MSE(const Matrix &y, const Matrix &t);// mean square error 평균제곱 오차 : 자세한 사항은 검색 요망(아니면 책에 나와있을듯)
        double ACE(const Matrix &y, const Matrix &t); //Cross Entropy Error 교차 엔트로피 오차 (위와 동일)
    }
    namespace ActivationFunctions {//활성화 함수들

        Matrix StepFunc(const Matrix &X);//계단함수
        Matrix SoftMax(const Matrix &x);//SoftMax
        Matrix ReLU(const Matrix &X);//ReLU
        Matrix Sigmoid(const Matrix &X);//Sigmoid

    }

    Matrix GetRandomNums(size_t rows, size_t cols, double start, double end);//start부터 end까지의 수로 이루어진 난수로 구성된 행렬을 반환한다.
    Matrix NumericalGradient(std::function<double(Matrix)> f, Matrix &x);//수치미분 : 아주 작은 수를 더한것으로 미분을 한다.
    Matrix exp(const Matrix &x);//exp함수 e의 x승이다.
    Matrix sqrt(const Matrix &x);//제곱근
    Matrix Pow(const Matrix &x, double count);//제곱 : count만큼 제곱한다
    Matrix Log(const Matrix &x);//로그
    void BroadCast(Matrix *A, Matrix *B);//행렬 A와 B를 연산할때 2개의 형태를 일치시켜주는 역할을 한다.

}


#endif /* Network_hpp */
