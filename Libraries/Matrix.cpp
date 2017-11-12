//
//  Network.cpp
//  NeuralNetwork
//
//  Created by cmh on 2017. 10. 1..
//  Copyright © 2017년 cmh. All rights reserved.
//

//
//  Matrix.cpp
//  MatrixClass
//
//  Created by cmh on 2017. 9. 20..
//  Copyright © 2017년 cmh. All rights reserved.
//

#include "Matrix.hpp"

//Copy Initializer
Matrix::Matrix(){
    //empty
}

Matrix::Matrix(MATRIX copy) : Mtrx(copy)
{
    //empty
}
Matrix::Matrix(std::vector<double> copy){
    *this = copy; //복사

}

//Functions
void Matrix::resize(size_t rows, size_t colums)
{
    Mtrx = MATRIX(rows, std::vector<double>(colums,0));//아... 귀찮다
    //vector초기화하는것도 vector(사이즈, 초기화할때 대입할 수) 이렇게 할수 있다.
    //그걸 2번써서 행렬 전체를 다 초기화 시켜버린것.
}

void Matrix::resize(size_t rows, size_t colums, double value) {
    Mtrx = MATRIX(rows, std::vector<double>(colums,value));
    //위와 동일
}



void Matrix::print() const
{
    int max_width = 0;
    for (int i = 0; i < Mtrx.size(); ++i) {

        for (int j = 0; j < Mtrx[0].size(); ++j) {

            if((int)log10(Mtrx[i][j])+1 > max_width)//log10에 대입하면 정수부분이 자릿수-1이다.
                max_width = (int)log10(Mtrx[i][j])+1;
        }
    }
    //최대 자릿수를 구한 중이다.
    if(max_width > 6)//이게 자릿수가 6을 넘어가면 e표현법으로 표현되면서 1.324324e-11뭐 이런식으로 나온다 그러면 자리가 11자리쯤 되기 떄문에 숫자가 출력될때 자리를 11자리로 정렬시켜 주는것이다.
        max_width = 11;//좀 어려운 개념이지만 cout.Width로 검색하면 자세한 설명을 볼수 있다.

    for(size_t i=0; i<Mtrx.size(); ++i)
    {
        for(size_t j=0; j<Mtrx[0].size(); ++j)
        {
            std::cout.width(max_width);
            std::cout<<Mtrx[i][j]<<" ";//원소별로 하나씩 돌아가며 출력한다.
        }
        std::cout<<std::endl;
    }
    std::cout<<std::endl;
}

void Matrix::resizeRandomlyWithStd(size_t rows, size_t colums,int node_num)
{
    std::mt19937 engine((unsigned int)time(NULL));//mt19937이라고, 난수출력 엔진이 있다.
    std::normal_distribution<double> dist(node_num, node_num);//표준분포 생성
    auto generater = std::bind(dist, engine);//generater 생성 engine과 dist를 사용한다.
    //auto는 형타입을 자동으로 감지하는 키워드로, C++11에서 새롭게 추가되었다.

    this->resize(rows, colums);

    for(size_t i=0; i<rows; i++)
    {
        for(size_t j=0; j<colums; j++)
        {
            Mtrx[i][j] = generater() / sqrt(node_num);//He초기값으로 분포시킨다.
        }
    }
}

void Matrix::resizeRandomly(size_t rows, size_t colums)
{
  //위와 비슷하니 알아서 알아듣도록
    std::mt19937 engine((unsigned int)time(NULL));
    std::uniform_real_distribution<double> dist(-1, 1);
    auto generater = std::bind(dist, engine);

    this->resize(rows, colums);
    for(size_t i=0; i<rows; i++)
    {
        for(size_t j=0; j<colums; j++)
        {
            Mtrx[i][j] = generater();
        }
    }

}
std::pair<size_t, size_t> Matrix::size() const
{
  //사이즈를 반환한다.
  //Mtrx.size()는 세로로 원소가 몇개인지,
  //Mtrx[0].size()는 가로로 원소가 몇개인지를 반환하게 된다.
  /*예)
  1 2 3
  4 5 6
  인 행렬의 사이즈 {2,3}
  */

    return {Mtrx.size(), Mtrx[0].size()};
}


Matrix Matrix::GetInverse() const
{

    Matrix matrix;//자신과 똑같은 행렬 선언
    matrix.resize(this->size().second, this->size().first);//자신의 사이즈와 순서를 반대로 하여 resize시킨다.

    for(size_t i=0; i < this->size().first; ++i)
    {
        for(size_t j=0; j < this->size().second; j++)
        {
            matrix[j][i] = (*this)[i][j];//자신과 순서 반대로 하여 대입 i,j의 위치 차이를 주목하면 됨
        }
    }
    return matrix;
}


void Matrix::Inverse()//위와 같으므로 생략
{
    Matrix tmp = *this;
    this->resize(tmp.size().second, tmp.size().first);
    for(size_t i=0; i < tmp.size().first; ++i)
    {
        for(size_t j=0; j < tmp.size().second; j++)
        {
            (*this)[j][i] = tmp[i][j];
        }
    }
}

double Matrix::Max() const//가장 큰 원소 반환, 어려운건 아니므로 설명 생략한다.
{
    double Rtrn = (*this)[0][0];//이거 그냥 0으로 초기화 해도 된다.
    for(size_t i=0; i<this->size().first; i++)
    {
        for(size_t j=0; j<this->size().second; j++)
        {
            if(Rtrn < (*this)[i][j])
                Rtrn = (*this)[i][j];
        }
    }
    return Rtrn;
}

//operator Overloading
std::vector<double>& Matrix::operator[](size_t idx)
{
    if(idx>=Mtrx.size())//인덱스 유효여부 확인
    {
        throw "Out of Index";

    }

    return Mtrx[idx];//유효할시 vector<double>로 되어있는 한 행을 반환한다.Matrix[인덱스]는 반환형이 vector<double>인셈, 그렇다면 여기서 Matrix[인덱스1][인덱스2]이면 vector<double>[인덱스2]와 같으므로 double을 반환하게 된다,
                              //결과적으로 [인덱스1]행 [인덱스2]열에 있는 원소를 반환하게 된다.
}


std::vector<double> Matrix::operator[](size_t idx) const//반환형이 일반적인 변수일뿐, 위와 동일하다.
{

    if(idx>=Mtrx.size())
    {
        throw "Out of Index";
    }

    return Mtrx[idx];
}


Matrix Matrix::operator^ (const Matrix &B) const//내적을 구하는 연산자 원래는 비트연산자이지만, 편의를 위해 사용하였다.
{
    Matrix A = *this;//나와같은 행렬 소환

    //다음 코드는
    if(A.size().second != B.size().first)//A의 가로길이가 B의 세로길이와 다를경우
    {
        if((A.size().first == 1 || A.size().second == 1) && (A.size().first == B.size().first))//A의 가로나 세로의 길이가 1이고 A의 세로길이가 B의 세로길이와 같을경우
        {
            //std::cout<<"전치행렬1"<<std::endl
            ;
            A.Inverse();//A를 전치행렬로 만들어 버린다.(내적을 가능하게 하기위해)

            //A.print();
        }else if((A.size().second == 1 || A.size().first == 1) && (A.size().second == B.size().second)){//A의 가로길이가 B의 가로길이와 같을경우
            //std::cout<<"전치행렬2"<<std::endl;
            A.Inverse();//A를 전치행렬로 만들어버리기
        }else
        {
            throw "합성곱을 할수 없습니다.(일치하는 차원의 수가 다릅니다.)";
        }
    }
    Matrix C;
    C.resize(A.size().first, B.size().second);

    for(size_t i=0; i< A.size().first; i++)
    {
        //A의 가로열 선택
        for(size_t j=0; j< B.size().second; j++)
        {
            //B의 세로열 선택
            for(size_t l=0; l<B.size().first; l++)
            {
                C[i][j] += A[i][l] * B[l][j];//더하기 대입
            }
        }
    }

    return C;
}

Matrix& Matrix::operator= (const MATRIX &matrix)//대압 연산자
{
    Mtrx = matrix;
    return *this;
}

Matrix& Matrix::operator= (const std::vector<double> &vector)//벡터용 대입 연산자
{

    this->resize(vector.size(), 1);
    for(size_t i=0; i<vector.size(); i++)
    {
        Mtrx[i][0] = vector[i];
    }

    return *this;
}

Matrix Matrix::operator+(const Matrix &copy) const//더하기
{

    Matrix A = *this;//자기 자신
    Matrix B = copy;//더할 대상

    Matrix C;//결과
    AIMath::BroadCast(&A, &B);//A와B 형태맞추기

    //std::cout<<"계산시작"<<std::endl;
    C.resize(A.size().first, A.size().second);

    for(size_t i=0; i<B.size().first; i++)
    {
        for(size_t j=0; j< B.size().second; j++)
        {
            C[i][j] = A[i][j] + B[i][j];//계산
        }
    }


    return C;
}

Matrix Matrix::operator+(const double &copy) const//double형과 같이하는용
{
    Matrix C;
    C.resize(this->size().first , this->size().second);

    for(size_t i=0; i<this->size().first; i++)
    {
        for(size_t j=0; j<this->size().second; j++)
        {
            C[i][j] = (*this)[i][j] + copy;
        }
    }
    return C;
}

Matrix operator+(const double &copy, const Matrix &Mtrx)//교환법칙의 성립을 위해 작성
{
    Matrix C;
    C.resize(Mtrx.size().first , Mtrx.size().second);

    for(size_t i=0; i<Mtrx.size().first; i++)
    {
        for(size_t j=0; j<Mtrx.size().second; j++)
        {
            C[i][j] = copy + Mtrx[i][j];
        }
    }
    return C;
}
//------------------------------------------------------------------------------
//이하 밑의 연산자 오버로딩들은 동일한 원리이므로 설명을 생략한다.
//------------------------------------------------------------------------------
Matrix operator-(const double &copy, const Matrix &Mtrx)
{
    Matrix C;
    C.resize(Mtrx.size().first , Mtrx.size().second);

    for(size_t i=0; i<Mtrx.size().first; i++)
    {
        for(size_t j=0; j<Mtrx.size().second; j++)
        {
            C[i][j] = copy - Mtrx[i][j];
        }
    }
    return C;
}

Matrix Matrix::operator-(const Matrix &copy) const
{
    Matrix A = *this;
    Matrix B = copy;

    Matrix C;
    AIMath::BroadCast(&A, &B);
    //cout<<"계산시작"<<endl;
    C.resize(A.size().first, A.size().second);
    for(size_t i=0; i<B.size().first; i++)
    {
        for(size_t j=0; j< B.size().second; j++)
        {
            C[i][j] = A[i][j] - B[i][j];
        }
    }


    return C;
}

Matrix Matrix::operator-(const double &copy) const
{
    Matrix C;
    C.resize(this->size().first , this->size().second);

    for(size_t i=0; i<this->size().first; i++)
    {
        for(size_t j=0; j<this->size().second; j++)
        {
            C[i][j] = (*this)[i][j] - copy;
        }
    }
    return C;
}


Matrix Matrix::operator*(const Matrix &copy) const
{
    Matrix A = *this;
    Matrix B = copy;

    Matrix C;
    AIMath::BroadCast(&A, &B);
    //cout<<"계산시작"<<endl;
    C.resize(A.size().first, A.size().second);
    for(size_t i=0; i<B.size().first; i++)
    {
        for(size_t j=0; j< B.size().second; j++)
        {
            C[i][j] = A[i][j] * B[i][j];
        }
    }


    return C;
}
Matrix Matrix::operator*(const double &copy) const
{
    Matrix C;
    C.resize(this->size().first , this->size().second);

    for(size_t i=0; i<this->size().first; i++)
    {
        for(size_t j=0; j<this->size().second; j++)
        {
            C[i][j] = (*this)[i][j] * copy;
        }
    }
    return C;
}
Matrix operator*(const double &copy, const Matrix &Mtrx)
{
    Matrix C;
    C.resize(Mtrx.size().first , Mtrx.size().second);

    for(size_t i=0; i<Mtrx.size().first; i++)
    {
        for(size_t j=0; j<Mtrx.size().second; j++)
        {
            C[i][j] = copy * Mtrx[i][j];
        }
    }
    return C;
}
Matrix Matrix::operator/(const Matrix &copy) const
{
    Matrix A = *this;
    Matrix B = copy;

    Matrix C;
    AIMath::BroadCast(&A, &B);
    //cout<<"계산시작"<<endl;
    C.resize(A.size().first, A.size().second);
    for(size_t i=0; i<B.size().first; i++)
    {
        for(size_t j=0; j< B.size().second; j++)
        {
            C[i][j] = A[i][j] / B[i][j];
        }
    }


    return C;
}
Matrix Matrix::operator/(const double &copy) const
{
    Matrix C;
    C.resize(this->size().first , this->size().second);

    for(size_t i=0; i<this->size().first; i++)
    {
        for(size_t j=0; j<this->size().second; j++)
        {
            C[i][j] = (*this)[i][j] / copy;
        }
    }
    return C;
}
Matrix operator/(const double &copy, const Matrix &Mtrx)
{
    Matrix C;
    C.resize(Mtrx.size().first , Mtrx.size().second);

    for(size_t i=0; i<Mtrx.size().first; i++)
    {
        for(size_t j=0; j<Mtrx.size().second; j++)
        {
            C[i][j] = copy / Mtrx[i][j];
        }
    }
    return C;
}


Matrix AIMath::ActivationFunctions::Sigmoid(const Matrix &X)//시그모이드 함수
{
    return 1 / (1 + exp(-1 * X));//행렬끼리의 연산이다.
}


double Matrix::GetSum() const//총합 구하기
{
    double rtrn = 0;
    for(size_t i=0; i<this->size().first; i++)
    {
        for(size_t j=0; j<this->size().second; j++)
        {
            rtrn += (*this)[i][j];
        }
    }
    return rtrn;
}

//증감연산자들, 마찬가지로 너무 쉬우니 패스
Matrix& Matrix::operator-=(const Matrix &mtrx) {
    *this = *this - mtrx;
    return *this;
}

Matrix &Matrix::operator+=(const Matrix &mtrx) {
    *this = *this + mtrx;
    return *this;
}

Matrix &Matrix::operator*=(const Matrix &mtrx) {
    *this = *this * mtrx;
    return *this;
}

Matrix &Matrix::operator/=(const Matrix &mtrx) {
    *this = *this / mtrx;
    return *this;
}
//모양 따라서 resize하는거
void Matrix::resize(const Matrix &target) {
    resize(target.size().first, target.size().second);
}

void Matrix::resize(const Matrix &target, double value) {//이것도 마찬가지
    resize(target.size().first, target.size().second, value);
}

double Matrix::GetAverage() const {//평균 구하기
    return GetSum() / (size().first * size().second);
}

Matrix Matrix::GetAbs() const {//절댓값 구하기

    Matrix Rtrn = (*this);
    for (int i = 0; i < size().first; ++i) {
        for (int j = 0; j < size().second; ++j) {
            if(Rtrn[i][j] < 0)
                Rtrn[i][j] = -1 * Rtrn[i][j];
        }
    }

    return Rtrn;
}



double AIMath::ErrorFunctions::MSE(const Matrix &y, const Matrix &t)//MSE손실함수이다.(y-t)**2의 원소별 평균을 구한다. (**는 제곱연산)
{
    return (Pow(y - t, 2).GetAverage());
}
double AIMath::ErrorFunctions::ACE(const Matrix &y, const Matrix &t)//ACE손실함수 CEE라고도 부른다. 사실 CEE이다.이름이 햇갈려버림....
{//Cross Entropy Error function
    const double delta = 1e-7;
    t.print();
    return -1 * (t * Log(y + delta)).GetSum();
}


Matrix AIMath::Log(const Matrix &x)//로그
{
    Matrix Rtrn;
    Rtrn.resize(x.size().first, x.size().second);
    for(size_t i=0; i<x.size().first; i++)
    {
        for(size_t j=0; j<x.size().second; j++)
        {
            Rtrn[i][j] = log(x[i][j]);
        }
    }
    return Rtrn;

}

Matrix AIMath::Pow(const Matrix &x, double count)//count제곱 구하기
{
    Matrix Rtrn;
    Rtrn.resize(x.size().first, x.size().second);
    for (int i = 0; i < x.size().first; ++i) {
        for (int j = 0; j < x.size().second; ++j) {
            Rtrn[i][j] = pow(x[i][j], count);
        }
    }
    return Rtrn;
}

Matrix AIMath::exp(const Matrix &x)//exp구하기
{//exp = e**x
    Matrix Rtrn;
    Rtrn.resize(x.size().first, x.size().second);
    for(size_t i=0; i<x.size().first; i++)
    {
        for(size_t j=0; j<x.size().second; j++)
        {
            Rtrn[i][j] = std::exp(x[i][j]);
        }
    }
    return Rtrn;
}

Matrix AIMath::ActivationFunctions::SoftMax(const Matrix &x)//소프트맥스 구하기
{
    double max = x.Max();
    Matrix exp_x = exp(x - max); // 오버플로 대책, 이렇게 값을 수정해도 결과에는 변함이 없다. double이 수용할수 있는 범위를 넘기지 않게 하기위해 쓴다.
    double sum_exp_x = exp_x.GetSum();
    return exp_x / sum_exp_x;
}

Matrix AIMath::ActivationFunctions::StepFunc(const Matrix &X) {//계단함수
    Matrix Rtrn;
    Rtrn.resize(X.size().first, X.size().second);

    for (int i = 0; i < X.size().first; ++i) {
        for (int j = 0; j < X.size().second; ++j) {
            if(X[i][j] >= 0)
            {
                Rtrn[i][j] = 1;
            }else{
                Rtrn[i][j] = 0;
            }

        }
    }
    return Rtrn;
}

Matrix AIMath::ActivationFunctions::ReLU(const Matrix &X) {//ReLU함수
    Matrix Rtrn;
    Rtrn.resize(X.size().first, X.size().second);
    for (int i = 0; i < X.size().first; ++i) {
        for (int j = 0; j < X.size().second; ++j) {
            if(X[i][j] > 0)
            {
                Rtrn[i][j] = X[i][j];
            }else{
                Rtrn[i][j] = 0;
            }

        }
    }
    return Rtrn;
}

Matrix AIMath::GetRandomNums(size_t rows, size_t cols, double start, double end)//랜덤한 수들을 반환한다.
{
    Matrix Rtrn;
    std::mt19937 engine((unsigned int)time(NULL));
    std::uniform_real_distribution<double> dist(start, end);
    auto generater = std::bind(dist, engine);

    Rtrn.resize(rows, cols);
    for(size_t i=0; i<rows; i++){
        for(size_t j=0; j<cols; j++){
            Rtrn[i][j] = generater();
        }

    }

    return Rtrn;
}

void AIMath::BroadCast(Matrix *A, Matrix *B) {
    //어렵지 않은 코드이니 설명은 생략한다.
    Matrix a,b;
    if(A->size() != B->size())
    {

        if(A->size().first==B->size().first && B->size().second==1) //broadCasting
        {
            //cout<<"1-1!"<<endl;
            a = *A;
            b.resize(a.size().first, a.size().second);
            for(size_t i=0; i<a.size().first; i++)
            {
                for(size_t j=0; j<a.size().second; j++)
                {
                    b[i][j] = (*B)[i][0];
                }
            }

        }else if(A->size().first==B->size().first && A->size().second==1)
        {
            //cout<<"1-2!"<<endl;
            b = (*B);
            a.resize(b.size().first, b.size().second);
            for(size_t i=0;i<b.size().first; i++)
            {
                for(size_t j=0; j<b.size().second; j++)
                {
                    a[i][j] = (*A)[i][0];
                }
            }
        }else if(A->size().second==B->size().second && B->size().first==1)
        {
            //cout<<"2-1!"<<endl;
            a = *A;
            b.resize(a.size().first, a.size().second);
            for(size_t i=0;i<a.size().first;i++)
            {
                for(size_t j=0;j<a.size().second;j++)
                {
                    b[i][j] = (*B)[0][j];
                }
            }
        }else if(A->size().second==B->size().second && A->size().first==1)
        {
            //cout<<"2-2"<<endl;
            b = (*B);
            a.resize(b.size().first, b.size().second);
            for(size_t i=0;i<b.size().first;i++)
            {
                for(size_t j=0;j<b.size().second;j++)
                {
                    a[i][j] = (*A)[0][j];
                }
            }
        }else if(A->size().second==1 && B->size().first==1)
        {
            //cout<<"3-1"<<endl;
            a.resize(A->size().first, B->size().second);
            b.resize(a.size().first, a.size().second);

            for(size_t i=0; i<a.size().first; i++)
            {
                for(size_t j=0; j<a.size().second; j++)
                {
                    a[i][j] = (*A)[i][0];
                }
            }

            for(size_t i=0;i<b.size().first;i++)
            {
                for(size_t j=0;j<b.size().second;j++)
                {
                    b[i][j] = (*B)[0][j];
                }
            }

        }else if(A->size().first==1 && B->size().second==1)
        {
            //cout<<"3-2"<<endl;
            a.resize(B->size().first, A->size().second);
            b.resize(a.size().first, a.size().second);

            for(size_t i=0; i<a.size().first; i++)
            {
                for(size_t j=0; j<a.size().second; j++)
                {
                    a[i][j] = (*A)[0][j];
                }
            }

            for(size_t i=0;i<b.size().first;i++)
            {
                for(size_t j=0;j<b.size().second;j++)
                {
                    b[i][j] = (*B)[i][0];
                }
            }

        }else if(A->size().first ==1  && A->size().second==1)
        {
            a.resize(B->size().first, B->size().second, (*A)[0][0]);
            b = *B;
        }else if(B->size().first==1 && B->size().second == 1)
        {
            a = *A;
            b.resize(A->size().first, A->size().second, (*B)[0][0]);
        }
        else{
            throw "행렬의 형태가 일치하지 않습니다.";
        }


    }else{
        return;
        //cout<<"일치"<<endl;
    }

    *A = a;
    *B = b;
}

Matrix AIMath::NumericalGradient(std::function<double(Matrix)> f, Matrix &x) //x = 가중치
{
    double h = 1e-4;
    Matrix grad;
    grad.resize(x.size().first, x.size().second);
    //x.print();
    for (int i = 0; i < x.size().first; ++i) {
        for (int j = 0; j < x.size().second; ++j) {
            double tmp_val = x[i][j];
            x[i][j] = tmp_val + h;

            double fxh1 = f(x);
            x[i][j] = tmp_val - h;

            double fxh2 = f(x);
            //cout<<x[i][j]<<endl;

            //cout<< "fxh1 : " <<fxh1<<endl;
            //cout<<"fxh2 : "<<fxh2<<endl;

            grad[i][j] = (fxh1 - fxh2) / (2*h);//미분의 정의에 따라 계산한다. h는 아주 작은 값

            //cout<<"grad: " <<grad[i][j]<<endl;

            x[i][j] = tmp_val;
        }
    }
    return grad;
}
//이하 설명 생략
Matrix AIMath::sqrt(const Matrix &x) {
    Matrix Rtrn;
    Rtrn.resize(x.size().first, x.size().second);
    for(size_t i=0; i<x.size().first; i++)
    {
        for(size_t j=0; j<x.size().second; j++)
        {
            Rtrn[i][j] = std::sqrt(x[i][j]);
        }
    }
    return Rtrn;
}

Matrix AIMath::AbsoluteValue(const Matrix &x) {
    Matrix Rtrn;
    Rtrn.resize(x.size().first, x.size().second);
    for(size_t i=0; i<x.size().first; i++)
    {
        for(size_t j=0; j<x.size().second; j++)
        {
            if(x[i][j] < 0)
                Rtrn[i][j] = -1 * x[i][j];
            else
                Rtrn[i][j] = x[i][j];
        }
    }
    return Rtrn;
}
