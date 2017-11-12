#include <iostream>
#include "Libraries/Network.h"

using namespace std;

/*
 * 시작하기전에 서론:
 * 일단 일반 뉴런 네트워크(오차역전파법 이용)을 구현한 소스 코드이고, XOR게이트를 학습시키는것이 목표이다.
 *
 * 네트워크의 구조는 다음과 같다.

 Network = Block(AffineLayer + ActivationLayer) + Block(AffineLayer + ActivationLayer) + ...
 Block은 AffineLayer와 ActivationLayer를 하나로 묶어주는 역할을 한다.
 Layer는 층이란 뜻으로, Input -> Layer -> output의 구조를 가진다. 이 순서대로 정보가 흐르는것을 forward propagation(오차역전파법)이라고하고,
 이 행위(정보를 흐르게 하는것)를 feedforward라고 한다.(앞에서 정보를 먹인다는 뜻이다.(뒤에서 싸고) )
 오차 역전파법으로 표현하면 반대로 정보가 흐른다. 이를 back propagation이라고 하고, 이 행위를 feedbackward라고 한다.
 AffineLayer는 (X^W) + b 를 구현한 레이어이다.(^는 행렬의 내적을 구하는 연산자다.(Matrix.hpp참고))
 ActivationLayer는 활성화 함수의 역할을 하는 레이어인데, Sigmoid만 구현되어 있다.(다른건 없다. 쓰면 오류남)

 +여론
 신경망의 문제 해결은 보통 2가지로 나뉜다. 바로 분류(Classification)과 회귀(Regression)이다.
 분류문제는 MNIST문제가 대표적이며 특정한 집합에 소속될 확률 등을 구한다. 마지막 층으로 SoftMaxWithLoss(SoftMax + CrossEntropyErrorFunction)을 사용한다.
 회귀문제는 밑에서 보는것 같이 특정한 값을 찾아간다. 대표적으로 XOR 게이트 학습이 있다. 보통 모든 레이어에 SIgmoid를 입히며, MSE 손실 함수를 사용한다.
 SoftMax 활성화 함수와 CrossEntropyErrorFunction는 회귀문제에 사용될수 없다.
 SoftMax는 출력값을 확률로 표현하는데, 특정한 값을 찾아가야하는 회귀와는 전혀 맞지 않고, CrossEntropyErrorFunction는 SoftMax와 함께 사용해야 제대로 사용할수 있다.(음수입력시 오류남)
 그럼, 소스를 보아보자

 +파일별 해석
 #main.cpp : 메인
 #Libraries/Matrix.hpp(.cpp) : 행렬의 연산과 관리를 만들어놓은 클래스이다. Matrix클래스가 존재한다. Matrix클래스는 행렬 자료형처럼 사용이 가능하다. 자세한 기능들은 헤더파일 참고
                                                  그 외에도 AIMath라는 네임스페이스가 있는데, 이건 인공지능에 사용되는 수리적 함수들을 모아둔 곳이다.
                                                  역시 기능들은 헤더파일 참고
#Libraries/Network.h(.cpp) : 뉴럴 네트워크를 만드는데 필요한 클래스이다. 자세한건 헤더파일 참고.

+include 구조
A->B는 A가 B를 include함을 뜻한다.

main.cpp -> Network.h -> Matrix.h
main.cpp에서는 Network.h와 Matrix.h를 모두 쓸수 있다. 이는 include의 개념이 "파일 끌어와 해당내용 붙여넣기"의 개념이기 때문이다.
사실 이런구조는 거의 안보인다. 처음에 만들때 좀  힘들었다.
*/

int main() {


    //블록 선언법 Block [name](AffineLayer([output_size]), [ActivationFunction])
    Block b1(AffineLayer(5), Sigmoid);//블록 1 선언
    Block b2(AffineLayer(1), Sigmoid);//블록2 선언
    //네트워크 선언법 NeuralNetwork [Name](input_size)
    NeuralNetwork Network(2);//네트워크 선언
    Network<<b1<<b2;//Network에 블록 추가


    int last = 0; //마지막 진행 %
    int num = 0;//학습 횟수

    cout<<"Train횟수 : ";
    cin>>num;
    for (int i = 0; i < num; ++i) {
        if(last != (int)(((double)i / num)*100)) //마지막 진행%가 현재 진행%랑 다를시(정수로 변환시킴)
        {

            cout<<(int)(((double)i / num)*100)<<"%"<<"   loss : "<<Network.Last_loss<<endl; //현재 진행 상황과 손실값이 얼마인지 출력(손실값 : 현재 출력되는 값이 정답과 얼마나 다른지 알려주는 값)
            last = (int)(((double)i / num)*100); //last갱신
        }
        /*다음을 학습시킨다. XOR게이트
        0 0     0
        1 0     1
        0 1     1
        11      0
        Network.Train(s)은 다음과 같은 구조를 가진다.
        void Train(std::pair<Matrix, Matrix> Train_Data, double Rate);
         void Trains(std::vector<std::pair<Matrix, Matrix>> Train_Datas, double Rate);
        vector는 쉽게말해 스마트한 배열이라고 할수 있는데, 자세한 사용은 인터넷에 검색하거나, Matrix.hpp에서 관련 구현에 대해 예제를 볼수 있다.
        pair는 {앞,뒤}형식으로 값이 이동되는데, [pair이름].first로 '앞'값을 [pair이름].last로 '뒤'값을 얻어낼수 있다.
        Rate는 학습률, 한번에 갱신하는 량을 뜻한다. 보통 0.1정도로 하고 층이 더 깊을수록 더 작아진다.
        */

        Network.Trains({
                              {Matrix({0,0}),Matrix({0})},
                              {Matrix({1,0}),Matrix({1})},
                              {Matrix({0,1}),Matrix({1})},
                              {Matrix({1,1}),Matrix({0})}
                      }, 0.1);

    }

    //여기부터 학습이 완료된 후 값을 출력하는 곳이다.

    Matrix x({0,0}); // {0,0}값을 가지는 행렬 선언
    cout<<"0 XOR 0 : ";
    Network.predict(x).print();//predict는 네트워크에다가 값을 대입한후 연산시킨다. 그리고 그 값을 가져온다.(print는 Matrix클래스에 선언되어있다. 해당 헤더파일 참고.) 이하는 같으므로 설명을 생략한다.
    cout<<"1 XOR 0 : ";
    x = {1,0};
    Network.predict(x).print();
    x = {0,1};
    cout<<"0 XOR 1 : ";
    Network.predict(x).print();
    x = {1,1};
    cout<<"1 XOR 1 : ";
    Network.predict(x).print();

    return 0;
}
