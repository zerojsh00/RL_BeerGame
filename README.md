# Multi-agents Beer Game
Beer Game은 공급사슬망 내 에이전트들이 상호 간에 수요, 재고 등의 정보를 공유할 수 없는 부분관찰환경(Partially Observable Environments)으로 구성되어 있으며, 이로 인해 공급사슬망 상류에서 주문량이 과도하게 결정되는 채찍 효과를 경험하도록 설계되어 있습니다. 

본 연구는 Beer Game의 각 에이전트(공장, 물류창고, 도매상, 소매상)들이 부분관찰환경에서 자신의 상태정보(재고량 및 주문량 등)만을 알고 있을 때 어떻게 주문 의사결정을 내려야 전체 공급사슬망의 총비용(재고 유지비용과 재고 부족비용)을 최소화 할 수 있을지를 시뮬레이션 합니다.

본 연구는 공급사슬망 내 모든 에이전트들이 기준재고정책과 같은 전통적인 재고관리기법을 따르지 않고, 강화학습의 DQN 알고리즘을 활용한 의사결정을 할 때 전체 공급사슬망의 총비용이 최소화될 수 있음을 보여줍니다.

## Installation

1. 아나콘다에서 Beer Game을 위한 환경을 설정합니다.
```
conda create python=3.6 --name beer-game-env
source activate beer-game-env
```

2. 레포지터리를 Clone 합니다.
```
git clone https://github.com/zerojsh00/RL_BeerGame
```

3. 루트 레포지터리에서 패키지를 설치합니다.
```
pip install -e .
```

## 참고문헌 
본 연구는 Afshin Oroojlooyjadid의 연구(https://arxiv.org/abs/1708.05924)의 후속연구입니다.
