import cloudpickle
import gym
from gym import error, spaces
from gym.utils import seeding
import itertools
from collections import deque
import numpy as np



class BeerGame(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, n_agents: int, env_type: str, n_turns_per_game=30, seed=None):
        super().__init__()
        self.n_agents = n_agents
        self.orders = [] # 주문하는 양
        self.shipments = [] # 배송하는 양
        self.stocks = [10] * self.n_agents # 현 재고
        self.holding_cost = np.zeros(self.n_agents, dtype=np.float)
        self.stockout_cost = np.zeros(self.n_agents, dtype=np.float)
        self.cum_holding_cost = np.zeros(self.n_agents, dtype=np.float)
        self.cum_stockout_cost = np.zeros(self.n_agents, dtype=np.float)
        self.score_weight = [[0.5] * self.n_agents, [1] * self.n_agents]
        self.turns = None
        self.turn = 0
        self.done = False
        self.np_random = None

        self.env_type = env_type
        if self.env_type not in ['classical', 'uniform_0_2', 'normal_10_4']:
            raise NotImplementedError("env_type must be in ['classical', 'uniform_0_2', 'normal_10_4']")

        self.n_turns = n_turns_per_game
        self.seed(seed)

        # TODO calculate state shape
        # self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)

    def _get_observations(self):
        observations = [None] * self.n_agents
        for i in range(self.n_agents):
            observations[i] = {'current_IP': self.stocks[i]}
        return observations


    def _get_rewards(self):
        return -(self.holding_cost + self.stockout_cost)

    def _get_demand(self):
        return np.random.poisson(7)
#        return self.turns[self.turn]

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.done = False
        self.stocks = [50] * self.n_agents # 현 재고

        # 모든 agent의 outbound orders를 초기화 함
        # retailer(0), wholesaler(1), distributor(2)는 upstream으로 [t-2, t-1] 시점에 주문하였으며, 각 10으로 초기화 함
        # manufacturer(3)는 t-1 시점에만 주문 내역이 있으며 10으로 함
        # t시점 주문량(action)은 추후 deque에 append 됨
        temp_orders_out = [[10, 10]] * (self.n_agents - 1) + [[10]]
        self.orders_out = [deque(x) for x in temp_orders_out]

        # 모든 agent outbound shipments를 초기화 함
        # retailer는 outbound shipments에 대한 리드타임이 존재하지 않음, 주문오는 족족 재고에서 처리됨
        # wholesaler(1), distributor(2), manufacturer(3), 그리고 게임의 외부참여자인 supplier의 [t-2, t-1]시점 배송량을 각 10으로 초기화 함
        # [주의] shipments_out의 경우, 여느 때와 달리 index 0은 wholesaler이며, index 3은 supplier임 !
        temp_shipments_out = [[10, 10]] * self.n_agents
        self.shipments_out = [deque(x) for x in temp_shipments_out]


        # initialize other variables
        self.holding_cost = np.zeros(self.n_agents, dtype=np.float)
        self.stockout_cost = np.zeros(self.n_agents, dtype=np.float)
        self.cum_holding_cost = np.zeros(self.n_agents, dtype=np.float)
        self.cum_stockout_cost = np.zeros(self.n_agents, dtype=np.float)
        self.turn = 0

        # 한번의 transition을 수행한 상태를 초기 상태로 설정함
        self.transition(action=[np.random.poisson(7) for i in range(4)])

        return self._get_observations()


    def render(self, mode='human'):
        if mode != 'human':
            raise NotImplementedError(f'Render mode {mode} is not implemented yet')
        index_dict = {0 : 'RETAILER', 1 : 'WHOLESALER', 2 : 'DISTRIBUTOR', 3 : 'MANUFACTURER' , 4 : 'SUPPLIER'}


        print('-'*30 + 'TURN {:02}'.format(self.turn) + '-'*30)
        for i in range(len(self.orders_in)) :
            print('[{:^15}]'.format(index_dict[i]) +'의 기초재고(beginning stocks) 상태 : {}'.format(self.beginning_stocks[i]))
        print('.')
        for i in range(len(self.orders_in)) :
            print('     [{:^15}]'.format(index_dict[i]) +'의 주문수령(inbound orders) 상태 : {}'.format(self.orders_in[i]))
        print('.')
        for i in range(len(self.orders_out)) :
            print('     [{:^15}]'.format(index_dict[i]) +'의 주문요청(outbound orders) 상태 : {}'.format(self.orders_out[i]))
        print('.')
        for i in range(len(self.shipments_in)) :
            print('     [{:^15}]'.format(index_dict[i]) +'의 배송수령(inbound shipments) 상태 : {}'.format(self.shipments_in[i]))
        print('.')
        print('     [{:^15}]'.format(index_dict[0]) +'의 배송발송(outbound shipments) 상태 : 해당 사항 없음')
        for i in range(len(self.shipments_out)) :
            print('     [{:^15}]'.format(index_dict[i+1]) +'의 배송발송(outbound shipments) 상태 : {}'.format(self.shipments_out[i]))
        print('.')
        for i in range(len(self.orders_in)) :
            print('[{:^15}]'.format(index_dict[i]) +'의 기말재고(ending stocks) 상태 : {}'.format(self.stocks[i]))
        print()


    def transition(self, action: list):

        self.beginning_stocks = self.stocks[:]

        # sanity checks
        if self.done:
            raise error.ResetNeeded('Environment is finished, please run env.reset() before taking actions')
        if any(np.array(action) < 0):
            raise error.InvalidAction(f"You can't order negative amount. You agents actions are: {action}")


        # 1) t시점에 각 agent는 downstream으로부터 주문(inbound order)을 받음
        # retailer는 inbound order에 대한 리드타임이 존재하지 않음, 단, 현장에서 확률적 수요가 발생함
        retailer_demand = self._get_demand()
        # wholesaler ~ manufacturer는 t시점에 downstream이 t-2 시점에 주문했던 사항을 주문 받음
        orders_t_2 = [order.popleft() for order in self.orders_out[:self.n_agents - 1]]
        self.orders_in = [retailer_demand] + orders_t_2
        # 게임의 외부 참여자인 manufacturer의 supplier도 주문을 받음
        supplier_demand = self.orders_out[-1].popleft()



        # 2) t시점에 각 agent는 upstream으로부터 inbound shipments(shipments_in)를 수령함
        # retailer(0), wholesaler(1), distributor(2)는 upstream이 t-2 시점에 '배송해주었던 배송량'을 배송받음
        # manufacturer(3)는 game의 참가자가 아닌 외부 supplier가 t-2시점에 '배송해주었던 배송량'을 배송받음
        self.shipments_in = [shipment.popleft() for shipment in self.shipments_out]
        self.stocks = [(stock + inc) for stock, inc in zip(self.stocks, self.shipments_in)] # 기보유재고 + 현시점 받은 배송품 수



        # 3) t시점에 각 agent는 t+2 시점에 downstream으로 보낼 배송량을 배송(queue에 append 함)하고 재고를 update함
        # 먼저, retailer는 downstream으로 배송을 하는 것이 아니므로 수요만큼 재고가 감소됨
        self.stocks[0] -= retailer_demand # 주문 미충족 시 백오더(음수 재고) 발생
        # wholesaler(1), distributor(2), manufacturer(3)의 shipment를 배송함
        for i in range(1, self.n_agents) :
            max_possible_shipment = max(0, self.stocks[i])   # 현재 보유중인 재고
            order = self.orders_in[i] + max(0, -self.stocks[i]) # 현재 들어온 주문과 백오더 (총 배송해주어야 하는 양)
            this_step_shipment = min(order, max_possible_shipment) # 총 배송해주어야 하는 양과 현실적으로 배송 가능한 양 비교
            self.shipments_out[i-1].append(this_step_shipment)
            self.stocks[i] -= self.orders_in[i] # 주문 미충족 시 백오더(음수 재고) 발생
        # 외부 참여자인 supplier의 경우, t+2 시점에는 현재 주문량 만큼을 100% manufacturer에게 공급할 수 있음
        self.shipments_out[-1].append(supplier_demand)



        # 4) t시점에 각 agent는 upstream에게 주문(outbound order)을 함
        for i in range(self.n_agents) :
            self.orders_out[i].append(action[i])

    def step(self, action: list):

        self.transition(action)

        # calculate costs
        self.holding_cost = np.zeros(self.n_agents, dtype=np.float)
        self.stockout_cost = np.zeros(self.n_agents, dtype=np.float)
        for i in range(self.n_agents):
            if self.stocks[i] >= 0:
                self.holding_cost[i] = self.stocks[i] * self.score_weight[0][i]
            else:
                self.stockout_cost[i] = -self.stocks[i] * self.score_weight[1][i]
        self.cum_holding_cost += self.holding_cost
        self.cum_stockout_cost += self.stockout_cost


        # calculate reward
        rewards = self._get_rewards()

        # check if done
        if self.turn == self.n_turns - 1:
            self.turn += 1
            self.done = True
        else:
            self.turn += 1
        state = self._get_observations()

        return state, rewards, self.done, {}
