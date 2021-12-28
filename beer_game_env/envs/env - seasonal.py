import cloudpickle
import gym
from gym import error, spaces
import itertools
from collections import deque
import numpy as np



class BeerGame(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, n_agents: int, env_type: str, BS_level: [95,90,95,74], n_turns_per_game=30):
        super().__init__()
        self.env_type = env_type
        # print("demand type >>> ", self.env_type)

        self.n_agents = n_agents
        self.orders = [] # 주문하는 양
        self.shipments = [] # 배송하는 양
        self.stocks = [50] * self.n_agents # 현 재고
        self.holding_cost = np.zeros(self.n_agents, dtype=np.float)
        self.stockout_cost = np.zeros(self.n_agents, dtype=np.float)
        # self.score_weight = [[0.5] * self.n_agents, [2] * self.n_agents]
        self.score_weight = [[0.5] * self.n_agents, [10, 0.1, 0.1, 0.1]] # Stock Out Cost가 Retailer에서만 발생하는 경우

        self.BS_level = BS_level
        self.turns = None
        self.turn = 0
        self.done = False

        np.random.seed(0)

        if self.env_type == 'seasonal_beer' :
            ratio = [1, 1.04, 1.34, 1.4, 1.76, 1.9, 1.61, 1.95, 1.94, 1.73, 1.62, 1.74, 1]
            data = []
            for i in range(1, len(ratio)):
                data += np.linspace(ratio[i-1],ratio[i],3).tolist()
            data_ = []
            for _ in range(20):
                data_ += [np.round(10*(val)+np.random.randint(2)) for val in data]
            self.seasonal_demand = deque(data_)

        elif self.env_type == 'sinusoidal' :
            seasonal_d = lambda d_max,d_var,t : np.floor(d_max/2 + d_max/2*np.sin(2*np.pi*(t + 2)/100*2) + np.random.randint(5, d_var))
            self.seasonal_demand = deque(seasonal_d(10,10,np.arange(100)))


        self.trans_hist = deque() # transition history를 기록하여 state vector를 만듦
        self.env_type = env_type
        if self.env_type not in ['poisson', 'sinusoidal', 'seasonal_beer']:
            raise NotImplementedError("env_type must be in ['poisson', 'sinusoidal', 'seasonal_beer']")

        self.n_turns = n_turns_per_game

         # TODO calculate state shape
        # self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)

    def calculate_state(self) :
#        IL = self.beginning_stocks # Beggining Inventory Level
        IL_stock = [max(0, each) for each in self.beginning_stocks]
        IL_stockout = [max(0, -each) for each in self.beginning_stocks]

        # On-order를 계산함
        # On-order = 주문한 내역(orders out) + 배송 오고 있는 내역(shipments out)
        orders_out_sum = [sum(each) for each in self.orders_out]
        shipments_out_sum = [sum(each) for each in self.shipments_out]
        OO = [sum(each) for each in zip(orders_out_sum, shipments_out_sum)] # On-order

        AO = self.orders_in # Arriving orders
        AS = self.shipments_in # Arriving Shipments

        return IL_stock, IL_stockout, OO, AO, AS

    def _get_BS_policy(self) :

        return self.BS_policy

    def _get_observations(self):

        IL_stock, IL_stockout, OO, AO, AS = self.calculate_state()
        state_vec = [] # t시점 각 agent의 상태 벡터 [IL OO AO AS]를 concatenate
        for agent in [list(each) for each in zip(*[IL_stock, IL_stockout, OO, AO, AS])] :
            state_vec += agent

        # BS Policy
        IP = [sum((state_vec[0], state_vec[1], state_vec[2])), sum((state_vec[5], state_vec[6], state_vec[7])), sum((state_vec[10], state_vec[11], state_vec[12])), sum((state_vec[15], state_vec[16], state_vec[17]))]
        # print("IP >>> ", IP)
        # print("AO >>> ", AO)
        # print("IP - AO >>> ", np.subtract(IP,AO))
        # print()
        self.BS_policy = list(map(max, zip([0]*4, np.subtract(self.BS_level, np.subtract(IP, AO)))))

        return state_vec


    def _get_rewards(self):
        return -(self.holding_cost + self.stockout_cost)

    def _get_demand(self):
        if self.env_type == 'normal' :
            return np.round(np.random.normal(10,2)).astype(int)
        elif self.env_type == "poisson" :
            return np.random.poisson(6)
        else :
            d = self.seasonal_demand.popleft()
            self.seasonal_demand.append(d)
            return d


    def reset(self):
        self.done = False
        self.stocks = [18] * self.n_agents # 현 재고

        # 모든 agent의 outbound orders를 초기화 함
        # retailer(0), wholesaler(1), distributor(2)는 upstream으로 [t-2, t-1] 시점에 주문하였으며, 각 10으로 초기화 함
        # manufacturer(3)는 t-1 시점에만 주문 내역이 있으며 10으로 함
        # t시점 주문량(action)은 추후 deque에 append 됨
        temp_orders_out = [[18, 18]] * (self.n_agents - 1) + [[18]]
        self.orders_out = [deque(x) for x in temp_orders_out]

        # 모든 agent outbound shipments를 초기화 함
        # retailer는 outbound shipments에 대한 리드타임이 존재하지 않음, 주문오는 족족 재고에서 처리됨
        # wholesaler(1), distributor(2), manufacturer(3), 그리고 게임의 외부참여자인 supplier의 [t-2, t-1]시점 배송량을 각 10으로 초기화 함
        # [주의] shipments_out의 경우, 여느 때와 달리 index 0은 wholesaler이며, index 3은 supplier임 !
        temp_shipments_out = [[18, 18]] * self.n_agents
        self.shipments_out = [deque(x) for x in temp_shipments_out]


        # initialize other variables
        self.holding_cost = np.zeros(self.n_agents, dtype=np.float)
        self.stockout_cost = np.zeros(self.n_agents, dtype=np.float)
        self.turn = 0
        self.trans_hist.clear()

        # 한번의 transition을 수행한 상태를 초기 상태로 설정함
        for _ in range(10) : # ★ 에이전트에 따라 바뀜
            self.current_action = [0] * 4 # d+x 룰에서 주문량만큼만 action을 취함. 즉 x = 0 으로 시작함
            # self.current_action = [10]*4 # 모두가 d+x가 아닌 경우
            # self.current_action = [0, 10, 10, 10] # Retailer만 d+x 룰을 적용함
            # self.current_action = [10, 0, 10, 10] # Wholesaler만 d+x 룰을 적용함
            # self.current_action = [10, 10, 0, 10] # Distributor만 d+x 룰을 적용함
            # self.current_action = [10, 10, 10, 0] # Manufacturer만 d+x 룰을 적용함

            self.transition(self.current_action, reset=True)
            self.trans_hist.append(self._get_observations())

        return self.trans_hist



    def render(self, episode, path, mode='human'):
        if mode != 'human':
            raise NotImplementedError(f'Render mode {mode} is not implemented yet')
        index_dict = {0 : 'RETAILER', 1 : 'WHOLESALER', 2 : 'DISTRIBUTOR', 3 : 'MANUFACTURER' , 4 : 'SUPPLIER'}

        with open(path, 'a') as f :
            f.write('-'*30 + 'EPISODE {:05} TURN {:02}'.format(episode, self.turn) + '-'*30+'\n')
            for i in range(len(self.orders_in)) :
                f.write('[{:^15}]'.format(index_dict[i]) +'의 기초재고(beginning stocks) 상태 : {}\n'.format(self.beginning_stocks[i]))

            f.write('.\n')
            for i in range(len(self.shipments_in)) :
                f.write('     [{:^15}]'.format(index_dict[i]) +'의 배송수령(inbound shipments) 상태 : {}\n'.format(self.shipments_in[i]))

            f.write('.\n')
            for i in range(len(self.orders_in)) :
                f.write('     [{:^15}]'.format(index_dict[i]) +'의 주문수령(inbound orders) 상태 : {}\n'.format(self.orders_in[i]))

            f.write('.\n')
            for i in range(len(self.current_action)) :
                f.write('     [{:^15}]'.format(index_dict[i]) +'의 현 시점 행동(current action) 상태 : {}\n'.format(self.current_action[i]))

            f.write('.\n')
            for i in range(len(self.orders_out)) :
                f.write('     [{:^15}]'.format(index_dict[i]) +'의 주문요청(outbound orders) 상태 : {}\n'.format(self.orders_out[i]))

            f.write('.\n')
            f.write('     [{:^15}]'.format(index_dict[0]) +'의 배송발송(outbound shipments) 상태 : 해당 사항 없음\n')
            for i in range(len(self.shipments_out)) :
                f.write('     [{:^15}]'.format(index_dict[i+1]) +'의 배송발송(outbound shipments) 상태 : {}\n'.format(self.shipments_out[i]))

            f.write('.\n')
            for i in range(len(self.orders_in)) :
                f.write('[{:^15}]'.format(index_dict[i]) +'의 기말재고(ending stocks) 상태 : {}\n'.format(self.stocks[i]))

            f.write('\n\n\n')


    def transition(self, action: list, reset=False):

        self.beginning_stocks = self.stocks[:]

        # sanity checks
        if self.done:
            raise error.ResetNeeded('Environment is finished, please run env.reset() before taking actions')

        # 1) t시점에 각 agent는 downstream으로부터 주문(inbound order)을 받음
        # retailer는 inbound order에 대한 리드타임이 존재하지 않음, 단, 현장에서 확률적 수요가 발생함
        if reset == True :
            retailer_demand = 18
        else :
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
        existing_stocks = [(max(0, stock) + inc) for stock, inc in zip(self.stocks, self.shipments_in)] # 기보유 + 현시점 받은 배송 결과의 실재고



        # 3) t시점에 각 agent는 t+2 시점에 downstream이 받을 배송량을 배송(queue에 append 함)하고 재고를 update함
        # 먼저, retailer는 downstream으로 배송을 하는 것이 아니므로 수요만큼 재고가 감소됨
        self.stocks[0] = existing_stocks[0] - (self.orders_in[0] + max(0, -self.stocks[0])) # 주문 미충족 시 백오더(음수 재고) 발생
        # wholesaler(1), distributor(2), manufacturer(3)의 shipment를 배송함
        for i in range(1, self.n_agents) :
            max_possible_shipment = existing_stocks[i] # 현재 물리적으로 보유중인 재고
            order = self.orders_in[i] + max(0, -self.stocks[i]) # 현재 들어온 주문과 백오더 (총 배송해주어야 하는 양)
            this_step_shipment = min(order, max_possible_shipment) # 총 배송해주어야 하는 양과 현실적으로 배송 가능한 양 비교
            self.shipments_out[i-1].append(this_step_shipment)
            self.stocks[i] = existing_stocks[i] - order # 주문 미충족 시 백오더(음수 재고) 발생
        # 외부 참여자인 supplier의 경우, t+2 시점에는 현재 주문량 만큼을 100% manufacturer에게 공급할 수 있음
        self.shipments_out[-1].append(supplier_demand)



        # 4) t시점에 각 agent는 upstream에게 주문(outbound order)을 함
        # 주문량은 d+x rule을 적용하여 주문함
        # ★ 에이전트에 따라 바뀜

        # All DQN-agents인 경우
        action_d_plus_x = [max(0,sum(x)) for x in zip(self.orders_in, action)]
        for i in range(self.n_agents) :
            self.orders_out[i].append(action_d_plus_x[i])

        # # All BS-Policy의 경우
        # for i in range(self.n_agents) :
        #     self.orders_out[i].append(action[i])

        # # Retailer만 DQN-agent인 경우
        # for i in range(self.n_agents) :
        #     self.orders_out[i].append(action[i])
        # self.orders_out[0][-1] += self.orders_in[0] # retailer만 신경망의 아웃풋에 수요량만큼을 더해줌
        # #
        # # Wholesaler만 DQN-agent인 경우
        # for i in range(self.n_agents) :
        #     self.orders_out[i].append(action[i])
        # self.orders_out[1][-1] += self.orders_in[1] # wholesaler만 신경망의 아웃풋에 수요량만큼을 더해줌
        #
        # # Distributor만 DQN-agent인 경우
        # for i in range(self.n_agents) :
        #     self.orders_out[i].append(action[i])
        # self.orders_out[2][-1] += self.orders_in[2] # distributor만 신경망의 아웃풋에 수요량만큼을 더해줌
        # #
        # # Manufacturer만 DQN-agent인 경우
        # for i in range(self.n_agents) :
        #     self.orders_out[i].append(action[i])
        # self.orders_out[3][-1] += self.orders_in[3] # distributor만 신경망의 아웃풋에 수요량만큼을 더해줌


    def step(self, action: list):
        self.trans_hist.popleft()
        self.current_action = action
        self.transition(action)
        self.trans_hist.append(self._get_observations())

        # calculate costs
        self.holding_cost = np.zeros(self.n_agents, dtype=np.float)
        self.stockout_cost = np.zeros(self.n_agents, dtype=np.float)
        for i in range(self.n_agents):
            if self.stocks[i] >= 0:
                self.holding_cost[i] = self.stocks[i] * self.score_weight[0][i]
            else:
                self.stockout_cost[i] = -self.stocks[i] * self.score_weight[1][i]

        # calculate reward
        rewards = self._get_rewards()

        # check if done
        if self.turn == self.n_turns - 1:
            self.turn += 1
            self.done = True
        else:
            self.turn += 1
        state = self.trans_hist

        return state, rewards, self.done, (self.beginning_stocks, self.orders_in, self.orders_out)
