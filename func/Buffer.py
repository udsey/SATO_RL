class Buffer():

    def __init__(self, buffer_size):

        self.state_buf = deque(maxlen=buffer_size)
        self.next_state_buf = deque(maxlen=buffer_size)
        self.action_buf = deque(maxlen=buffer_size)
        self.reward_buf = deque(maxlen=buffer_size)
        self.done_buf = deque(maxlen=buffer_size)
        self.buffer_size = 0

    def add(self, state, next_state, action, reward, done):

        self.state_buf.append(state)
        self.next_state_buf.append(next_state)
        self.action_buf.append(action)
        self.reward_buf.append(reward)
        self.done_buf.append(done)
        self.buffer_size +=1

    def sample(self, batch_size):
        
        indexes = np.random.randint(self.buffer_size, size=batch_size)
        mb_state = []
        mb_next_state = []
        mb_action = []
        mb_reward = []
        mb_done = []

        for i in indexes:
            mb_state.append(self.state_buf[i])
            mb_next_state.append(self.next_state_buf[i])
            mb_action.append(self.action_buf[i])
            mb_reward.append(self.reward_buf[i])
            mb_done.append(self.done_buf[i])

        return mb_state, mb_next_state, mb_action, mb_reward, mb_done

    def __len__(self):

        return self.buffer_size
