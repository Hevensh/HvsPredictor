import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import tensorflow as tf

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False

MyEarlyStopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=12,mode='min')

class MyModel(tf.keras.Model):
  def __init__(self):
    super().__init__()
    self.dense = tf.keras.layers.Dense(3, activation="linear")
    self.outputs = tf.keras.layers.Dense(1, name="predictions")

  def call(self, inputs):
    x = self.dense(inputs)
    return self.outputs(x)

class CautiousLoss(tf.keras.losses.Loss):
  def call(self, y_true, y_pred):
    return tf.reduce_mean(tf.math.square(y_pred-y_true)*
                          (2-tf.math.sign(y_pred)*tf.math.sign(y_true)))

class MyPredictor():
    #  用于预测数据并展示
    #  目前仅支持单列数据进行预测

  def __init__(self,real, diff_degree = 1, metrics = 'mse'):
    #  导入真实数据
    self.realData = real
    self.data_len = len(real);
    self.diff_degree = diff_degree
    self.metrics = metrics
    
    self.data0 = np.zeros([self.data_len,diff_degree+1])
    self.trainData = np.zeros([self.data_len-diff_degree])

    self.data0[:,0] = real
    for i in range(diff_degree):
      self.data0[i+1:, i+1] = np.diff(self.data0[i:,i])

    self.trainData = self.data0[diff_degree:, diff_degree]

    self.total_profit = 0
    self.daily_profit = 0
    self.callback_rate = 0


  def deTrend(self,window_len=12,poly_degree=0):
    self.window_len = window_len;
    
    self.sample_len = self.data_len-self.diff_degree-window_len
    self.slicesX = np.zeros([self.sample_len,window_len])
    self.slicesT = np.zeros([self.sample_len,1])
    for i in range(self.sample_len):
      self.slicesX[i,:] = self.trainData[i:i+window_len]
      self.slicesT[i] = self.trainData[i+window_len]
    
    self.window_pattern = np.zeros([poly_degree,window_len])

    for i in range(poly_degree):
      self.window_pattern[i,:] = np.arange(1-window_len,1)**i
    
    moment_estimate_matrix = np.dot(
        self.window_pattern.T,
        np.linalg.inv(np.dot(self.window_pattern,self.window_pattern.T))
        )
    
    self.TrendData = np.dot(self.slicesX,moment_estimate_matrix)
    
    self.deTrendDataX = np.zeros([self.sample_len,window_len])
    self.deTrendDataX = self.slicesX-np.dot(self.TrendData,self.window_pattern)
    
    self.deTrendDataT = np.zeros([self.sample_len,1])
    self.TrendSum = self.TrendData.sum(axis=1)
    self.TrendSum.resize([self.sample_len,1])
    self.deTrendDataT = self.slicesT-self.TrendSum

  def Options_set(self,train_rate=0.6,validate_rate=0.2,loss_func=CautiousLoss()):
    self.train_len = int(train_rate*self.sample_len)
    self.validate_len = int((validate_rate+train_rate)*self.sample_len)
    self.loss_func = loss_func 

  def compile_and_fit(self,model = MyModel(),early_stopping = MyEarlyStopping):
    self.model = model
    self.model.compile(loss=self.loss_func, optimizer='adam')

    self.model.fit(self.deTrendDataX[:self.train_len], 
              self.deTrendDataT[:self.train_len], 
              epochs = 150, batch_size=10, 
              callbacks=[early_stopping], 
              validation_data=(
                  self.deTrendDataX[self.train_len:self.validate_len], 
                  self.deTrendDataT[self.train_len:self.validate_len])
              )
    
    self.T_pre = np.array(model(self.deTrendDataX))
    T_pre = np.array(model(self.deTrendDataX))
    T_pre.resize([len(T_pre)])
    T_pre += self.TrendData.sum(axis=1)
    
    for i in range(self.diff_degree):
        T_pre += self.data0[self.window_len+self.diff_degree-1:-1,self.diff_degree-1-i]
    self.predData = T_pre
    self.predData.resize([len(self.predData)])

  def result_plot(self,start_from = [], end = []):
    if start_from == []:
      start_from = self.validate_len
    elif start_from < self.window_len+self.diff_degree:
      start_from = self.window_len+self.diff_degree

    if end == []:
      end = self.data_len
    elif end > self.data_len:
      end = self.data_len

    plt.subplot(2,1,1)
    plt.plot(range(start_from-self.window_len,end),
             self.trainData[start_from-self.window_len-self.diff_degree:end-self.diff_degree],'b',
             range(start_from,end),self.TrendSum[start_from-self.window_len-self.diff_degree:
                                                 end-self.window_len-self.diff_degree],'g--',
             range(start_from,end),self.T_pre[start_from-self.window_len-self.diff_degree:
                                              end-self.window_len-self.diff_degree],'r--')
    plt.legend(['real','trend','pred'])
    plt.grid('on')

    plt.subplot(2,1,2)
    plt.plot(range(start_from-self.window_len-self.diff_degree,end),
             self.realData[start_from-self.window_len-self.diff_degree:end],'b',
             range(start_from,end),self.predData[start_from-self.window_len-self.diff_degree:
                                                 end-self.window_len-self.diff_degree],'r--')
    plt.legend(['real','pred'])
    plt.grid('on')    

  def slices_plot(self,which=1):
    which -= 1
    plt.plot(range(which,which+self.window_len),self.slicesX[which],'b',
             which+self.window_len,self.slicesT[which],'bo',
             range(which,which+self.window_len),np.dot(self.TrendData[which],self.window_pattern),'g--',
             which+self.window_len,self.TrendSum[which],'go',
             which+self.window_len,self.T_pre[which],'rx')
    plt.legend(['real','real','trend','trend','pred'])
    plt.grid('on')
    
  def profit(self,start_from = []):
    if start_from == []:
      start_from = self.validate_len
    if start_from < self.window_len+self.diff_degree:
      start_from = self.window_len+self.diff_degree
    self.start_from = start_from
    realchange = np.diff(self.realData[start_from-1:])
    predchange = self.predData[start_from-self.window_len-self.diff_degree:]-self.realData[start_from-1:-1]
    profit = realchange*predchange*10000
    self.total_profit = profit.sum()
    self.daily_profit = self.total_profit/len(profit)
    self.maximum_drawdown = 0
    temp_maximum_drawdown = 0
    for i in range(len(profit)):
      temp_maximum_drawdown = min(0,profit[i]+temp_maximum_drawdown)
      if temp_maximum_drawdown<self.maximum_drawdown:
        self.maximum_drawdown = temp_maximum_drawdown

  def __repr__(self):
    if self.metrics == 'mse':
      self.mse = np.sum((self.predData[self.validate_len-self.window_len-self.diff_degree:]
                         -self.realData[self.validate_len:])**2)/(self.sample_len-self.validate_len)
      return '\n'.join([f'The model uses {self.loss_func}',
                       f'MSE on test set is {self.mse}'])
    elif self.metrics == 'profit':
      total_profit = '{:.4f}'.format(self.total_profit)
      daily_profit = '{:.4f}'.format(self.daily_profit)
      maximum_drawdown = '{:.4f}'.format(self.maximum_drawdown)
      return '\n'.join([
            f'The model uses {self.loss_func}',
            f'Backtest from date {self.start_from}',
            f'Total profit is : {total_profit}%',
            f'Daily profit is : {daily_profit}%',
            f'Maximum drawdown is : {maximum_drawdown}%'])
